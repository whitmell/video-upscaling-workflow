import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
import glob
import io
import os
import cv2
import json
import numpy as np
import spandrel
import time
import torch
from PIL import Image
from tqdm import tqdm

class BatchUpscaler:
    # Class variables for status tracking
    _status = "idle"
    _message = ""
    _current_batch = 0
    _total_batches = 0
    _processed_files = 0
    _total_files = 0
    _start_time = None
    _last_update_time = None

    def __init__(self, model_path=None, num_streams=4):
        """Initialize the BatchUpscaler with a model path and multiple CUDA streams."""
        current_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(os.path.dirname(current_dir))
        self.default_model_path = os.path.join(parent_dir, "resources", "models", "upscaling", "RealESRGAN_x4plus.pth")
        self.model_path = model_path or self.default_model_path
        self.model = None
        self.streams = [torch.cuda.Stream() for _ in range(num_streams)]
        self.use_half_precision = torch.cuda.is_available()  # Flag to track precision mode
        self._load_model()
        self.last_image_memory = 100  # Default estimate in MB
    
    def _load_model(self):
        print(f"Loading model from {self.model_path}...")
        try:
            m = spandrel.ModelLoader().load_from_file(self.model_path)
            assert isinstance(m, spandrel.ImageModelDescriptor)
            self.model = m.cuda().eval()
            self.use_half_precision = False  # Try without half precision first
            print("Model loaded successfully")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise

    def unload_model(self):
        """Explicitly unload the model and clear CUDA memory."""
        if self.model is not None:
            del self.model
            torch.cuda.empty_cache()
            self.model = None
            print("Model unloaded and CUDA memory cleared")
    
    def __del__(self):
        """Ensure model is unloaded when the object is garbage collected."""
        self.unload_model()

    def pil_image_to_torch_bgr(self, img):
        """Simplified tensor conversion"""
        img = img[:, :, ::-1]  # flip RGB to BGR
        img = np.transpose(img, (2, 0, 1))  # HWC to CHW
        img = np.ascontiguousarray(img) / 255  # Rescale to [0, 1]
        
        # Simpler GPU transfer, no pin_memory overhead
        tensor = torch.from_numpy(img).unsqueeze(0).float().cuda()
        return tensor

    @staticmethod
    def torch_bgr_to_pil_image(tensor):
        if tensor.ndim == 4:
            if tensor.shape[0] != 1:
                raise ValueError(f"{tensor.shape} does not describe a BCHW tensor")
            tensor = tensor.squeeze(0)
        assert tensor.ndim == 3, f"{tensor.shape} does not describe a CHW tensor"
        arr = tensor.float().cpu().clamp_(0, 1).numpy()
        arr = 255.0 * np.moveaxis(arr, 0, 2)  # CHW to HWC, rescale
        arr = arr.round().astype(np.uint8)
        arr = arr[:, :, ::-1]  # flip BGR to RGB
        return Image.fromarray(arr, "RGB")

    def process_image(self, image):
        """Process a single image through the model."""
        with torch.no_grad():
            return self.model(image)

    def save_image(self, output_img, output_path, original_path=None, move=False):
        """Save the processed image and optionally move the original."""
        try:
            if output_img.dtype in (np.float32, np.float64):
                output_img = (output_img * 255.0).clip(0, 255).astype(np.uint8)
            
            output_img = torch.flip(output_img, dims=[1])
            pil_img = self.torch_bgr_to_pil_image(output_img)
            pil_img.save(output_path)
            
            if move and original_path:
                # Move original to processed directory
                processed_dir = os.path.join(os.path.dirname(os.path.dirname(output_path)), "processed")
                os.makedirs(processed_dir, exist_ok=True)
                processed_path = os.path.join(processed_dir, os.path.basename(original_path))
                os.rename(original_path, processed_path)
                
            return True
        except Exception as e:
            print(f"Error saving image {output_path}: {e}")
            return False

    def get_status(self):
        """Get the current processing status as a JSON-serializable dict."""
        if self._status == "running" and self._start_time and self._processed_files > 0:
            elapsed = time.time() - self._start_time
            files_per_sec = self._processed_files / elapsed if elapsed > 0 else 0
            remaining_files = self._total_files - self._processed_files
            eta_seconds = remaining_files / files_per_sec if files_per_sec > 0 else 0
            
            # Format time as HH:MM:SS
            eta_formatted = time.strftime('%H:%M:%S', time.gmtime(eta_seconds))
            
            message = f"Processing: {self._processed_files}/{self._total_files} " \
                      f"[{self._processed_files/self._total_files:.1%}] " \
                      f"({files_per_sec:.2f} img/s, ETA: {eta_formatted})"
            
            # Update the message
            self._message = message
        
        return {
            "status": self._status,
            "message": self._message,
            "progress": {
                "current_batch": self._current_batch,
                "total_batches": self._total_batches,
                "processed_files": self._processed_files,
                "total_files": self._total_files,
                "percent_complete": round(self._processed_files / self._total_files * 100, 1) if self._total_files > 0 else 0
            }
        }
    
    async def process_directory(self, input_dir, output_dir, move=False, batch_size=16):
        """Process all images in a directory with batching and efficient resource use."""
        # Reset status tracking variables
        BatchUpscaler._status = "running"
        BatchUpscaler._message = "Starting processing..."
        BatchUpscaler._processed_files = 0
        BatchUpscaler._start_time = time.time()
        BatchUpscaler._last_update_time = time.time()
        
        # Check if model is loaded
        if self.model is None:
            self._load_model()
        
        # Get all files in input directory
        files = glob.glob(os.path.join(input_dir, '*'))
        total_files = len(files)
        BatchUpscaler._total_files = total_files
        
        if total_files == 0:
            BatchUpscaler._status = "idle"
            BatchUpscaler._message = f"No files found in {input_dir}"
            print(BatchUpscaler._message)
            return {"status": "completed", "processed": 0, "total": 0}
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Optimize batch size based on GPU memory
        optimal_batch_size = self.get_optimal_batch_size()
        batch_size = min(batch_size, optimal_batch_size)
        
        # Create batches
        batches = [files[i:i+batch_size] for i in range(0, len(files), batch_size)]
        BatchUpscaler._total_batches = len(batches)
        
        print(f"Starting batch processing of {total_files} files...")
        BatchUpscaler._message = f"Starting batch processing of {total_files} files..."
        
        # Custom progress bar that updates our status
        progress = tqdm(total=total_files, desc="Processing images")
        
        processed_count = 0
        errors = []
        
        # Process each batch
        for batch_idx, batch in enumerate(batches):
            BatchUpscaler._current_batch = batch_idx + 1
            BatchUpscaler._message = f"Processing batch {batch_idx + 1}/{len(batches)}"
            
            try:
                processed = await self._process_batch(batch, output_dir, move)
                processed_count += processed
                BatchUpscaler._processed_files += processed
                progress.update(len(batch))
            except Exception as e:
                error_msg = f"Error processing batch {batch_idx}: {e}"
                print(error_msg)
                BatchUpscaler._message = error_msg
                errors.append({"batch": batch_idx, "error": str(e)})
        
        progress.close()
        
        # Update status
        BatchUpscaler._status = "idle"
        BatchUpscaler._message = f"Completed processing {processed_count} files" + (
            " with errors" if errors else ""
        )
        
        return {
            "status": "completed" if not errors else "completed_with_errors",
            "processed": processed_count,
            "total": total_files,
            "errors": errors
        }
        
    def get_optimal_batch_size(self):
        """Determine optimal batch size based on image size and available VRAM"""
        # Get free VRAM in bytes
        free_vram, total_vram = torch.cuda.mem_get_info()
        free_vram_mb = free_vram / (1024 * 1024)
        
        # Estimate memory per image based on recent processing
        if hasattr(self, 'last_image_memory') and self.last_image_memory > 0:
            mem_per_image_mb = self.last_image_memory
        else:
            # Default conservative estimate (100MB per 1080p image)
            mem_per_image_mb = 100
        
        # Reserve 20% of free memory for overhead
        usable_memory_mb = free_vram_mb * 0.8
        
        # Calculate how many images we can fit
        max_batch_size = max(1, int(usable_memory_mb / mem_per_image_mb))
        
        # Cap at a reasonable maximum
        return min(16, max_batch_size)
    
    async def _process_batch(self, batch, output_dir, move=False):
        """Simplified batch processing"""
        results = []
        
        # Load and process images directly, no complex prefetching
        with ThreadPoolExecutor(max_workers=8) as io_pool:  # Reduced from 16
            # Load images
            load_futures = {
                io_pool.submit(cv2.imread, path, cv2.IMREAD_UNCHANGED): path 
                for path in batch if self._is_image_file(path) and not os.path.isdir(path)
            }
            
            # Process each image as it completes loading
            for future in as_completed(load_futures):
                path = load_futures[future]
                try:
                    img = future.result()
                    if img is None:
                        continue
                        
                    # Process individual image
                    tensor = self.pil_image_to_torch_bgr(img)
                    output = self.process_image(tensor)
                    
                    # Save output path
                    base_name = os.path.basename(path)
                    out_path = os.path.join(output_dir, base_name)
                    results.append((output, out_path, path))
                except Exception as e:
                    print(f"Error processing {path}: {e}")
        
        # Save all results
        with ThreadPoolExecutor(max_workers=8) as save_pool:  # Reduced from 16
            save_futures = []
            for output, out_path, orig_path in results:
                save_futures.append(
                    save_pool.submit(self.save_image, output, out_path, orig_path, move)
                )
            
            for future in as_completed(save_futures):
                future.result()
        
        return len(results)

    async def _prefetch_images(self, batch, queue, done_event):
        """Prefetch and load images in background"""
        with ThreadPoolExecutor(max_workers=16) as pool:  # Increased for i9-14900HX
            # Submit all load operations
            futures = []
            valid_paths = []
            for file_path in batch:
                if os.path.isdir(file_path) or not self._is_image_file(file_path):
                    continue
                futures.append(pool.submit(cv2.imread, file_path, cv2.IMREAD_UNCHANGED))
                valid_paths.append(file_path)
            
            # As results complete, add them to the queue
            for i, future in enumerate(as_completed(futures)):
                try:
                    img = future.result()
                    await queue.put((img, valid_paths[i]))
                except Exception as e:
                    print(f"Error loading {valid_paths[i]}: {e}")
                    await queue.put((None, valid_paths[i]))
        
        # Signal that all images have been loaded
        done_event.set()
    
    @staticmethod
    def _is_image_file(file_path):
        """Check if a file is a supported image format."""
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']
        return any(file_path.lower().endswith(ext) for ext in image_extensions)

    def process_batch_on_gpu(self, tensors):
        """Process multiple images in a single GPU pass"""
        # Stack all tensors into a single batch
        if not tensors:
            return []
        
        stacked_batch = torch.cat(tensors, dim=0)
        
        # Process the entire batch at once
        start = time.time()
        with torch.no_grad():
            output = self.model(stacked_batch)
        end = time.time()
        print(f"Operation took {end-start:.4f} seconds")
        
        # Split the batch back into individual results
        return torch.split(output, 1)
