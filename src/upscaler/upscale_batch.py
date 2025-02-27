import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
import glob
import io
import os
import cv2
import numpy as np
import spandrel
import torch
from PIL import Image
from tqdm import tqdm

class BatchUpscaler:
    def __init__(self, model_path=None, num_streams=4):
        """Initialize the BatchUpscaler with a model path and multiple CUDA streams."""
        current_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(os.path.dirname(current_dir))
        self.default_model_path = os.path.join(parent_dir, "resources", "models", "upscaling", "RealESRGAN_x4plus.pth")
        self.model_path = model_path or self.default_model_path
        self.model = None
        self.streams = [torch.cuda.Stream() for _ in range(num_streams)]
        self._load_model()
    
    def _load_model(self):
        """Load the AI model into memory with mixed precision."""
        print(f"Loading model from {self.model_path}...")
        try:
            m = spandrel.ModelLoader().load_from_file(self.model_path)
            assert isinstance(m, spandrel.ImageModelDescriptor)
            self.model = m.cuda().eval()
            
            # Enable mixed precision
            if torch.cuda.is_available():
                self.model = self.model.half()  # Convert to FP16
                print("Model loaded with mixed precision")
            else:
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

    @staticmethod
    def pil_image_to_torch_bgr(img):
        img = img[:, :, ::-1]  # flip RGB to BGR
        img = np.transpose(img, (2, 0, 1))  # HWC to CHW
        img = np.ascontiguousarray(img) / 255  # Rescale to [0, 1]
        
        # Use pinned memory for faster CPU->GPU transfer
        tensor = torch.from_numpy(img).unsqueeze(0).float()
        tensor = tensor.pin_memory().to('cuda', non_blocking=True)
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

    async def process_directory(self, input_dir, output_dir, move=False, batch_size=16):
        """Process all images in a directory with batching and efficient resource use."""
        # Check if model is loaded
        if self.model is None:
            self._load_model()
        
        # Get all files in input directory
        files = glob.glob(os.path.join(input_dir, '*'))
        total_files = len(files)
        
        if total_files == 0:
            print(f"No files found in {input_dir}")
            return {"status": "completed", "processed": 0, "total": 0}
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Create batches
        batches = [files[i:i+batch_size] for i in range(0, len(files), batch_size)]
        
        print(f"Starting batch processing of {total_files} files...")
        progress = tqdm(total=total_files, desc="Processing images")
        
        processed_count = 0
        errors = []
        
        # Process each batch
        for batch_idx, batch in enumerate(batches):
            try:
                processed = await self._process_batch(batch, output_dir, move)
                processed_count += processed
                progress.update(len(batch))
            except Exception as e:
                print(f"Error processing batch {batch_idx}: {e}")
                errors.append({"batch": batch_idx, "error": str(e)})
        
        progress.close()
        
        return {
            "status": "completed" if not errors else "completed_with_errors",
            "processed": processed_count,
            "total": total_files,
            "errors": errors
        }
    
    async def _process_batch(self, batch, output_dir, move=False):
        """Process a batch of images with overlapping I/O and GPU operations"""
        # Create prefetch queue for next batch items
        prefetch_queue = asyncio.Queue(maxsize=16)
        loading_done = asyncio.Event()
        
        # Start prefetch worker
        asyncio.create_task(self._prefetch_images(batch, prefetch_queue, loading_done))
        
        results = []
        gpu_batch = []
        gpu_batch_paths = []
        gpu_batch_size = 4  # Process 4 images at a time on GPU
        
        # Process images as they become available
        while not loading_done.is_set() or not prefetch_queue.empty():
            try:
                img, path = await asyncio.wait_for(prefetch_queue.get(), timeout=0.1)
                if img is not None:
                    # Preprocess and add to GPU batch
                    tensor = self.pil_image_to_torch_bgr(img)
                    gpu_batch.append(tensor)
                    gpu_batch_paths.append(path)
                    
                    # Process on GPU when batch is full
                    if len(gpu_batch) >= gpu_batch_size:
                        batch_results = self.process_batch_on_gpu(gpu_batch)
                        for output, orig_path in zip(batch_results, gpu_batch_paths):
                            base_name = os.path.basename(orig_path)
                            out_path = os.path.join(output_dir, base_name)
                            results.append((output, out_path, orig_path))
                        
                        # Reset batch
                        gpu_batch = []
                        gpu_batch_paths = []
            except asyncio.TimeoutError:
                continue
        
        # Process remaining images in the batch
        if gpu_batch:
            batch_results = self.process_batch_on_gpu(gpu_batch)
            for output, orig_path in zip(batch_results, gpu_batch_paths):
                base_name = os.path.basename(orig_path)
                out_path = os.path.join(output_dir, base_name)
                results.append((output, out_path, orig_path))
        
        # Save results in parallel
        with ThreadPoolExecutor(max_workers=16) as save_pool:  # Increased workers for i9
            save_futures = []
            for output, out_path, orig_path in results:
                save_futures.append(
                    save_pool.submit(self.save_image, output, out_path, orig_path, move)
                )
            
            # Wait for all saves to complete
            for future in save_futures:
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
        with torch.no_grad():
            output = self.model(stacked_batch)
        
        # Split the batch back into individual results
        return torch.split(output, 1)