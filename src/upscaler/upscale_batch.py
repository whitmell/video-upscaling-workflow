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
        # self._load_model()
        self.last_image_memory = 100  # Default estimate in MB
        
        # Enable OpenCV optimizations
        cv2.setUseOptimized(True)
        if cv2.useOptimized():
            print("OpenCV optimizations enabled")
        else:
            print("OpenCV optimizations not available")
    
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
        """Simplified tensor conversion with explicit stream management"""
        img = img[:, :, ::-1]  # flip RGB to BGR
        img = np.transpose(img, (2, 0, 1))  # HWC to CHW
        img = np.ascontiguousarray(img) / 255  # Rescale to [0, 1]
        
        # Create CPU tensor first
        tensor = torch.from_numpy(img).unsqueeze(0).float()
        
        # Transfer to GPU with non-blocking operation using current stream
        with torch.cuda.stream(torch.cuda.current_stream()):
            tensor = tensor.cuda(non_blocking=True)
        
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
            with torch.cuda.stream(torch.cuda.Stream()):
                # Create a detached copy to avoid in-place modification error
                detached_tensor = output_img.detach()
                
                # Extract the tensor data without in-place operations
                np_img = detached_tensor.squeeze(0).permute(1, 2, 0).float().clamp(0, 1).cpu().numpy()
                
                # Convert BGR to RGB and scale to 8-bit in one operation
                np_img = (np_img[:, :, ::-1] * 255.0).astype(np.uint8)
                
                # Use compression settings optimized for speed
                if output_path.lower().endswith(('.png')):
                    # Fastest PNG encoding (compression level 1)
                    cv2.imwrite(output_path, np_img, [cv2.IMWRITE_PNG_COMPRESSION, 1])
                elif output_path.lower().endswith(('.jpg', '.jpeg')):
                    # Fast JPEG encoding (95% quality)
                    cv2.imwrite(output_path, np_img, [cv2.IMWRITE_JPEG_QUALITY, 95])
                else:
                    cv2.imwrite(output_path, np_img)
                
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
        """Batch processing with true GPU batching"""
        total_start = time.time()
        
        # Phase 1: Loading images
        load_start = time.time()
        results = []
        input_tensors = []
        paths = []
        
        with ThreadPoolExecutor(max_workers=8) as io_pool:
            load_futures = {
                io_pool.submit(cv2.imread, path, cv2.IMREAD_UNCHANGED): path 
                for path in batch if self._is_image_file(path) and not os.path.isdir(path)
            }
            
            for future in as_completed(load_futures):
                path = load_futures[future]
                try:
                    img = future.result()
                    if img is None:
                        continue
                    
                    # Convert to tensor but don't process yet
                    conversion_start = time.time()
                    tensor = self.pil_image_to_torch_bgr(img)
                    print(f"Tensor conversion took: {time.time() - conversion_start:.4f}s")
                    
                    input_tensors.append(tensor)
                    paths.append(path)
                except Exception as e:
                    print(f"Error processing {path}: {e}")
        
        load_time = time.time() - load_start
        print(f"Loading {len(input_tensors)} images took: {load_time:.4f}s")
        
        # Phase 2: GPU processing
        gpu_start = time.time()
        outputs = self.process_batch_on_gpu(input_tensors)
        gpu_time = time.time() - gpu_start
        print(f"GPU processing took: {gpu_time:.4f}s")
        
        # Phase 3: Saving results
        save_start = time.time()
        with ThreadPoolExecutor(max_workers=16) as save_pool:  # Increased from 8 to 16
            save_futures = []
            for output, orig_path in zip(outputs, paths):
                base_name = os.path.basename(orig_path)
                out_path = os.path.join(output_dir, base_name)
                save_futures.append(
                    save_pool.submit(self.save_image, output, out_path, orig_path, move)
                )
            
            for future in as_completed(save_futures):
                future.result()
        
        save_time = time.time() - save_start
        print(f"Saving results took: {save_time:.4f}s")
        
        total_time = time.time() - total_start
        print(f"Total batch processing time: {total_time:.4f}s")
        print(f"Time breakdown: Load {load_time/total_time*100:.1f}% | GPU {gpu_time/total_time*100:.1f}% | Save {save_time/total_time*100:.1f}%")
        
        # Add explicit synchronization before returning to ensure all CUDA operations complete
        torch.cuda.synchronize()
        
        return len(outputs)

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
        """Process multiple images in a single GPU pass with proper synchronization"""
        if not tensors:
            return []
        
        # Stack all tensors into a single batch
        stacked_batch = torch.cat(tensors, dim=0)
        
        # Process the entire batch and properly time GPU execution
        start = time.time()
        with torch.no_grad():
            output = self.model(stacked_batch)
            # This ensures we wait for the operation to complete
            torch.cuda.current_stream().synchronize()
        end = time.time()
        print(f"Actual GPU execution took {end-start:.4f} seconds")
        
        # Split the batch back into individual results (already synchronized)
        return torch.split(output, 1)

    ### PROCESS FILES SEQUENTIALLY ###
    def process_directory_seq(self, input_dir, output_dir, move=False):
        """Process all images sequentially, one at a time, avoiding thread pools and async operations."""
        # Reset status tracking variables
        BatchUpscaler._status = "running"
        BatchUpscaler._message = "Starting sequential processing..."
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
        BatchUpscaler._total_batches = total_files  # For status reporting
        
        if total_files == 0:
            BatchUpscaler._status = "idle"
            BatchUpscaler._message = f"No files found in {input_dir}"
            print(BatchUpscaler._message)
            return {"status": "completed", "processed": 0, "total": 0}
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"Starting sequential processing of {total_files} files...")
        BatchUpscaler._message = f"Starting sequential processing of {total_files} files..."
        
        # Custom progress bar that updates our status
        progress = tqdm(total=total_files, desc="Processing images")
        
        processed_count = 0
        errors = []
        
        # Process each file individually
        for file_idx, file_path in enumerate(files):
            BatchUpscaler._current_batch = file_idx + 1  # Reusing batch counter for individual files
            BatchUpscaler._message = f"Processing file {file_idx + 1}/{total_files}"
            
            if os.path.isdir(file_path) or not self._is_image_file(file_path):
                progress.update(1)
                continue
                
            try:
                # Process single image
                success = self._process_single_image(file_path, output_dir, move)
                if success:
                    processed_count += 1
                    BatchUpscaler._processed_files += 1
                progress.update(1)
            except Exception as e:
                error_msg = f"Error processing file {file_path}: {e}"
                print(error_msg)
                BatchUpscaler._message = error_msg
                errors.append({"file": file_path, "error": str(e)})
                progress.update(1)
        
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

    def _process_single_image(self, file_path, output_dir, move=False):
        """Process a single image file from start to finish."""
        total_start = time.time()
        
        # Phase 1: Load image
        load_start = time.time()
        try:
            img = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
            if img is None:
                print(f"Could not load image: {file_path}")
                return False
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return False
        
        # Convert to tensor
        conversion_start = time.time()
        tensor = self.pil_image_to_torch_bgr(img)
        conversion_time = time.time() - conversion_start
        print(f"Tensor conversion took: {conversion_time:.4f}s")
        
        load_time = time.time() - load_start
        print(f"Loading image took: {load_time:.4f}s")
        
        # Phase 2: GPU processing
        gpu_start = time.time()
        with torch.no_grad():
            output = self.model(tensor)
        gpu_time = time.time() - gpu_start
        print(f"GPU processing took: {gpu_time:.4f}s")
        
        # Phase 3: Save result
        save_start = time.time()
        base_name = os.path.basename(file_path)
        out_path = os.path.join(output_dir, base_name)
        success = self.save_image(output, out_path, file_path, move)
        save_time = time.time() - save_start
        print(f"Saving result took: {save_time:.4f}s")
        
        total_time = time.time() - total_start
        print(f"Total image processing time: {total_time:.4f}s")
        print(f"Time breakdown: Load {load_time/total_time*100:.1f}% | GPU {gpu_time/total_time*100:.1f}% | Save {save_time/total_time*100:.1f}%")
        
        # Make sure CUDA operations are complete
        torch.cuda.synchronize()
        
        return success

    def upscale_ncnn(self, input_path, output_path):
        """
        Use Real-ESRGAN NCNN Vulkan executable to process images.
        This is often faster than PyTorch for certain GPUs.
        
        Args:
            input_path: Directory containing input images
            output_path: Directory for output images
        """
        import subprocess
        import re
        import os
        import sys
        from pathlib import Path

        # Adjust these parameters for your hardware
        LOAD_THREADS = 2     # Increased from 2
        PROCESS_THREADS = 6 # Significantly increased from 4
        SAVE_THREADS = 3     # Increased from 2
        TILE_SIZE = 0     # Explicitly set tile size (was auto/0)
        
        # Reset status tracking variables
        BatchUpscaler._status = "running"
        BatchUpscaler._message = "Starting NCNN upscaling..."
        BatchUpscaler._processed_files = 0
        BatchUpscaler._start_time = time.time()
        BatchUpscaler._last_update_time = time.time()
        
        # Create output directory if it doesn't exist
        os.makedirs(output_path, exist_ok=True)
        
        # Count total files for progress tracking
        files = [f for f in os.listdir(input_path) if self._is_image_file(os.path.join(input_path, f))]
        total_files = len(files)
        BatchUpscaler._total_files = total_files
        BatchUpscaler._total_batches = 1  # NCNN processes in a single batch
        
        if total_files == 0:
            BatchUpscaler._status = "idle"
            BatchUpscaler._message = f"No files found in {input_path}"
            print(BatchUpscaler._message)
            return {"status": "completed", "processed": 0, "total": 0}
        
        print(f"Starting NCNN processing of {total_files} files...")
        BatchUpscaler._message = f"Starting NCNN processing of {total_files} files..."
        
        # Prepare command
        base_dir = Path(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
        ncnn_exec = base_dir / "resources" / "realesrgan-ncnn-vulcan" / "realesrgan-ncnn-vulkan.exe"
        model_path = "D:\\Video\\video-upscaling-workflow\\resources\\models\\upscaling"
        
        # Ensure executable exists
        if not os.path.exists(ncnn_exec):
            error_msg = f"NCNN executable not found at {ncnn_exec}"
            print(error_msg)
            BatchUpscaler._status = "error"
            BatchUpscaler._message = error_msg
            return {"status": "error", "message": error_msg}
        
        command = [
            str(ncnn_exec),
            "-i", input_path,
            "-o", output_path,
            "-m", model_path,
            "-n", "realesrgan-x4plus",
            "-g", "0",       # Use GPU 0
            "-t", str(TILE_SIZE), # Add explicit tile size
            "-j", f"{LOAD_THREADS}:{PROCESS_THREADS}:{SAVE_THREADS}",
            "-v",            # Verbose output
        ]
        
        # Regex patterns for progress tracking
        done_pattern = re.compile(r'(.+) -> (.+) done')
        percent_pattern = re.compile(r'(\d+)%')
        
        # Display command we're executing
        print(f"Executing: {' '.join(command)}")
        
        # Execute command and monitor output in real-time
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,  # Line buffering
            universal_newlines=True
        )
        
        current_file_progress = 0
        last_file = None
        
        try:
            # Read output line by line in real-time
            while process.poll() is None or process.stdout:
                line = process.stdout.readline()
                if not line:
                    if process.poll() is not None:
                        break
                    continue
                
                line = line.strip()
                # Print output directly to console to see real-time progress
                print(line, flush=True)
                
                # Check if a file was completed
                done_match = done_pattern.search(line)
                if done_match:
                    source_file = done_match.group(1)
                    dest_file = done_match.group(2)
                    last_file = os.path.basename(dest_file)
                    
                    # Increment processed files counter
                    BatchUpscaler._processed_files += 1
                    
                    # Calculate progress stats
                    progress_pct = (BatchUpscaler._processed_files / total_files) * 100
                    elapsed = time.time() - BatchUpscaler._start_time
                    files_per_sec = BatchUpscaler._processed_files / elapsed if elapsed > 0 else 0
                    remaining = (total_files - BatchUpscaler._processed_files) / files_per_sec if files_per_sec > 0 else 0
                    
                    # Update status message with comprehensive information
                    BatchUpscaler._message = (
                        f"Processing: {BatchUpscaler._processed_files}/{total_files} "
                        f"[{progress_pct:.1f}%] ({files_per_sec:.2f} img/s, "
                        f"ETA: {time.strftime('%H:%M:%S', time.gmtime(remaining))}) "
                        f"Last: {last_file}"
                    )
                    current_file_progress = 0
                    continue
                
                # Check for percentage updates on current file
                percent_match = percent_pattern.search(line)
                if percent_match:
                    current_file_progress = int(percent_match.group(1))
                    # Update status with both overall progress and current file progress
                    BatchUpscaler._message = (
                        f"Processing: {BatchUpscaler._processed_files}/{total_files} "
                        f"[{BatchUpscaler._processed_files/total_files*100:.1f}%] "
                        f"Current file: {current_file_progress}%"
                    )
            
            # Wait for process to complete
            process.wait()
            
            # Check return code
            if process.returncode != 0:
                BatchUpscaler._status = "error"
                BatchUpscaler._message = f"NCNN process exited with code {process.returncode}"
                return {"status": "error", "message": BatchUpscaler._message}
            
            BatchUpscaler._status = "idle"
            BatchUpscaler._message = f"Completed processing {BatchUpscaler._processed_files} files"
            
            return {
                "status": "completed",
                "processed": BatchUpscaler._processed_files,
                "total": total_files
            }
            
        except Exception as e:
            BatchUpscaler._status = "error"
            BatchUpscaler._message = f"Error during NCNN processing: {str(e)}"
            print(f"Error: {str(e)}")
            
            # Make sure to terminate the process if something goes wrong
            if process.poll() is None:
                process.terminate()
                process.wait()
                
            return {"status": "error", "message": str(e)}
