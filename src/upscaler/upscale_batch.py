import asyncio
from concurrent.futures import ThreadPoolExecutor
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
    def __init__(self, model_path=None):
        """Initialize the BatchUpscaler with a model path."""
        current_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(os.path.dirname(current_dir))
        self.default_model_path = os.path.join(parent_dir, "resources", "models", "upscaling", "RealESRGAN_x4plus.pth")
        self.model_path = model_path or self.default_model_path
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load the AI model into memory."""
        print(f"Loading model from {self.model_path}...")
        try:
            m = spandrel.ModelLoader().load_from_file(self.model_path)
            assert isinstance(m, spandrel.ImageModelDescriptor)
            self.model = m.cuda().eval()
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
        return torch.from_numpy(img).unsqueeze(0).float().cuda()

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
        """Process a batch of images efficiently."""
        # Use a thread pool for I/O operations (loading images)
        with ThreadPoolExecutor(max_workers=8) as io_pool:
            # Load images in parallel
            load_futures = []
            for file_path in batch:
                # Skip directories and non-image files
                if os.path.isdir(file_path) or not self._is_image_file(file_path):
                    continue
                load_futures.append(io_pool.submit(cv2.imread, file_path, cv2.IMREAD_UNCHANGED))
            
            # Process images
            results = []
            valid_paths = [p for p in batch if not os.path.isdir(p) and self._is_image_file(p)]
            
            # Process each image with the model
            for i, future in enumerate(load_futures):
                try:
                    img = future.result()
                    if img is None:
                        print(f"Could not load {valid_paths[i]}")
                        continue
                    
                    # Convert and process with model
                    tensor = self.pil_image_to_torch_bgr(img)
                    output = self.process_image(tensor)
                    
                    # Calculate output path
                    base_name = os.path.basename(valid_paths[i])
                    out_path = os.path.join(output_dir, base_name)
                    
                    # Add to results for saving
                    results.append((output, out_path, valid_paths[i]))
                except Exception as e:
                    print(f"Error processing {valid_paths[i]}: {e}")
            
            # Save results in parallel
            with ThreadPoolExecutor(max_workers=8) as save_pool:
                save_futures = []
                for output, out_path, orig_path in results:
                    save_futures.append(
                        save_pool.submit(self.save_image, output, out_path, orig_path, move)
                    )
                
                # Wait for all saves to complete
                for future in save_futures:
                    future.result()
        
        return len(results)
    
    @staticmethod
    def _is_image_file(file_path):
        """Check if a file is a supported image format."""
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']
        return any(file_path.lower().endswith(ext) for ext in image_extensions)