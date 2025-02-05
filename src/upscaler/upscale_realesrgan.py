# import os
# import cv2
# import torch
# import numpy as np
# from PIL import Image
# from tqdm import tqdm
# from image import FrameDataset
# from torch.utils.data import DataLoader
# from concurrent.futures import ThreadPoolExecutor
# from basicsr.archs.rrdbnet_arch import RRDBNet
# from realesrgan import RealESRGANer

# torch.backends.cudnn.benchmark = True

# # Load Real-ESRGAN model
# model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
#                 num_block=23, num_grow_ch=32, scale=4)
# upsampler = RealESRGANer(
#     scale=4,
#     model_path='/home/whit/dev/video-upscaling/models/RealESRGAN_x4plus.pth',
#     model=model,
#     tile=0,
#     tile_pad=10,
#     pre_pad=0,
#     half=True
# )

# def process_batch(batch):
#     images = []
#     paths = []

#     for img, path in batch:
#         # img_tensor = torch.from_numpy(
#         #     (torch.ByteTensor(torch.ByteStorage.from_buffer(img.tobytes()))
#         #         .view(img.size[1], img.size[0], 3)
#         #         .float() / 255.0).numpy()
#         #     ).permute(2, 0, 1).numpy()

#         # img_np = np.array(img, dtype=np.float32) / 255.0  # shape: (H, W, 3)
#         # img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0) 
         
#         image = cv2.imread(path, cv2.IMREAD_UNCHANGED)       
#         images.append(image)
#         paths.append(path)


#     results = []
#     for i in range(len(paths)):
#         output_img, _ = upsampler.enhance(images[i], outscale=4)
#         results.append((output_img, paths[i]))
    
#     return results

# def save_image(result, output_dir):
#     output_img, path = result
#     base_name = os.path.basename(path)
#     out_path = os.path.join(output_dir, base_name)
#     try:
#         if output_img.dtype in (np.float32, np.float64):
#             output_img = (output_img * 255.0).clip(0, 255).astype(np.uint8)
#         output_img = output_img[..., ::-1]
#         pil_img = Image.fromarray(output_img)
#         pil_img.save(out_path)
#     except Exception as e:
#         print(f"Error saving image {out_path}: {e}")


# def process_single_image(image_path):
#     image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    
#     image_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float()
    
#     if torch.cuda.is_available():
#         image_tensor = image_tensor.cuda()
    
#     output_img, _ = upsampler.enhance(image_tensor.cpu().numpy(), outscale=4)
#     return output_img

# def process_dataset(dataset: FrameDataset, output_dir):
#     dataloader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=8, pin_memory=True, collate_fn=lambda x: x)
#     total_files = len(dataset)
    
#     print(f"Starting batch processing of {total_files} files...")
#     progress_bar = tqdm(total=total_files, desc="Processing images")
    
#     count = 1
#     with ThreadPoolExecutor(max_workers=4) as executor:
#         for batch in dataloader:
#             results = process_batch(batch)
#             for result in results:
#                 executor.submit(save_image, result, output_dir)
#                 progress_bar.update(1)
#             print(f"Processed batch {count}")
#             count += 1
    
#     progress_bar.close()
#     print("Processing complete!")
