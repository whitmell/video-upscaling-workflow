
import asyncio
import cProfile
from concurrent.futures import ThreadPoolExecutor
import io
import os
import pstats
import cv2
import numpy as np
import spandrel
import torch
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
REALESRGAN_PATH = os.path.join(parent_dir, "resources", "models", "upscaling", "RealESRGAN_x4plus.pth")

def load_model(model_path):
    # load the model
    m = spandrel.ModelLoader().load_from_file(model_path)
    assert isinstance(m, spandrel.ImageModelDescriptor)
    m.cuda().eval()
    return m

def collate_fn(batch):
    return batch

def profile_code(func):
    def wrapper(*args, **kwargs):
        pr = cProfile.Profile()
        pr.enable()
        result = func(*args, **kwargs)
        pr.disable()
        s = io.StringIO()
        sortby = 'cumulative'
        ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        ps.print_stats()
        print(s.getvalue())
        return result
    return wrapper

async def async_load_data(dataloader, data_queue):
    for batch in dataloader:
        await data_queue.put(batch)

async def read_image_async(path, loop):
    return await loop.run_in_executor(None, cv2.imread, path, cv2.IMREAD_UNCHANGED)

def process_image(image: torch.Tensor, model) -> torch.Tensor:
    with torch.no_grad():
        return model(image)

def pil_image_to_torch_bgr(img: Image.Image) -> torch.Tensor:
    # img = np.array(img.convert("RGB"))
    img = img[:, :, ::-1]  # flip RGB to BGR
    img = np.transpose(img, (2, 0, 1))  # HWC to CHW
    img = np.ascontiguousarray(img) / 255  # Rescale to [0, 1]
    return torch.from_numpy(img).unsqueeze(0).float().cuda()

def torch_bgr_to_pil_image(tensor: torch.Tensor) -> Image.Image:
    if tensor.ndim == 4:
        # If we're given a tensor with a batch dimension, squeeze it out
        # (but only if it's a batch of size 1).
        if tensor.shape[0] != 1:
            raise ValueError(f"{tensor.shape} does not describe a BCHW tensor")
        tensor = tensor.squeeze(0)
    assert tensor.ndim == 3, f"{tensor.shape} does not describe a CHW tensor"
    # TODO: is `tensor.float().cpu()...numpy()` the most efficient idiom?
    arr = tensor.float().cpu().clamp_(0, 1).numpy()  # clamp
    arr = 255.0 * np.moveaxis(arr, 0, 2)  # CHW to HWC, rescale
    arr = arr.round().astype(np.uint8)
    arr = arr[:, :, ::-1]  # flip BGR to RGB
    return Image.fromarray(arr, "RGB")
            
def process_sub_batch(batch, model, loop):
    images = []
    paths = []

    for path in batch:
        print(path)
        future = asyncio.ensure_future(read_image_async(path, loop))
        image = asyncio.run(future)
        image_tensor = pil_image_to_torch_bgr(image)
        images.append(image_tensor)
        paths.append(path)


    results = []
    for i in range(len(paths)):
        output_img = process_image(images[i], model)
        results.append((output_img, paths[i]))

    return results

def save_image(result, output_dir, move=False):
    output_img, path = result
    base_name = os.path.basename(path)
    out_path = os.path.join(output_dir, base_name)
    try:
        if output_img.dtype in (np.float32, np.float64):
            output_img = (output_img * 255.0).clip(0, 255).astype(np.uint8)
        output_img = torch.flip(output_img, dims=[1])
        pil_img = torch_bgr_to_pil_image(output_img)
        pil_img.save(out_path)

        try:
            # Move the processed source image to the processed directory
            processed_dir = os.path.join(os.path.dirname(output_dir), "processed")
            os.makedirs(processed_dir, exist_ok=True)
            processed_path = os.path.join(processed_dir, base_name)
            os.rename(path, processed_path)
        except Exception as e:
            print(f"Error moving image {path} to {processed_path}: {e}")
    except Exception as e:
        print(f"Error saving image {out_path}: {e}")

@profile_code
async def process_batch_async(batch: list[str], output_dir, move=False):
    print("Loading model...")
    
    model = load_model(REALESRGAN_PATH)

    if model is None:
        print("Model is not loaded. Exiting...")
        return

    dataloader = DataLoader(batch, batch_size=8, shuffle=False, num_workers=0, pin_memory=True, collate_fn=collate_fn)
    loop = asyncio.get_event_loop()
    data_queue = asyncio.Queue(maxsize=4)  # Limit queue size to prevent excessive memory usage

    # Start loading data asynchronously
    asyncio.create_task(async_load_data(dataloader, data_queue))

    with ThreadPoolExecutor(max_workers=8) as save_executor, \
            ThreadPoolExecutor(max_workers=4) as move_executor:
        count = 1
        while count <= len(batch):
            try:
                batch = await asyncio.wait_for(data_queue.get(), timeout=1.0)  # Timeout to handle empty queue
                data_queue.task_done()
                results = process_sub_batch(batch, model, loop)
                for result in results:
                    save_executor.submit(save_image, result, output_dir, move)
                    if move:
                        move_executor.submit(move, result, move)
                count += 1
            except asyncio.TimeoutError:
                print("Queue is empty, waiting...")
                if data_queue.empty():
                    break

    print("Processing complete!")