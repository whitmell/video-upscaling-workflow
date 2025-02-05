import spandrel 
from torchvision import transforms
import torch
from PIL import Image
import os
import cv2
import numpy as np
from tqdm import tqdm
from .image import FrameDataset
from torch.utils.data import DataLoader
from concurrent.futures import ThreadPoolExecutor

# Define collate_fn as a top-level function
def collate_fn(batch):
    return batch

# Function to load the image
def load_image(image_path):
    return Image.open(image_path).convert("RGB")

# Function to perform inference using the RealESRGAN_x4plus model
def perform_inference(model, image):
    # Define the transformation to preprocess the image
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.unsqueeze(0))  # Add batch dimension
    ])
    
    # Preprocess the image
    input_tensor = transform(image).cuda()
    
    # Perform inference
    output_tensor = model(input_tensor)
    
    # Postprocess the output
    output_image = transforms.ToPILImage()(output_tensor.squeeze(0))
    return output_image

def process(image: torch.Tensor, model) -> torch.Tensor:
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

def process_single(path, model):

    image = cv2.imread(path, cv2.IMREAD_UNCHANGED)       
    image_tensor = pil_image_to_torch_bgr(image)

    output_img = process(image_tensor, model)
    
    return [(output_img, path)]


def process_batch(batch, model):
    images = []
    paths = []

    for path in batch:
         
        image = cv2.imread(path, cv2.IMREAD_UNCHANGED)       
        image_tensor = pil_image_to_torch_bgr(image)
        images.append(image_tensor)
        paths.append(path)


    results = []
    for i in range(len(paths)):
        output_img = process(images[i], model)
        results.append((output_img, paths[i]))
    
    return results

def save_image(result, output_dir):
    output_img, path = result
    base_name = os.path.basename(path)
    out_path = os.path.join(output_dir, base_name)
    try:
        if output_img.dtype in (np.float32, np.float64):
            output_img = (output_img * 255.0).clip(0, 255).astype(np.uint8)
        output_img = torch.flip(output_img, dims=[1])
        pil_img = torch_bgr_to_pil_image(output_img)
        pil_img.save(out_path)
    except Exception as e:
        print(f"Error saving image {out_path}: {e}")



# use the model
    
def process_dir(input_dir, output_dir, model="D:\\Video\\Models\\upscale\\RealESRGAN_x4plus.pth"):

    model = load_model(model)

    if model is None:
        print("Model is not loaded. Exiting...")
        return
    
    total_files = len([f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))])
    
    print(f"Starting batch processing of {total_files} files...")
    progress_bar = tqdm(total=total_files, desc="Processing images")
    
    count = 1

    dataset = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
    for img in dataset:
        results = process_single(img, model)
        for result in results:
            save_image(result, output_dir)
            progress_bar.update(1)
        print(f"Processed image {count}")
        count += 1
    
    progress_bar.close()
    print("Processing complete!")

def process_dataset(dataset: FrameDataset, output_dir, model="D:\\Video\\Models\\upscale\\RealESRGAN_x4plus.pth"):

    model = load_model(model)

    if model is None:
        print("Model is not loaded. Exiting...")
        return
    
    dataloader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=8, pin_memory=True, collate_fn=collate_fn)
    total_files = len(dataset)
    
    print(f"Starting batch processing of {total_files} files...")
    progress_bar = tqdm(total=total_files, desc="Processing images")
    
    count = 1
    with ThreadPoolExecutor(max_workers=4) as executor:
        for batch in dataloader:
            results = process_batch(batch, model)
            for result in results:
                executor.submit(save_image, result, output_dir)
                progress_bar.update(1)
            print(f"Processed batch {count}")
            count += 1
    
    progress_bar.close()
    print("Processing complete!")

def load_model(model_path):
    # load the model
    m = spandrel.ModelLoader().load_from_file(model_path)
    assert isinstance(m, spandrel.ImageModelDescriptor)
    m.cuda().eval()
    return m

# spandrel_extra_arches.install()
# # load a model from disk
# model = spandrel.ModelLoader()

# # make sure it's an image to image model
# assert isinstance(model, spandrel.ImageModelDescriptor)

# # send it to the GPU and put it in inference mode
# model.cuda().eval()
