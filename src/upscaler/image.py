import glob
import os
from PIL import Image
from torch.utils.data import Dataset


class FrameDataset(Dataset):
    def __init__(self, frames_dir, start=0):
        self.img_paths = sorted(glob.glob(os.path.join(frames_dir, '*.png')))[start:]
        
    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        return img_path