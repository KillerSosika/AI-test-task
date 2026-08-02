import os
import random
import torch
import numpy as np
import rasterio
import cv2
from torch.utils.data import Dataset
from src.finetune.supervision import generate_loftr_supervision

class SatelliteLoFTRDataset(Dataset):
    def __init__(self, data_dir, pair_list, coarse_scale=8, crop_size=512):
        self.data_dir = data_dir
        self.pairs = pair_list
        self.coarse_scale = coarse_scale
        self.crop_size = crop_size

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        pair = self.pairs[idx]

        with rasterio.open(pair["before_b04"]) as src:
            img0_full = src.read(1).astype(np.float32)
        with rasterio.open(pair["after_b04"]) as src:
            img1_full = src.read(1).astype(np.float32)

        h, w = img0_full.shape
        
        # 1. DYNAMIC CROP: each epoch selects a random crop region
        if h > self.crop_size and w > self.crop_size:
            y = random.randint(0, h - self.crop_size)
            x = random.randint(0, w - self.crop_size)
            img0 = img0_full[y:y+self.crop_size, x:x+self.crop_size]
            img1 = img1_full[y:y+self.crop_size, x:x+self.crop_size]
        else:
            img0, img1 = img0_full, img1_full

        # 2. DYNAMIC AUGMENTATION: random flips for better generalization
        if random.random() > 0.5:
            img0 = cv2.flip(img0, 1)  # horizontal
            img1 = cv2.flip(img1, 1)
        if random.random() > 0.5:
            img0 = cv2.flip(img0, 0)  # vertical
            img1 = cv2.flip(img1, 0)

        # Normalize for pseudo-ground-truth generation (SIFT operates on 8-bit images)
        img0_8u = np.clip(img0 / 3000.0 * 255.0, 0, 255).astype(np.uint8)
        img1_8u = np.clip(img1 / 3000.0 * 255.0, 0, 255).astype(np.uint8)

        # Generate the ground-truth matrix after crop and augmentation
        supervision = generate_loftr_supervision(img0_8u, img1_8u, self.coarse_scale)

        # Convert to tensor format expected by the transformer [0, 1]
        t_img0 = torch.from_numpy(img0 / 3000.0).unsqueeze(0).clamp(0, 1)
        t_img1 = torch.from_numpy(img1 / 3000.0).unsqueeze(0).clamp(0, 1)

        sample = {
            "image0": t_img0,
            "image1": t_img1,
            **supervision
        }
        return sample