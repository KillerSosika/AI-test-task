import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np

class DeforestationPairDataset(Dataset):
    def __init__(self, image_paths, tiler, transform=None):
        self.image_paths = image_paths
        self.tiler = tiler
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img1 = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8) 
        
        tx = np.random.uniform(-30, 30)
        ty = np.random.uniform(-30, 30)
        angle = np.random.uniform(-5, 5)
        
        center = (img1.shape[1] // 2, img1.shape[0] // 2)
        M_rot = cv2.getRotationMatrix2D(center, angle, 1.0)
        M_rot[0, 2] += tx
        M_rot[1, 2] += ty
        
        img2 = cv2.warpAffine(img1, M_rot, (img1.shape[1], img1.shape[0]), borderMode=cv2.BORDER_REFLECT)
        
        tensor1 = torch.from_numpy(img1).permute(2, 0, 1).float() / 255.0
        tensor2 = torch.from_numpy(img2).permute(2, 0, 1).float() / 255.0
        gt_matrix = torch.from_numpy(M_rot).float()
        
        return {"image0": tensor1, "image1": tensor2, "ground_truth_M": gt_matrix}