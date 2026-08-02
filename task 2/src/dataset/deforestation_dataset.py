import os
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
import kagglehub
import rasterio

class DeforestationDataset(Dataset):
    """
    Handles the 36GB 'deforestation-in-ukraine' Kaggle dataset.
    Downloads data automatically and reads heavy .jp2 files efficiently.
    """
    def __init__(self, use_cache: bool = True, transform=None):
        """
        Args:
            use_cache (bool): If True, uses kagglehub to locate or download the dataset.
            transform: Optional torchvision transforms.
        """
        print("[INFO] Checking dataset via kagglehub...")
        self.data_dir = Path(kagglehub.dataset_download("isaienkov/deforestation-in-ukraine"))
        print(f"[INFO] Dataset located at: {self.data_dir}")
        
        self.transform = transform
        
        # Parse all .jp2 images in the dataset
        self.image_paths = sorted(list(self.data_dir.rglob("*.jp2")))
        
        # Optional: Parse .gml or .xml for Ground Truth ROIs (to be implemented)
        self.gml_paths = sorted(list(self.data_dir.rglob("*.gml")))
        
        print(f"[INFO] Found {len(self.image_paths)} JP2 images and {len(self.gml_paths)} GML annotations.")

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> dict:
        img_path = str(self.image_paths[idx])
        
        # Read .jp2 efficiently using rasterio
        with rasterio.open(img_path) as src:
            # Read only the first (and only) channel
            img_array = src.read(1) 
            
            # Now we have an array of shape (H, W).
            # Expand it to (H, W, 1) so the network does not complain about a missing channel
            img_array = np.expand_dims(img_array, axis=-1)
            
            # Safe normalization (satellite images are often uint16)
            if img_array.dtype.kind in 'ui':
                max_val = np.iinfo(img_array.dtype).max
                img_array = (img_array / max_val).astype(np.float32)
            else:
                img_array = img_array.astype(np.float32)
            
        if self.transform:
            img_array = self.transform(img_array)
            
        return {
            "image": img_array,
            "path": img_path
        }
if __name__ == "__main__":
    print("🚀 Starting Dataset Initialization & Download...")
    dataset = DeforestationDataset()
    
    if len(dataset) > 0:
        print(f"✅ Total images found: {len(dataset)}")
        print("⏳ Loading first image into memory (Lazy Load Test)...")
        sample = dataset[0]
        
        print(f"🖼️ First image path: {sample['path']}")
        print(f"📊 Image shape: {sample['image'].shape}")
        print("🎉 Successfully loaded the first image without crashing RAM!")
    else:
        print("❌ Dataset is empty or not found.")