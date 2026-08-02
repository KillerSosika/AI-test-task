import os
import random
import torch
import numpy as np
import rasterio
from rasterio.enums import Resampling
from torch.utils.data import Dataset
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt

class PairedDeforestationDataset(Dataset):
    def __init__(self, data_dir: str, tile_size: int = 512, crops_per_pair: int = 20):
        """
        Dataset that finds B04 files, groups them by location (tile),
        sorts them by date, and creates pairs (oldest -> newest).
        It also automatically fetches the B08 band.
        """
        self.data_dir = Path(data_dir)
        self.tile_size = tile_size
        self.crops_per_pair = crops_per_pair
        
        # 1. Find all B04 files
        all_b04_paths = list(self.data_dir.rglob("*_B04.jp2"))
        
        # 2. Group by tiles.
        # File name looks like: T36UYA_20160212T084052_B04.jp2
        grouped_files = defaultdict(list)
        for path in all_b04_paths:
            filename = path.name
            parts = filename.split('_')
            if len(parts) >= 2:
                tile_id = parts[0]      # Example: T36UYA
                date_str = parts[1]     # Example: 20160212T084052
                month = int(date_str[4:6])
                if month in [12, 1, 2, 3]:
                    continue
                grouped_files[tile_id].append((date_str, path))
                
        # 3. Create BEFORE and AFTER pairs
        self.pairs = []
        for tile_id, files in grouped_files.items():
            if len(files) < 2:
                continue # Skip if there is only one image for the location
                
            # Sort by date (oldest to newest)
            files.sort(key=lambda x: x[0])
            
            # Take the oldest (BEFORE) and newest (AFTER)
            date_before, path_before_b04 = files[0]
            date_after, path_after_b04 = files[-1]
            
            # Generate B08 paths by replacing the suffix
            path_before_b08 = Path(str(path_before_b04).replace("_B04.jp2", "_B08.jp2"))
            path_after_b08 = Path(str(path_after_b04).replace("_B04.jp2", "_B08.jp2"))
            
            # Add to the dataset only if all 4 files exist
            if path_before_b08.exists() and path_after_b08.exists():
                self.pairs.append({
                    "tile_id": tile_id,
                    "date_before": date_before,
                    "date_after": date_after,
                    "before_b04": path_before_b04,
                    "before_b08": path_before_b08,
                    "after_b04": path_after_b04,
                    "after_b08": path_after_b08
                })

        print(f"✅ Found {len(self.pairs)} unique locations with BEFORE/AFTER pairs.")

    def __len__(self):
        # Multiply the number of locations by the number of crops from one large image
        return len(self.pairs) * self.crops_per_pair

    def __getitem__(self, idx):
        pair_idx = idx // self.crops_per_pair
        pair = self.pairs[pair_idx]
        
        # Read the files
        with rasterio.open(pair["before_b04"]) as src_b:
            img_b = src_b.read(1)
        with rasterio.open(pair["after_b04"]) as src_a:
            img_a = src_a.read(1)
            
        # Find ROI (using your get_smart_roi_crops)
        from src.data.roi import get_smart_roi_crops
        valid_crops = get_smart_roi_crops(img_b, img_a, tile_size=self.tile_size, num_crops=1)
        
        if not valid_crops:
            # If a bad tile shows up, return zero tensors (or perform a recursive search)
            return {"image0": torch.zeros((1, self.tile_size, self.tile_size)), 
                    "image1": torch.zeros((1, self.tile_size, self.tile_size))}
            
        x, y = valid_crops[0]
        
        crop_b = img_b[y:y+self.tile_size, x:x+self.tile_size]
        crop_a = img_a[y:y+self.tile_size, x:x+self.tile_size]
        
        # Normalization and CLAHE
        import cv2
        def prep(img):
            img_8bit = np.clip(img / 3000.0 * 255.0, 0, 255).astype(np.uint8)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            img_clahe = clahe.apply(img_8bit)
            return torch.from_numpy(img_clahe.astype(np.float32) / 255.0).unsqueeze(0)
            
        return {
            "image0": prep(crop_b),
            "image1": prep(crop_a)
        }

# ==========================================
# VISUALIZATION BLOCK (runs only directly)
# ==========================================
def verify_dataset_groups():
    print("🔍 Starting visual verification of image pairs...")
    dataset_path = os.path.expanduser(r"~\.cache\kagglehub\datasets\isaienkov\deforestation-in-ukraine\versions\1")
    
    dataset = PairedDeforestationDataset(dataset_path)
    
    if len(dataset.pairs) == 0:
        print("❌ No pairs found. Check the file structure.")
        return

    # Show the first 3 unique locations (or fewer if there are few)
    num_to_show = min(3, len(dataset.pairs))
    
    fig, axes = plt.subplots(num_to_show, 2, figsize=(12, 5 * num_to_show))
    if num_to_show == 1:
        axes = [axes] # Keep indexing consistent for a single row
        
    fig.suptitle("Group formation check (Red band B04 only)", fontsize=16)

    for i in range(num_to_show):
        pair = dataset.pairs[i]
        
        # Load thumbnails (1/20 of the original size) to avoid exhausting memory
        scale_factor = 0.05 
        
        with rasterio.open(pair["before_b04"]) as src:
            thumbnail_shape = (1, int(src.height * scale_factor), int(src.width * scale_factor))
            img_before = src.read(out_shape=thumbnail_shape, resampling=Resampling.bilinear)[0]
            
        with rasterio.open(pair["after_b04"]) as src:
            thumbnail_shape = (1, int(src.height * scale_factor), int(src.width * scale_factor))
            img_after = src.read(out_shape=thumbnail_shape, resampling=Resampling.bilinear)[0]

        # Normalize for display
        img_before = np.clip(img_before / 3000.0, 0, 1)
        img_after = np.clip(img_after / 3000.0, 0, 1)

        axes[i][0].imshow(img_before, cmap='gray')
        axes[i][0].set_title(f"BEFORE: {pair['tile_id']} ({pair['date_before']})")
        axes[i][0].axis('off')

        axes[i][1].imshow(img_after, cmap='gray')
        axes[i][1].set_title(f"AFTER: {pair['tile_id']} ({pair['date_after']})")
        axes[i][1].axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    verify_dataset_groups()