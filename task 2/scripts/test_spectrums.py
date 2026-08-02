import os
import matplotlib.pyplot as plt
import rasterio
from rasterio.enums import Resampling
from pathlib import Path
import numpy as np
from collections import defaultdict

def visualize_all_spectrums():
    print("🔍 Looking for images to visualize different spectral bands...")
    dataset_path = os.path.expanduser(r"~\.cache\kagglehub\datasets\isaienkov\deforestation-in-ukraine\versions\1")
    
    # 1. Find all B04 files for grouping
    all_b04 = list(Path(dataset_path).rglob("*_B04.jp2"))
    
    groups = defaultdict(list)
    for path in all_b04:
        parts = path.name.split('_')
        if len(parts) >= 2:
            tile_id = parts[0]
            date_str = parts[1]
            groups[tile_id].append((date_str, path))
            
    # 2. Find the first location with at least 2 dates
    valid_pair = None
    for tile_id, files in groups.items():
        if len(files) >= 2:
            files.sort(key=lambda x: x[0])
            valid_pair = {
                "tile_id": tile_id,
                "date_before": files[0][0],
                "path_before_b04": files[0][1],
                "date_after": files[-1][0],
                "path_after_b04": files[-1][1],
            }
            break
            
    if not valid_pair:
        print("❌ No locations with image pairs were found.")
        return

    # Main Sentinel-2 bands at 10 m resolution
    bands_to_show = ["B02", "B03", "B04", "B08"]
    
    fig, axes = plt.subplots(2, len(bands_to_show), figsize=(18, 9))
    fig.suptitle(f"Spectral comparison for location {valid_pair['tile_id']}\n"
                 f"Top row: BEFORE ({valid_pair['date_before']}) | Bottom row: AFTER ({valid_pair['date_after']})", 
                 fontsize=16)
    
    # Reduce the image size so Matplotlib does not consume all RAM
    scale_factor = 0.05 
    
    for i, band in enumerate(bands_to_show):
        # Build paths by simply replacing B04 with the target band
        path_before = str(valid_pair['path_before_b04']).replace("B04", band)
        path_after = str(valid_pair['path_after_b04']).replace("B04", band)
        
        # BEFORE
        if os.path.exists(path_before):
            with rasterio.open(path_before) as src:
                shape = (1, int(src.height * scale_factor), int(src.width * scale_factor))
                img_b = src.read(out_shape=shape, resampling=Resampling.bilinear)[0]
                img_b = np.clip(img_b / 3000.0, 0, 1)
            axes[0, i].imshow(img_b, cmap='gray')
            axes[0, i].set_title(f"BEFORE ({band})")
        else:
            axes[0, i].text(0.5, 0.5, "Missing file", ha='center', va='center')
        axes[0, i].axis('off')
        
        # AFTER
        if os.path.exists(path_after):
            with rasterio.open(path_after) as src:
                shape = (1, int(src.height * scale_factor), int(src.width * scale_factor))
                img_a = src.read(out_shape=shape, resampling=Resampling.bilinear)[0]
                img_a = np.clip(img_a / 3000.0, 0, 1)
            axes[1, i].imshow(img_a, cmap='gray')
            axes[1, i].set_title(f"AFTER ({band})")
        else:
            axes[1, i].text(0.5, 0.5, "Missing file", ha='center', va='center')
        axes[1, i].axis('off')
        
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    visualize_all_spectrums()