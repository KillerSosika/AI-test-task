import os
import sys
from pathlib import Path
import rasterio
import numpy as np

# Add the project root to sys.path so imports from src/ work correctly
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.dataset import PairedDeforestationDataset
from src.data.roi import get_smart_roi_crops
from src.matching.loftr_matcher import LoftrMatcher
from src.visualization.visualizer import Visualizer

def main():
    print("🚀 Starting the ML pipeline for deforestation detection...")

    # 1. Data and weight paths
    data_dir = os.path.expanduser(r"~\.cache\kagglehub\datasets\isaienkov\deforestation-in-ukraine\versions\1")
    weights_path = r"weights\loftr_satellite_dense_finetuned.pth" 

    # 2. Initialize components
    print("📦 Loading dataset structure...")
    dataset = PairedDeforestationDataset(data_dir=data_dir)

    if len(dataset.pairs) == 0:
        print("❌ No image pairs were found for processing. Check the date filters in dataset.py.")
        return

    print("🧠 Initializing the LoFTR model...")
    # If weights exist, they will be loaded; otherwise, the base version will be used
    matcher = LoftrMatcher(weights_path=weights_path if os.path.exists(weights_path) else None)
    visualizer = Visualizer()

    # 3. Take the best location for the test
    pair = dataset.pairs[0]
    print(f"🌍 Processing location: {pair['tile_id']} | BEFORE: {pair['date_before']} -> AFTER: {pair['date_after']}")

    # 4. Read full-size B04 images
    print("📸 Reading spectral data...")
    with rasterio.open(pair["before_b04"]) as src_b:
        img_before_full = src_b.read(1)
    with rasterio.open(pair["after_b04"]) as src_a:
        img_after_full = src_a.read(1)

    # 5. Find safe ROI zones (without black NoData triangles)
    tile_size = 512
    print("🔍 Searching for safe ROI zones...")
    valid_crops = get_smart_roi_crops(img_before_full, img_after_full, tile_size=tile_size, num_crops=1)

    if not valid_crops:
        print("❌ Could not find a shared NoData-free region in these images.")
        return

    x, y = valid_crops[0]
    print(f"✂️ Cropping a clean 512x512 tile at coordinates X:{x}, Y:{y}")

    # Crop the tiles
    img_before_crop = img_before_full[y:y+tile_size, x:x+tile_size]
    img_after_crop = img_after_full[y:y+tile_size, x:x+tile_size]

    # 6. Run inference through the network
    print("⚡ Matching features...")
    mkpts0, mkpts1, mconf = matcher.match(img_before_crop, img_after_crop)
    print(f"🎯 Found {len(mkpts0)} raw match points!")

    # --- ADDED CONFIDENCE FILTER ---
    threshold = 0.80  # Confidence threshold (80%). Can be tuned from 0.5 to 0.99
    
    valid_mask = mconf > threshold
    mkpts0_clean = mkpts0[valid_mask]
    mkpts1_clean = mkpts1[valid_mask]
    mconf_clean = mconf[valid_mask]
    
    print(f"🛡️ After strict filtering (>{threshold}): {len(mkpts0_clean)} points remain.")

    # 7. Visualization
    print("🎨 Generating chart...")
    visualizer.draw_matches(
        img_before_crop,
        img_after_crop,
        mkpts0_clean,  # Pass cleaned coordinates
        mkpts1_clean,  # Pass cleaned coordinates
        title=f"LoFTR Clean Matches (>{threshold}): {pair['tile_id']} | Points: {len(mkpts0_clean)}"
    )

    # 7. Visualization
    print("🎨 Generating chart...")
    visualizer.draw_matches(
        img_before_crop,
        img_after_crop,
        mkpts0,
        mkpts1,
        title=f"LoFTR Matches: {pair['tile_id']} ({pair['date_before']} vs {pair['date_after']}) | Points: {len(mkpts0)}"
    )

if __name__ == "__main__":
    main()