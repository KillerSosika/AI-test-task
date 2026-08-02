import sys
from pathlib import Path
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt

project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

from src.dataset.pair_generator import SeasonalPairGenerator
from src.matching.loftr_matcher import DeepImageMatcher

def numpy_to_tensor(img: np.ndarray, device: str) -> torch.Tensor:
    """Converts an RGB numpy image into the tensor format required by LoFTR."""
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    tensor = torch.from_numpy(gray).float() / 255.0
    return tensor.unsqueeze(0).unsqueeze(0).to(device)

def main():
    print("🚀 Starting the inference pipeline...")
    
    data_dir = project_root / "data" / "raw"
    generator = SeasonalPairGenerator(data_dir=str(data_dir))
    matcher = DeepImageMatcher(pretrained='outdoor')
    
    img1, img2, orig_path = generator.generate_pair()
    print(f"✅ Generated pair based on: {orig_path.name}")
    
    tensor1 = numpy_to_tensor(img1, matcher.device)
    tensor2 = numpy_to_tensor(img2, matcher.device)
    
    print("🧠 Extracting features using LoFTR...")
    with torch.no_grad():
        input_dict = {"image0": tensor1, "image1": tensor2}
        correspondences = matcher.matcher(input_dict)
    
    mkpts0 = correspondences['keypoints0'].cpu().numpy()
    mkpts1 = correspondences['keypoints1'].cpu().numpy()
    
    if len(mkpts0) < 4:
        print("❌ Found too few points for matching.")
        return

    H, inliers_mask = cv2.findHomography(mkpts0, mkpts1, cv2.USAC_MAGSAC, 3.0)
    inliers_mask = inliers_mask.flatten() > 0
    
    good_matches_count = np.sum(inliers_mask)
    print(f"🎯 Points found: {len(mkpts0)} | Valid matches (Inliers): {good_matches_count}")
    
    kps1 = [cv2.KeyPoint(p[0], p[1], 1) for p in mkpts0]
    kps2 = [cv2.KeyPoint(p[0], p[1], 1) for p in mkpts1]
    
    all_matches = [cv2.DMatch(i, i, 0) for i in range(len(mkpts0))]
    
    matches_mask = [int(m) for m in inliers_mask]
    
    img_draw = cv2.drawMatches(
        img1, kps1, img2, kps2, all_matches, None,
        matchColor=(0, 255, 0),         # Green lines for Inliers
        singlePointColor=(255, 0, 0),   # Red dots for keypoints
        matchesMask=matches_mask,       
        flags=0
    )
                               
    plt.figure(figsize=(12, 6))
    plt.imshow(img_draw)
    plt.title(f"LoFTR Matching | Inliers: {good_matches_count} / {len(mkpts0)}", fontweight='bold')
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    main()