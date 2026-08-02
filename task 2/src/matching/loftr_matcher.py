import torch
import numpy as np
import cv2
from kornia.feature import LoFTR
from src.matching.base_matcher import BaseMatcher

class LoftrMatcher(BaseMatcher):
    # Added flag use_custom_weights
    def __init__(self, weights_path=None, device='cuda', use_custom_weights=False):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.matcher = LoFTR(pretrained='outdoor').to(self.device)
        
        # Load your weights only if the flag is True
        if weights_path and use_custom_weights:
            print(f"🧠 Loading custom LoFTR weights: {weights_path}")
            state_dict = torch.load(weights_path, map_location=self.device, weights_only=True)
            self.matcher.load_state_dict(state_dict)
        else:
            print("🧠 Using the base LoFTR weights (outdoor) for debugging.")
            
        self.matcher.eval()

    def _prepare_image(self, img: np.ndarray) -> torch.Tensor:
        """
        Prepares a tensor by applying CLAHE to improve edges (very important for different seasons).
        """
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
        # 1. Normalize raw Sentinel data (0-3000) to 8-bit format (0-255)
        img_8bit = np.clip(img / 3000.0 * 255.0, 0, 255).astype(np.uint8)
        
        # 2. Apply CLAHE (local contrast)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img_clahe = clahe.apply(img_8bit)
        
        # 3. Convert to float [0, 1] for PyTorch
        img_float = img_clahe.astype(np.float32) / 255.0
            
        tensor = torch.from_numpy(img_float).unsqueeze(0).unsqueeze(0)
        return tensor.to(self.device)

    @torch.no_grad()
    def match(self, img_before: np.ndarray, img_after: np.ndarray):
        input_dict = {
            "image0": self._prepare_image(img_before),
            "image1": self._prepare_image(img_after)
        }
        
        out = self.matcher(input_dict)
        
        mkpts0 = out['keypoints0'].cpu().numpy()
        mkpts1 = out['keypoints1'].cpu().numpy()
        mconf = out['confidence'].cpu().numpy()
        
        return mkpts0, mkpts1, mconf