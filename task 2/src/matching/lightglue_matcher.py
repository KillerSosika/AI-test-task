import time
import torch
import cv2
import numpy as np
from lightglue import LightGlue, ALIKED
from .base_matcher import BaseMatcher, MatchResult

class LightGlueImageMatcher(BaseMatcher):
    """
    Level 3 matcher.
    ALIKED -> LightGlue -> RANSAC (Official cvg/LightGlue implementation)
    """
    def __init__(
        self,
        device: str | None = None,
        ransac_thresh: float = 3.0,
        max_keypoints: int = 2048,
        debug: bool = False,
    ):
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.debug = debug
        self.ransac_thresh = ransac_thresh

        print(f"[INFO] Initializing LightGlue on {self.device.upper()}")

        self.extractor = ALIKED(max_num_keypoints=max_keypoints).to(self.device)
        self.matcher = LightGlue(features="aliked").to(self.device)

        self.extractor.eval()
        self.matcher.eval()
        print("[INFO] ALIKED + LightGlue loaded")

    def _img_to_tensor(self, img: np.ndarray):
        if img.dtype == np.uint8:
            img = img.astype(np.float32) / 255.0

        if img.ndim == 2:
            img = np.stack([img] * 3, axis=-1)
        elif img.shape[2] == 1:
            img = np.repeat(img, 3, axis=2)

        tensor = torch.from_numpy(img)
        tensor = tensor.permute(2, 0, 1).unsqueeze(0)
        return tensor.to(self.device)

    @torch.no_grad()
    def match(self, img1: np.ndarray, img2: np.ndarray) -> MatchResult:
        start = time.time()

        if self.debug:
            print("\n========== LIGHTGLUE DEBUG ==========")
            print(f"Image1: {img1.shape}")
            print(f"Image2: {img2.shape}")

        t1 = self._img_to_tensor(img1)
        t2 = self._img_to_tensor(img2)

        feat1 = self.extractor({"image": t1})
        feat2 = self.extractor({"image": t2})

        if isinstance(feat1, list): feat1 = feat1[0]
        if isinstance(feat2, list): feat2 = feat2[0]

        n1 = feat1["keypoints"].shape[1] if feat1["keypoints"].ndim == 3 else feat1["keypoints"].shape[0]
        n2 = feat2["keypoints"].shape[1] if feat2["keypoints"].ndim == 3 else feat2["keypoints"].shape[0]

        if self.debug:
            print(f"Keypoints image1 : {n1}")
            print(f"Keypoints image2 : {n2}")

        if n1 < 4 or n2 < 4:
            if self.debug: print("Too few keypoints.")
            return MatchResult(np.empty((0, 2)), np.empty((0, 2)), np.empty(0, dtype=bool), (time.time() - start) * 1000)

        result = self.matcher({
            "image0": feat1,
            "image1": feat2,
        })

        if self.debug:
            print("Result keys:", result.keys())

        # 1. Safely extract objects
        matches = result.get("matches")
        scores = result.get("scores")

        if matches is None or scores is None:
            if self.debug: print("No matches or scores returned.")
            return MatchResult(np.empty((0, 2)), np.empty((0, 2)), np.empty(0, dtype=bool), (time.time() - start) * 1000)

        # 2. Unpack lists
        if isinstance(matches, list): matches = matches[0]
        if isinstance(scores, list): scores = scores[0]

        # 3. Debug tensors before conversion
        if self.debug:
            print("Matches shape:", matches.shape)
            print("First 10 matches:\n", matches[:10])

        # 4. Convert to numpy
        if torch.is_tensor(matches): matches = matches.cpu().numpy()
        if torch.is_tensor(scores): scores = scores.cpu().numpy()

        # 5. Remove batch dimension
        if matches.ndim == 3: matches = matches[0]
        if scores.ndim == 2: scores = scores.squeeze()

        if self.debug:
            print("Raw matches before filtering:", len(matches))

        # 6. Filter by confidence
        confidence_threshold = 0.90
        valid_mask = scores > confidence_threshold
        
        matches = matches[valid_mask]
        scores = scores[valid_mask]
        
        if self.debug:
            print(f"Matches after filtering (>{confidence_threshold}):", len(matches))

        # 7. Check count after filtering
        if len(matches) < 4:
            if self.debug: print("Not enough matches after filtering.")
            return MatchResult(np.empty((0, 2)), np.empty((0, 2)), np.empty(0, dtype=bool), (time.time() - start) * 1000)
        
        # Safely extract keypoints
        kpts0 = feat1["keypoints"].squeeze(0).cpu().numpy() if feat1["keypoints"].ndim == 3 else feat1["keypoints"].cpu().numpy()
        kpts1 = feat2["keypoints"].squeeze(0).cpu().numpy() if feat2["keypoints"].ndim == 3 else feat2["keypoints"].cpu().numpy()

        pts1 = kpts0[matches[:, 0]]
        pts2 = kpts1[matches[:, 1]]

        mask = None
        if len(pts1) >= 8:
            _, mask = cv2.findHomography(pts1, pts2, cv2.USAC_MAGSAC, self.ransac_thresh)

        inliers = np.zeros(len(pts1), dtype=bool) if mask is None else mask.ravel().astype(bool)

        if self.debug:
            print("Inliers:", int(inliers.sum()))
            print(f"Precision: {round(100 * inliers.mean(), 2)} %" if len(inliers) > 0 else "0 %")
            print(f"Execution: {round((time.time() - start) * 1000, 2)} ms")
            print("=====================================\n")

        return MatchResult(
            keypoints0=pts1,
            keypoints1=pts2,
            inliers_mask=inliers,
            execution_time_ms=(time.time() - start) * 1000,
            scores=scores
        )