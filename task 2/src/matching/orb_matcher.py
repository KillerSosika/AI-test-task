import time
import cv2
import numpy as np
from .base_matcher import BaseMatcher, MatchResult

class ORBMatcher(BaseMatcher):
    """
    Level 0 Alternative: ORB (Oriented FAST and Rotated BRIEF).
    Much faster than SIFT, but potentially less robust to extreme changes.
    """
    def __init__(self, max_features: int = 1000, ransac_thresh: float = 5.0):
        self.ransac_thresh = ransac_thresh
        self.orb = cv2.ORB_create(max_features)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    def _to_uint8_gray(self, img: np.ndarray) -> np.ndarray:
        """Forces any input into an 8-bit grayscale numpy array for OpenCV."""
        # Convert float32 [0, 1] back to uint8 [0, 255]
        if img.dtype == np.float32 or img.dtype == np.float64:
            img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
            
        # Drop to 2D
        if len(img.shape) == 3:
            if img.shape[2] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            elif img.shape[2] == 1:
                img = img.squeeze(-1)
        return img
        
    def match(self, img1: np.ndarray, img2: np.ndarray) -> MatchResult:
        start_time = time.time()

        img1_safe = self._to_uint8_gray(img1)
        img2_safe = self._to_uint8_gray(img2)
                
        kp1, des1 = self.orb.detectAndCompute(img1_safe, None)
        kp2, des2 = self.orb.detectAndCompute(img2_safe, None)
        
        if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
            return MatchResult(np.empty((0, 2)), np.empty((0, 2)), np.empty(0, dtype=bool), (time.time() - start_time) * 1000)
            
        good_matches = self.matcher.match(des1, des2)
        
        pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches])
        pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches])
        
        inliers_mask = np.zeros(len(good_matches), dtype=bool)
        if len(good_matches) >= 4:
            _, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, self.ransac_thresh)
            if mask is not None:
                inliers_mask = mask.ravel().astype(bool)
                
        return MatchResult(pts1, pts2, inliers_mask, (time.time() - start_time) * 1000)


    