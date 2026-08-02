import time
import cv2
import numpy as np
from .base_matcher import BaseMatcher, MatchResult

class SIFTMatcher(BaseMatcher):
    """
    Level 0 Baseline: Classical SIFT feature matching.
    Includes Lowe's ratio test and RANSAC for outlier filtering.
    """
    def __init__(self, ratio_threshold: float = 0.75, ransac_thresh: float = 5.0):
        """
        Args:
            ratio_threshold (float): Threshold for Lowe's ratio test (lower is stricter).
            ransac_thresh (float): Reprojection error threshold for RANSAC.
        """
        self.ratio_threshold = ratio_threshold
        self.ransac_thresh = ransac_thresh
        
        self.sift = cv2.SIFT_create()
        self.matcher = cv2.BFMatcher()

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
        
        kp1, des1 = self.sift.detectAndCompute(img1_safe, None)
        kp2, des2 = self.sift.detectAndCompute(img2_safe, None)
        
        # Handle edge cases (e.g., solid blue water where SIFT finds nothing)
        if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
            return MatchResult(
                keypoints0=np.empty((0, 2)),
                keypoints1=np.empty((0, 2)),
                inliers_mask=np.empty((0,), dtype=bool),
                execution_time_ms=(time.time() - start_time) * 1000
            )
            
        raw_matches = self.matcher.knnMatch(des1, des2, k=2)
        
        good_matches = []
        for m, n in raw_matches:
            if m.distance < self.ratio_threshold * n.distance:
                good_matches.append(m)
                
        pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches])
        pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches])
        
        inliers_mask = np.zeros(len(good_matches), dtype=bool)
        
        if len(good_matches) >= 4:
            _, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, self.ransac_thresh)
            if mask is not None:
                inliers_mask = mask.ravel().astype(bool)
                
        execution_time_ms = (time.time() - start_time) * 1000
        
        return MatchResult(
            keypoints0=pts1,
            keypoints1=pts2,
            inliers_mask=inliers_mask,
            execution_time_ms=execution_time_ms
        )

    