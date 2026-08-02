from abc import ABC, abstractmethod
import numpy as np
from dataclasses import dataclass
from typing import Optional

@dataclass
class MatchResult:
    """
    Unified output format for all image matching algorithms.
    Ensures metrics and visualization modules receive consistent data.
    """
    keypoints0: np.ndarray      # Coordinates of points in the first image, shape: (N, 2)
    keypoints1: np.ndarray      # Coordinates of points in the second image, shape: (N, 2)
    inliers_mask: np.ndarray    # Boolean array indicating valid matches after RANSAC/filtering, shape: (N,)
    execution_time_ms: float    # Algorithm execution time in milliseconds
    scores: np.ndarray = None
    
    @property
    def num_keypoints(self) -> int:
        """Returns the total number of detected keypoints."""
        return len(self.keypoints0)
        
    @property
    def num_inliers(self) -> int:
        """Returns the number of verified inliers."""
        if self.inliers_mask is None:
            return 0
        return int(np.sum(self.inliers_mask))


class BaseMatcher(ABC):
    """
    Abstract base class defining the strict contract for all matching models.
    """
    
    @abstractmethod
    def match(self, img_before: np.ndarray, img_after: np.ndarray):
        """
        Accepts two images (tiles).

        Returns a tuple (mkpts1, mkpts2, confidence):
        - mkpts1 (np.ndarray): coordinates of points in the BEFORE image in shape (N, 2)
        - mkpts2 (np.ndarray): coordinates of points in the AFTER image in shape (N, 2)
        - confidence (np.ndarray): model confidence for each pair (N,)
        """
        pass