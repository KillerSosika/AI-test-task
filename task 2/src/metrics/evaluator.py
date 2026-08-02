import numpy as np
from typing import Dict, Tuple
from src.matching.base_matcher import BaseMatcher, MatchResult

class Evaluator:
    """
    Evaluates image matching algorithms uniformly based on the MatchResult contract.
    """
    
    @staticmethod
    def evaluate(matcher: BaseMatcher, img1: np.ndarray, img2: np.ndarray) -> Tuple[Dict, MatchResult]:
        """
        Runs any provided matcher and calculates core performance metrics.
        Returns both the metrics dictionary and the raw MatchResult object.
        """
        result = matcher.match(img1, img2)
        
        precision = 0.0
        if result.num_keypoints > 0:
            precision = (result.num_inliers / result.num_keypoints) * 100
            
        metrics = {
            "Total Matches": result.num_keypoints,
            "Valid Inliers": result.num_inliers,
            "Precision (%)": round(precision, 2),
            "Execution Time (ms)": round(result.execution_time_ms, 2)
        }
        
        # Return both the table dictionary and the raw result for visualization
        return metrics, result