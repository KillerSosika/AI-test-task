import numpy as np

class CloudFilter:
    def __init__(self, cloud_threshold=200, max_cloud_percent=0.2, haze_threshold=150, max_haze_percent=0.5):
        """
        Module for filtering out cloudy regions in satellite images.

        Parameters are normalized for 8-bit images (0-255):
        - cloud_threshold: Brightness threshold for dense (opaque) clouds.
        - max_cloud_percent: Maximum allowed share of dense clouds in the ROI.
        - haze_threshold: Brightness threshold for semi-transparent haze.
        - max_haze_percent: Maximum allowed share of haze in the ROI.
        """
        self.cloud_threshold = cloud_threshold
        self.max_cloud_percent = max_cloud_percent
        self.haze_threshold = haze_threshold
        self.max_haze_percent = max_haze_percent

    def is_valid_roi(self, tile: np.ndarray) -> bool:
        """
        Checks whether a tile is suitable for training/inference.
        """
        total_pixels = tile.size
        
        # 1. Check for dense clouds (solid white color)
        cloud_pixels = np.sum(tile > self.cloud_threshold)
        cloud_percent = cloud_pixels / total_pixels
        
        if cloud_percent > self.max_cloud_percent:
            return False
            
        # 2. Check for strong haze (bright areas where contours are lost)
        haze_pixels = np.sum(tile > self.haze_threshold)
        haze_percent = haze_pixels / total_pixels
        
        if haze_percent > self.max_haze_percent:
            return False
            
        return True