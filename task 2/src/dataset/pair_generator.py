import cv2
import numpy as np
from pathlib import Path
import random
from typing import Tuple

class SeasonalPairGenerator:
    """
    Generates synthetic seasonal image pairs for matching algorithms.
    Simulates illumination, contrast, and viewpoint changes to mimic 
    different seasons and satellite passes.
    """
    def __init__(self, data_dir: str):
        """
        Args:
            data_dir (str): Path to the raw dataset directory containing class folders.
        """
        self.data_dir = Path(data_dir)
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Dataset directory not found: {self.data_dir}")
            
        self.classes = [d for d in self.data_dir.iterdir() if d.is_dir()]
        print(f"[INFO] Initialized Pair Generator. Found {len(self.classes)} classes.")

    def get_random_image_path(self) -> Path:
        """Selects a random image from a random class."""
        random_class = random.choice(self.classes)
        images = list(random_class.glob("*.jpg"))
        if not images:
            raise ValueError(f"No images found in {random_class}")
        return random.choice(images)

    def apply_seasonal_transform(self, image: np.ndarray) -> np.ndarray:
        """
        Simulates winter/autumn conditions by altering HSV channels.
        Reduces saturation (less vegetation) and shifts brightness.
        """
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)
        
        sat_scale = random.uniform(0.3, 0.7)
        hsv[:, :, 1] = hsv[:, :, 1] * sat_scale
        
        val_scale = random.uniform(0.8, 1.2)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * val_scale, 0, 255)
        
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

    def apply_viewpoint_transform(self, image: np.ndarray) -> np.ndarray:
        """
        Simulates a different satellite viewpoint using affine transformations
        (rotation and scaling).
        """
        rows, cols, _ = image.shape
        angle = random.uniform(-25, 25)
        scale = random.uniform(0.85, 1.15)
        
        M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, scale)
        return cv2.warpAffine(image, M, (cols, rows))

    def generate_pair(self, image_path: Path = None) -> Tuple[np.ndarray, np.ndarray, Path]:
        """
        Generates an original and transformed image pair.
        
        Returns:
            Tuple containing:
            - Original Image (RGB numpy array)
            - Transformed Image (RGB numpy array)
            - Path to the original image
        """
        if image_path is None:
            image_path = self.get_random_image_path()
            
        img_orig = cv2.imread(str(image_path))
        if img_orig is None:
            raise FileNotFoundError(f"Failed to read {image_path}")
            
        img_orig = cv2.cvtColor(img_orig, cv2.COLOR_BGR2RGB)
        
        img_transformed = self.apply_seasonal_transform(img_orig)
        img_transformed = self.apply_viewpoint_transform(img_transformed)
        
        return img_orig, img_transformed, image_path