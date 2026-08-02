import numpy as np
from typing import List, Tuple

class TileGenerator:
    """
    Handles both standard grid tiling and smart ROI-based extraction for large satellite images.
    """
    def __init__(self, tile_size: int = 512, overlap: int = 128):
        self.tile_size = tile_size
        self.overlap = overlap

    def split(self, image: np.ndarray) -> List[List[np.ndarray]]:
        """
        Legacy grid-based tiling. 
        Splits the entire image into a grid of overlapping tiles.
        """
        h, w = image.shape[:2]
        stride = self.tile_size - self.overlap
        tiles = []
        
        for y in range(0, h - self.tile_size + 1, stride):
            row_tiles = []
            for x in range(0, w - self.tile_size + 1, stride):
                tile = image[y:y + self.tile_size, x:x + self.tile_size]
                row_tiles.append(tile)
            tiles.append(row_tiles)
            
        return tiles


    def extract_roi_tiles(self, image: np.ndarray, bboxes: List[Tuple[int, int, int, int]]) -> List[np.ndarray]:
        tiles = []
        h, w = image.shape[:2]
        extracted_centers = set()

        for (xmin, ymin, xmax, ymax) in bboxes:
            padding = int(self.tile_size * 0.2)
            x_start = max(0, xmin - padding)
            y_start = max(0, ymin - padding)
            x_end = min(w, xmax + padding)
            y_end = min(h, ymax + padding)

            stride = self.tile_size - self.overlap

            for y in range(y_start, max(y_start + 1, y_end - self.tile_size + 1), stride):
                for x in range(x_start, max(x_start + 1, x_end - self.tile_size + 1), stride):
                    
                    y1, x1 = y, x
                    y2, x2 = y1 + self.tile_size, x1 + self.tile_size

                    if y2 > h:
                        y2, y1 = h, max(0, h - self.tile_size)
                    if x2 > w:
                        x2, x1 = w, max(0, w - self.tile_size)

                    center = (x1 + x2 // 2, y1 + y2 // 2)
                    
                    if center not in extracted_centers:
                        tile = image[y1:y2, x1:x2]
                        if tile.shape[0] == self.tile_size and tile.shape[1] == self.tile_size:
                            
                        
                            if np.std(tile) > 50.0: 
                                tiles.append(tile)
                                extracted_centers.add(center)

        return tiles