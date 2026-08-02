import xml.etree.ElementTree as ET
import numpy as np
from pathlib import Path
from typing import List, Tuple

class GMLParser:
    """
    Robust GML/XML parser that handles various geographic coordinate tags.
    """
    def __init__(self, gml_path: str):
        self.gml_path = Path(gml_path)
        
    def extract_polygons(self) -> List[np.ndarray]:
        """
        Extracts polygons by searching for multiple coordinate tags (posList, coordinates, pos).
        """
        if not self.gml_path.exists():
            return []

        try:
            tree = ET.parse(self.gml_path)
            root = tree.getroot()
        except ET.ParseError:
            return []

        polygons = []
        
        # Helper to extract numbers from text string
        def parse_coords_text(text: str):
            if not text:
                return
            # Split by whitespace or commas
            tokens = text.replace(',', ' ').strip().split()
            try:
                coords = list(map(float, tokens))
                if len(coords) >= 4 and len(coords) % 2 == 0:
                    poly = np.array(coords).reshape(-1, 2)
                    polygons.append(poly)
            except ValueError:
                pass

        # Search across all elements, ignoring namespaces (using local name)
        for elem in root.iter():
            # Get local tag name without namespace (e.g., '{http://...}posList' -> 'posList')
            tag = elem.tag.split('}')[-1]
            
            if tag in ['posList', 'coordinates', 'pos']:
                parse_coords_text(elem.text)
                
        return polygons

    def get_bounding_boxes(self) -> List[Tuple[int, int, int, int]]:
        polygons = self.extract_polygons()
        bboxes = []
        
        for poly in polygons:
            x_min, y_min = np.min(poly, axis=0)
            x_max, y_max = np.max(poly, axis=0)
            # Ensure it's a valid bounding box with positive area
            if x_max > x_min and y_max > y_min:
                bboxes.append((int(x_min), int(y_min), int(x_max), int(y_max)))
            
        return bboxes