import json
import csv
from pathlib import Path
from typing import Any, Dict, List


def save_csv(data: List[Dict[str, Any]], filepath: str, fieldnames: List[str] = None) -> None:
    if not data:
        return
        
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    if fieldnames is None:
        fieldnames = list(data[0].keys())

    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)

def load_json(filepath: str) -> Any:
    """Reads a JSON file."""
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(data: Any, filepath: str) -> None:
    """
    Saves data to a JSON file.
    Automatically creates all required parent directories.
    """
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)