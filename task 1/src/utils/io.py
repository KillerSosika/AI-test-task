import json
import csv
from pathlib import Path
from typing import Any, Dict, List

def save_json(data: Any, filepath: str) -> None:
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=2)

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
    """Читає JSON файл."""
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(data: Any, filepath: str) -> None:
    """
    Зберігає дані у JSON файл. 
    Автоматично створює всі необхідні батьківські директорії.
    """
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)