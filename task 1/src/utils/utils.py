import os
import json
import random
from typing import Any
from dotenv import load_dotenv

def set_hf_token() -> None:
    """Loads HF_TOKEN from .env file and sets it in the environment."""
    load_dotenv()
    token = os.getenv("HF_TOKEN")
    if token:
        os.environ["HF_TOKEN"] = token
        print("✅ HF_TOKEN successfully loaded from .env")
    else:
        print("⚠️ No HF_TOKEN found in .env. Hugging Face downloads might be rate-limited.")

def load_json(filepath: str) -> Any:
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_json(data: Any, filepath: str) -> None:
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def seed_everything(seed: int = 42) -> None:
    """Ensures reproducibility across random, numpy, and torch."""
    random.seed(seed)
    # Якщо використовуєш torch/numpy, розкоментуй:
    # import numpy as np
    # import torch
    # np.random.seed(seed)
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)