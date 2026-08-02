import torch
from torch.utils.data import Dataset
from src.dataset.pair_generator import SeasonalPairGenerator
from src.matching.loftr_matcher import DeepImageMatcher

class SatellitePairDataset(Dataset):
    """
    PyTorch Dataset wrapper for the SeasonalPairGenerator.
    Prepares and preprocesses image pairs for the fine-tuning loop.
    """
    def __init__(self, data_dir: str, length: int = 1000, device: str = "cpu"):
        self.generator = SeasonalPairGenerator(data_dir=data_dir)
        self.length = length
        self.device = device
        self.matcher_helper = DeepImageMatcher(device=device)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # Generate synthetic satellite pair
        img1, img2, _ = self.generator.generate_pair()
        
        # Preprocess using the matcher's built-in method, remove batch dim for DataLoader
        tensor1 = self.matcher_helper._preprocess(img1).squeeze(0)
        tensor2 = self.matcher_helper._preprocess(img2).squeeze(0)
        
        return {"image0": tensor1, "image1": tensor2}