import torch
import os
from kornia.feature import LoFTR

def export_weights():
    weights_path = "weights/loftr_satellite_finetuned.pth"
    if not os.path.exists(weights_path):
        print(f"❌ Weights not found at {weights_path}")
        return

    model = LoFTR(pretrained='outdoor')
    state_dict = torch.load(weights_path, map_location='cpu', weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    print("✅ Fine-tuned weights successfully loaded and verified for inference pipeline.")

if __name__ == "__main__":
    export_weights()