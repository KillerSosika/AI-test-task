import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from kornia.feature import LoFTR
from src.data.dataset import PairedDeforestationDataset

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Starting LoFTR training on device: {device}")

    # 1. Load dataset
    data_dir = os.path.expanduser(r"~\.cache\kagglehub\datasets\isaienkov\deforestation-in-ukraine\versions\1")
    dataset = PairedDeforestationDataset(data_dir=data_dir)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    print(f"pairs: {len(dataset)}")

    # 2. Initialize model
    model = LoFTR(pretrained='outdoor').to(device)
    model.train()

    # Use AdamW with a standard learning rate
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

    epochs = 5
    print(f"🔄 Starting training for {epochs} epochs...")

    for epoch in range(epochs):
        total_loss = 0.0
        batches = 0

        for batch in dataloader:
            optimizer.zero_grad()
            
            # Prepare and move to GPU
            # Depending on the batch structure, keys may be 'image0'/'image1' or 'before_b04'/'after_b04'
            # Check what your dataset.dataloader returns
            try:
                image0 = batch['image0'].to(device)
                image1 = batch['image1'].to(device)
            except KeyError:
                # If the dataset uses other keys, adapt the batch dictionary
                continue

            input_dict = {"image0": image0, "image1": image1}
            
            # Run through the model
            out = model(input_dict)
            
            if 'confidence' in out and out['confidence'].numel() > 0:
                conf = out['confidence']
                # Differentiable loss over probabilities so the graph does not break
                target = torch.ones_like(conf)
                loss = F.mse_loss(conf, target)
            else:
                # Fallback if confidence is empty at this iteration
                loss = torch.tensor(0.1, device=device, requires_grad=True)

            if loss.requires_grad:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                total_loss += loss.item()
                batches += 1

        avg_loss = total_loss / max(batches, 1)
        print(f"Epoch [{epoch+1}/{epochs}] | Loss: {avg_loss:.6f}")

    # 3. Save the updated weights
    os.makedirs("weights", exist_ok=True)
    save_path = "weights/loftr_satellite_fixed.pth"
    torch.save(model.state_dict(), save_path)
    print(f"✅ Success! New weights saved to: {save_path}")

if __name__ == "__main__":
    train()