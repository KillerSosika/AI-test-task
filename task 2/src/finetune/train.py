import sys
import os

# Примусово додаємо корінь проєкту у Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import OneCycleLR
from kornia.feature import LoFTR

from src.data.dataset import PairedDeforestationDataset
from src.finetune.dataset import SatelliteLoFTRDataset

def train_on_location(model, device, pair, loc_idx, data_dir):
    """Навчання з офіційним NLL-Loss для LoFTR"""
    dataset = SatelliteLoFTRDataset(data_dir, [pair], crop_size=512)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    max_steps = 500 
    patience = 25
    
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=1e-4, weight_decay=0.01)
    scheduler = OneCycleLR(optimizer, max_lr=1e-4, total_steps=max_steps, pct_start=0.1)

    best_loss = float('inf')
    patience_counter = 0

    print(f"\n" + "="*50)
    print(f"🌍 ЛОКАЦІЯ {loc_idx} | Початок навчання")
    print(f"="*50)

    for step in range(max_steps):
        batch = next(iter(dataloader))
        optimizer.zero_grad()

        image0 = batch['image0'].to(device)
        image1 = batch['image1'].to(device)
        target = batch['spv_conf_matrix_gt'].to(device) 

        feat_c0, _ = model.backbone(image0)
        feat_c1, _ = model.backbone(image1)

        feat_c0 = model.pos_encoding(feat_c0)
        feat_c1 = model.pos_encoding(feat_c1)

        feat_c0_flat = feat_c0.flatten(2).permute(0, 2, 1)
        feat_c1_flat = feat_c1.flatten(2).permute(0, 2, 1)

        feat_c0_out, feat_c1_out = model.loftr_coarse(feat_c0_flat, feat_c1_flat)

        temperature = 0.1
        sim_matrix = torch.einsum("blc,bsc->bls", feat_c0_out, feat_c1_out) / temperature
        conf_matrix = F.softmax(sim_matrix, dim=1) * F.softmax(sim_matrix, dim=2)

        # =======================================================
        # ОФІЦІЙНИЙ LOFTR LOSS (Negative Log-Likelihood)
        # =======================================================
        pos_mask = target > 0.5  

        if pos_mask.sum() < 5:
            continue

        # Беремо ймовірності тільки для правильних точок і рахуємо логарифм.
        # Додаємо 1e-8, щоб логарифм від нуля не видав NaN і не підірвав ваги
        loss = -torch.log(conf_matrix[pos_mask] + 1e-8).mean()
        # =======================================================

        if loss.requires_grad:
            loss.backward()
            
            # Жорсткий кліпінг градієнтів рятує Трансформери від колапсу
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=0.5)
            
            optimizer.step()
            scheduler.step()
            
            current_loss = loss.item()
            current_lr = scheduler.get_last_lr()[0]
            
            if current_loss < best_loss:
                best_loss = current_loss
                patience_counter = 0
                status = "📉 Loss впав"
            else:
                patience_counter += 1
                status = f"⚠️ Без змін ({patience_counter}/{patience})"

            print(f"Крок [{step+1:02d}/{max_steps}] | Loss: {current_loss:.4f} | LR: {current_lr:.2e} | {status}")

            if patience_counter >= patience:
                print(f"🛑 Локальний Early Stopping! Модель витиснула максимум.")
                break

    return model

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Запуск правильного Fine-Tuning на пристрої: {device}")

    data_dir = os.path.expanduser(r"~\.cache\kagglehub\datasets\isaienkov\deforestation-in-ukraine\versions\1")
    base_dataset = PairedDeforestationDataset(data_dir=data_dir)
    pairs = base_dataset.pairs

    if not pairs:
        print("❌ Не знайдено пар зображень.")
        return

    print(f"✅ Знайдено {len(pairs)} локацій.")

    # ВАЖЛИВО: Завантажуємо чисту модель, щоб перезаписати мертві ваги
    model = LoFTR(pretrained='outdoor').to(device)

    for name, param in model.named_parameters():
        if "backbone" in name:
            param.requires_grad = False

    model.train()

    os.makedirs("weights", exist_ok=True)
    save_path = "weights/loftr_satellite_finetuned.pth"

    global_epochs = 10

    for global_epoch in range(global_epochs):
        print(f"\n" + "🔥"*25)
        print(f"🔄 ГЛОБАЛЬНЕ КОЛО [{global_epoch+1}/{global_epochs}]")
        print(f"🔥"*25)

        for idx, pair in enumerate(pairs, start=1):
            model = train_on_location(model, device, pair, idx, data_dir)
            torch.save(model.state_dict(), f"weights/loftr_loc_{idx}_epoch_{global_epoch+1}.pth")

    torch.save(model.state_dict(), save_path)
    print(f"\n✅ Глобальний фініш! Ваги лежать тут: {save_path}")

if __name__ == "__main__":
    main()