import sys
import argparse
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.utils.config import load_config
from src.utils.utils import load_json, set_hf_token, seed_everything
from src.deep_learning.dataset import NERDatasetBuilder
from src.training.fine_tuner import BERTFineTuner

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    return parser.parse_args()

def main():
    args = parse_args()
    set_hf_token()
    seed_everything(42)
    config = load_config()

    print("Loading JSON dataset...")    
    train_data = load_json(f"{config['paths']['data_final']}/train.json")

    trainer = BERTFineTuner(
        model_name=config['model_params']['bert']['model_name'], 
        output_dir=config['paths']['models_finetuned']
    )

    print("Building Hugging Face Dataset...")
    dataset_builder = NERDatasetBuilder(tokenizer=trainer.tokenizer)
    hf_train = dataset_builder.build_hf_dataset(train_data)
    # hf_val = dataset_builder.build_hf_dataset(val_data)

    print(f"Starting Fine-tuning ({args.epochs} epochs, batch size {args.batch_size})...")
    trainer.train(train_data=hf_train, epochs=args.epochs, batch_size=args.batch_size)
    print("✨ BERT Fine-tuning complete!")

if __name__ == "__main__":
    main()