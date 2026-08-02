import argparse
import csv
import random
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.utils.config import load_config
from src.utils.io import save_json
from src.generation.template_generator import TemplateGenerator

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate strict synthetic NER dataset.")
    parser.add_argument("--samples", type=int, default=5, help="Positive examples per mountain.")
    parser.add_argument("--negatives", type=int, default=500, help="Number of negative examples to add.")
    return parser.parse_args()

def main():
    args = parse_args()
    config = load_config()

    csv_path = f"{config['paths']['data_processed']}/mountains.csv"
    train_path = f"{config['paths']['data_final']}/train.json"
    val_path = f"{config['paths']['data_final']}/validation.json"

    # 1. Load mountains
    mountains = []
    try:
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("name"):
                    mountains.append(row["name"])
    except FileNotFoundError:
        print("Error: mountains.csv not found. Run scrape_mountains.py first.")
        sys.exit(1)

    print(f"Loaded {len(mountains)} mountains.")

    # 2. Generate data
    generator = TemplateGenerator(mountains)
    
    print(f"Generating positive examples ({args.samples} per mountain)...")
    positive_data = generator.generate_positive(num_per_mountain=args.samples)
    
    print(f"Generating {args.negatives} negative examples (distractors)...")
    negative_data = generator.generate_negative(num_samples=args.negatives)

    # 3. Merge and shuffle
    all_data = positive_data + negative_data
    random.shuffle(all_data)

    # 4. 80/20 split
    split_idx = int(len(all_data) * 0.8)
    train_data = all_data[:split_idx]
    val_data = all_data[split_idx:]

    # 5. Save
    save_json(train_data, train_path)
    save_json(val_data, val_path)

    print(f"Dataset ready! Total: {len(all_data)} sentences.")
    print(f"Train: {len(train_data)} | Validation: {len(val_data)}")

if __name__ == "__main__":
    main()