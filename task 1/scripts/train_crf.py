import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.utils.config import load_config
from src.utils.utils import load_json
from src.utils.features import sent2features
from src.models.crf_model import CRFModel

def main():
    config = load_config()
    
    print("Loading training data...")
    train_data = load_json(f"{config['paths']['data_processed']}/train.json")
    
    x_train = [sent2features(item["tokens"]) for item in train_data]
    y_train = [item["labels"] for item in train_data]

    print("Training CRF model...")
    crf = CRFModel(max_iterations=100)
    crf.train(x_train, y_train)

    save_path = f"{config['paths']['models_crf']}/crf_baseline.pkl"
    print(f"Saving model to {save_path}...")
    crf.save(save_path)
    
    print("Done!")

if __name__ == "__main__":
    main()