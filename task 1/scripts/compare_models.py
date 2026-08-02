import sys
import csv
from pathlib import Path
import transformers

# Add the project root to the import path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.utils.config import load_config
from src.models.dictionary_ner import DictionaryNER
from src.models.crf_model import CRFModel
from src.models.bert_model import BERTModel

def main():
    transformers.logging.set_verbosity_error()
    config = load_config()
    
    text = "Last year I visited Nepal to see Mount Everest, but next time I want to conquer K2 and maybe a small hill near my house."
    print("=" * 60)
    print(" 🔥 SIMPLE COMPARISON OF ALL MODELS (POLYMORPHISM) 🔥")
    print("=" * 60)
    print(f"Input Text: {text}\n")

    # ---------------------------------------------------------
    # 1. Initialize all models
    # ---------------------------------------------------------
    
    dict_model = DictionaryNER(ignore_case=True)
    csv_path = f"{config['paths']['data_processed']}/mountains.csv"
    with open(csv_path, "r", encoding="utf-8") as f:
        mountains = [row["name"] for row in csv.DictReader(f) if row.get("name")]
    dict_model.load_dictionary(mountains)

    crf_model = CRFModel()
    crf_model.load(f"{config['paths']['models_crf']}/crf_baseline.pkl")

    bert_model = BERTModel(model_path=config['paths']['models_finetuned'])

    # ---------------------------------------------------------
    # 2. ITERATION
    # ---------------------------------------------------------
    
    models = [
        ("1. Dictionary lookup (Dictionary)", dict_model),
        ("2. Classical ML (CRF Model)", crf_model),
        ("3. Neural model (Fine-Tuned BERT)", bert_model)
    ]

    for name, model in models:
        print(f"--- {name} ---")
        
        result = model.predict(text)
        
        if isinstance(model, (DictionaryNER, CRFModel)):
            for token, label in zip(result["tokens"], result["labels"]):
                if label != "O":
                    print(f"  ✅ {token:<15} {label}")
                    
        else: 
            for entity in result:
                word = entity.get('word', '')
                score = entity.get('score', 0.0)
                print(f"  ✅ {word:<15} (confidence: {score:.4f})")
        
        print() #

if __name__ == "__main__":
    main()