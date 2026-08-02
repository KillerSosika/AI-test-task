import json
import time
from typing import Dict, List, Any
from pathlib import Path

class ModelEvaluator:
    """Class for fair evaluation of NER models on a test dataset."""

    def __init__(self, dict_model, crf_model, bert_model):
        self.models = {
            "Dictionary": dict_model,
            "CRF": crf_model,
            "BERT": bert_model
        }

    def _calculate_metrics(self, true_labels: List[List[str]], pred_labels: List[List[str]]) -> Dict[str, float]:
        """Flat metric calculation for MOUNTAIN entities."""
        true_flat = [label for seq in true_labels for label in seq]
        pred_flat = [label for seq in pred_labels for label in seq]

        tp = sum(1 for t, p in zip(true_flat, pred_flat) if t != "O" and t == p)
        fp = sum(1 for t, p in zip(true_flat, pred_flat) if p != "O" and t != p)
        fn = sum(1 for t, p in zip(true_flat, pred_flat) if t != "O" and t == "O")

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        return {"precision": precision, "recall": recall, "f1": f1}

    def evaluate_all(self, test_data_path: str) -> Dict[str, Dict[str, float]]:
        """Runs test data through all models and returns metrics."""
        print(f"📊 Starting a fair evaluation on the dataset: {test_data_path}")
        
        with open(test_data_path, "r", encoding="utf-8") as f:
            test_data = json.load(f)

        results = {}

        for model_name, model in self.models.items():
            print(f"🔄 Evaluating {model_name}...")
            all_true = []
            all_pred = []
            
            start_time = time.time()
            for item in test_data:
                text = " ".join(item["tokens"])
                all_true.append(item["labels"])
                
                if model_name == "BERT":
                    pred = model.predict(text)
                    pred_labels = ["O"] * len(item["tokens"])
                    for ent in pred:
                        pass 
                else:
                    pred = model.predict(text)
                    all_pred.append(pred.get("labels", ["O"] * len(item["tokens"])))
                    
            inference_time = time.time() - start_time
            
            metrics = self._calculate_metrics(all_true, all_pred)
            metrics["speed_ms"] = (inference_time / len(test_data)) * 1000
            results[model_name] = metrics
            
        return results