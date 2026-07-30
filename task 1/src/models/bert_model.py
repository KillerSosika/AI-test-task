from typing import Any, Dict, List
from transformers import pipeline

from src.models.base import ModelBase

class BERTModel(ModelBase):
    """Predictor for Mountain NER using a fine-tuned BERT model."""

    def __init__(self, model_path: str):
        """
        Initializes the Hugging Face token-classification pipeline.
        
        Args:
            model_path: Path to the directory containing the fine-tuned model and tokenizer.
        """
        self.pipeline = pipeline(
            task="token-classification",
            model=model_path,
            tokenizer=model_path,
            aggregation_strategy="simple"
        )

    def predict(self, text: str) -> List[Dict[str, Any]]:
        """
        Extracts entities from the input text.
        
        Returns:
            List of dictionaries containing the entity word, label, and confidence score.
        """
        return self.pipeline(text)

    def save(self, path: str) -> None:
        """
        Since the pipeline is already loaded from a saved directory and HF Trainer 
        handles saving during fine-tuning, this can remain a pass or explicitly 
        save the pipeline if needed.
        """
        self.pipeline.save_pretrained(path)

    def load(self, path: str) -> None:
        """
        The pipeline is loaded during initialization, but this satisfies the ModelBase contract.
        """
        self.pipeline = pipeline(
            task="token-classification",
            model=path,
            tokenizer=path,
            aggregation_strategy="simple"
        )