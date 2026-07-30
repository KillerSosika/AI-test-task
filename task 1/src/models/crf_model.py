import pickle
import re
from pathlib import Path
from typing import Any, Dict, List

import sklearn_crfsuite
from src.models.base import ModelBase
from src.utils.features import sent2features

class CRFModel(ModelBase):
    """Wrapper for the sklearn-crfsuite model."""

    def __init__(self, max_iterations: int = 100):
        self.model = sklearn_crfsuite.CRF(
            algorithm="lbfgs",
            c1=0.1,
            c2=0.1,
            max_iterations=max_iterations,
            all_possible_transitions=True,
        )

    def train(self, x_train: List[List[Dict[str, Any]]], y_train: List[List[str]]) -> None:
        """Trains the CRF model."""
        self.model.fit(x_train, y_train)

    def predict_features(self, x_test: List[List[Dict[str, Any]]]) -> List[List[str]]:
        """Predicts labels for raw feature sequences (used by CRFTrainer)."""
        return self.model.predict(x_test)

    def predict(self, text: str) -> Dict[str, List[str]]:
        """
        Satisfies ModelBase: strictly accepts a string and returns tokens + labels.
        Polymorphic interface for the master orchestrator.
        """
        tokens = re.findall(r"[\w']+|[.,!?;]", text)
        
        if not tokens:
            return {"tokens": [], "labels": []}
            
        features = sent2features(tokens)
        labels = self.model.predict([features])[0]
        
        return {"tokens": tokens, "labels": labels}

    def save(self, filepath: str) -> None:
        """Saves the trained model to disk using pickle."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump(self.model, f)

    def load(self, filepath: str) -> None:
        """Loads a trained model from disk."""
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")
        with path.open("rb") as f:
            self.model = pickle.load(f)