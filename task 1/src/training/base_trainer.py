from abc import ABC, abstractmethod
from typing import Any, Dict

class BaseTrainer(ABC):
    """Abstract base class for all model trainers."""
    
    @abstractmethod
    def train(self, train_data: Any, val_data: Any = None, **kwargs) -> None:
        """Executes the training loop."""
        pass

    @abstractmethod
    def evaluate(self, val_data: Any) -> Dict[str, float]:
        """Evaluates the model on validation data and returns metrics."""
        pass

    @abstractmethod
    def save(self, path: str) -> None:
        """Saves the trained model state."""
        pass