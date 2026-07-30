from abc import ABC, abstractmethod
from typing import Any

class ModelBase(ABC):
    """Abstract base class for all NER inference models."""
    
    @abstractmethod
    def predict(self, text: str) -> Any:
        """Runs inference on a single string of text."""
        pass

    @abstractmethod
    def save(self, path: str) -> None:
        """Saves the model weights/config to the specified path."""
        pass

    @abstractmethod
    def load(self, path: str) -> None:
        """Loads the model weights/config from the specified path."""
        pass