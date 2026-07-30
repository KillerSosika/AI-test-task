from abc import ABC, abstractmethod
from typing import Any, List

class _PlaceholderDict(dict):
    def __missing__(self, key: str) -> str:
        return "unknown"

class BaseTextGenerator(ABC):
    """Abstract interface for text generation."""

    @abstractmethod
    def generate(self, **context: Any) -> str:
        raise NotImplementedError

    def generate_batch(self, batch_size: int = 1, **context: Any) -> List[str]:
        return [self.generate(**context) for _ in range(batch_size)]