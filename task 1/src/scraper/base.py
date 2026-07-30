from abc import ABC, abstractmethod
from typing import List, Dict

class BaseScraper(ABC):
    """Абстрактний базовий клас для всіх скраперів."""
    
    @abstractmethod
    def fetch_data(self) -> List[Dict]:
        """Метод для збору даних. Повинен бути реалізований у дочірніх класах."""
        pass

    @abstractmethod
    def save_data(self, filepath: str) -> None:
        """Метод для збереження зібраних даних."""
        pass