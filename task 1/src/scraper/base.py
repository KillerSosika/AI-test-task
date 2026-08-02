from abc import ABC, abstractmethod
from typing import List, Dict

class BaseScraper(ABC):
    """Abstract base class for all scrapers."""
    
    @abstractmethod
    def fetch_data(self) -> List[Dict]:
        """Method to collect data. Must be implemented by child classes."""
        pass

    @abstractmethod
    def save_data(self, filepath: str) -> None:
        """Method to save collected data."""
        pass