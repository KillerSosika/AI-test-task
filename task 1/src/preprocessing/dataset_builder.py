import random
from typing import Dict, List, Tuple

from src.generation.base import BaseTextGenerator
from src.preprocessing.annotator import NERAnnotator

class DatasetBuilder:
    """Orchestrates text generation, annotation, and splitting."""

    def __init__(
        self,
        generator: BaseTextGenerator,
        annotator: NERAnnotator,
        samples_per_entity: int = 5
    ):
        self.generator = generator
        self.annotator = annotator
        self.samples_per_entity = samples_per_entity

    def build(self, entities: List[Dict[str, str]]) -> List[Dict[str, list]]:
        """Generates and annotates synthetic data."""
        dataset = []
        
        for item in entities:
            entity_name = item.get("name", "")
            if not entity_name:
                continue

            # 1. Generate texts
            texts = self.generator.generate_batch(
                batch_size=self.samples_per_entity,
                mountain=entity_name,
                country=item.get("country", "Unknown"),
                elevation_m=item.get("elevation_m", "Unknown")
            )

            # 2. Annotate texts
            for text in texts:
                annotated = self.annotator.annotate(text, entity_name)
                
                # Verify that the entity was successfully embedded and annotated
                if f"B-{self.annotator.entity_label}" in annotated["labels"]:
                    dataset.append(annotated)

        return dataset

    def split(
        self, 
        dataset: List[Dict], 
        train_ratio: float = 0.8, 
        val_ratio: float = 0.1
    ) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """Splits the dataset into train, validation, and test sets."""
        random.shuffle(dataset)
        
        total = len(dataset)
        train_end = int(total * train_ratio)
        val_end = train_end + int(total * val_ratio)

        train_data = dataset[:train_end]
        val_data = dataset[train_end:val_end]
        test_data = dataset[val_end:]

        return train_data, val_data, test_data