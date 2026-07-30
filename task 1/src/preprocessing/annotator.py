import re
from typing import Dict, List

class NERAnnotator:
    """Assigns BIO tags to text based on the provided entity."""

    def __init__(self, entity_label: str = "MOUNTAIN"):
        self.entity_label = entity_label

    def annotate(self, text: str, entity: str) -> Dict[str, List[str]]:
        """
        Tokenizes text and applies BIO tags.
        
        Args:
            text: The generated text (e.g., "We climbed Mount Everest.")
            entity: The target entity (e.g., "Mount Everest")
            
        Returns:
            A dictionary containing 'tokens' and 'labels' lists.
        """
        # Simple word tokenizer keeping punctuation separate
        tokens = re.findall(r"[\w']+|[.,!?;]", text)
        labels = ["O"] * len(tokens)

        if not entity:
            return {"tokens": tokens, "labels": labels}

        entity_tokens = re.findall(r"[\w']+|[.,!?;]", entity)
        entity_len = len(entity_tokens)

        if entity_len == 0:
            return {"tokens": tokens, "labels": labels}

        i = 0
        while i <= len(tokens) - entity_len:
            current_window = [t.lower() for t in tokens[i:i + entity_len]]
            target_window = [e.lower() for e in entity_tokens]
            
            if current_window == target_window:
                labels[i] = f"B-{self.entity_label}"
                for j in range(1, entity_len):
                    labels[i + j] = f"I-{self.entity_label}"
                i += entity_len
            else:
                i += 1

        return {"tokens": tokens, "labels": labels}