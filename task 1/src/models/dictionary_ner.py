import re
import json
from pathlib import Path
from typing import Dict, List, Tuple

from src.models.base import ModelBase

class DictionaryNER(ModelBase):
    """Baseline NER model using exact token sequence matching."""

    def __init__(self, ignore_case: bool = True):
        self.ignore_case = ignore_case
        self.entity_sequences: List[Tuple[str, ...]] = []

    def load_dictionary(self, entities: List[str]) -> None:
        """Loads a list of entity names into the dictionary."""
        for name in entities:
            # Tokenize the name using the same logic as the annotator[cite: 12]
            tokens = re.findall(r"[\w']+|[.,!?;]", name)
            if not tokens:
                continue
                
            if self.ignore_case:
                tokens = [t.lower() for t in tokens]
                
            self.entity_sequences.append(tuple(tokens))
            
        # Sort by sequence length descending to match longest phrases first[cite: 12]
        self.entity_sequences.sort(key=len, reverse=True)

    def predict(self, text: str) -> Dict[str, List[str]]:
        """Predicts BIO labels for a given text using dictionary lookups."""
        tokens = re.findall(r"[\w']+|[.,!?;]", text)
        labels = ["O"] * len(tokens)
        
        # Prepare search tokens[cite: 12]
        search_tokens = [t.lower() for t in tokens] if self.ignore_case else tokens

        i = 0
        while i < len(tokens):
            match_found = False
            
            # Check against all known entity sequences[cite: 12]
            for seq in self.entity_sequences:
                seq_len = len(seq)
                
                # If the sequence fits in the remaining tokens[cite: 12]
                if i + seq_len <= len(tokens):
                    if tuple(search_tokens[i:i + seq_len]) == seq:
                        labels[i] = "B-MOUNTAIN"
                        for j in range(1, seq_len):
                            labels[i + j] = "I-MOUNTAIN"
                            
                        i += seq_len
                        match_found = True
                        break
                        
            if not match_found:
                i += 1

        return {"tokens": tokens, "labels": labels}

    def save(self, filepath: str) -> None:
        """Saves the dictionary sequences to a JSON file."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            # Convert tuples to lists for JSON serialization
            json.dump([list(seq) for seq in self.entity_sequences], f, ensure_ascii=False)

    def load(self, filepath: str) -> None:
        """Loads the dictionary sequences from a JSON file."""
        path = Path(filepath)
        if not path.exists():
            print(f"Warning: Dictionary file not found at {path}")
            return
            
        with path.open("r", encoding="utf-8") as f:
            sequences = json.load(f)
            self.entity_sequences = [tuple(seq) for seq in sequences]