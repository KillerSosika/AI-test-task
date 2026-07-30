import random
import re
from typing import Dict, List, Any

class TemplateGenerator:
    """Generates synthetic NER data with strict token alignment and negative examples."""

    def __init__(self, mountains: List[str]):
        self.mountains = mountains
        
        # 1. Позитивні шаблони (де точно є гора)
        self.positive_templates = [
            "We climbed {mountain} last week.",
            "My dream is to visit {mountain} in the winter.",
            "The expedition to {mountain} was incredibly dangerous.",
            "I saw {mountain} from the airplane window.",
            "{mountain} is one of the highest peaks in the world.",
            "They set up base camp near {mountain}.",
            "Conquering {mountain} takes months of preparation."
        ]

        # 2. Негативні шаблони (дистрактори - країни, загальні слова, займенники)
        self.distractors = [
            "Nepal", "China", "Switzerland", "the hill", "a small hill", 
            "the river", "the city", "my house", "Europe", "Asia",
            "the valley", "the lake", "London", "Paris", "Kyiv"
        ]
        
        self.negative_templates = [
            "Last year I visited {distractor} to see the sights.",
            "I want to conquer my fears and maybe {distractor}.",
            "The expedition to {distractor} was cancelled.",
            "I saw {distractor} from the airplane window.",
            "{distractor} is very beautiful in the spring.",
            "I climbed a small tree near {distractor}.",
            "We walked around {distractor} all day."
        ]

    def _tokenize(self, text: str) -> List[str]:
        """Універсальний токенізатор для ідеального збігу з CRF та Baseline."""
        return re.findall(r"[\w']+|[.,!?;]", text)

    def generate_positive(self, num_per_mountain: int = 3) -> List[Dict[str, Any]]:
        """Генерує приклади зі справжніми горами (B-MOUNTAIN, I-MOUNTAIN)."""
        dataset = []
        for mountain in self.mountains:
            templates = random.choices(self.positive_templates, k=num_per_mountain)
            for template in templates:
                parts = template.split("{mountain}")
                if len(parts) != 2:
                    continue
                
                left_tokens = self._tokenize(parts[0])
                mountain_tokens = self._tokenize(mountain)
                right_tokens = self._tokenize(parts[1])

                tokens = []
                labels = []

                # Додаємо ліву частину
                tokens.extend(left_tokens)
                labels.extend(["O"] * len(left_tokens))

                # Додаємо саму гору
                if mountain_tokens:
                    tokens.append(mountain_tokens[0])
                    labels.append("B-MOUNTAIN")
                    for token in mountain_tokens[1:]:
                        tokens.append(token)
                        labels.append("I-MOUNTAIN")

                # Додаємо праву частину
                tokens.extend(right_tokens)
                labels.extend(["O"] * len(right_tokens))

                dataset.append({"tokens": tokens, "labels": labels})
                
        return dataset

    def generate_negative(self, num_samples: int = 1000) -> List[Dict[str, Any]]:
        """Генерує негативні приклади, де всі токени отримують лейбл 'O'."""
        dataset = []
        for _ in range(num_samples):
            template = random.choice(self.negative_templates)
            distractor = random.choice(self.distractors)
            
            sentence = template.format(distractor=distractor)
            tokens = self._tokenize(sentence)
            labels = ["O"] * len(tokens)
            
            dataset.append({"tokens": tokens, "labels": labels})
            
        return dataset