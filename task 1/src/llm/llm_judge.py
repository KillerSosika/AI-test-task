import json
import urllib.request
from typing import List, Dict, Any

class LocalLLMJudge:
    """Uses a local LLM via Ollama to filter and validate NER predictions."""

    def __init__(self, model_name: str = "llama3", base_url: str = "http://localhost:11434/api/generate"):
        self.model_name = model_name
        self.base_url = base_url

    def filter_entities(self, candidates: List[str]) -> List[str]:
        """
        Takes a list of extracted entities and returns only the verified mountains.
        """
        if not candidates:
            return []

        # Формуємо жорсткий промпт для моделі, щоб вона віддавала ТІЛЬКИ JSON
        prompt = (
            "You are an expert geographer and data annotator. "
            f"Review this list of extracted geographical entities: {candidates}\n"
            "Your task is to identify and return ONLY the actual, specific mountain peaks.\n"
            "Rules:\n"
            "1. Filter out countries (e.g., Nepal, China).\n"
            "2. Filter out general nouns (e.g., hill, river, house, mountain).\n"
            "3. Keep ONLY specific mountain names (e.g., Mount Everest, K2, Matterhorn).\n"
            "4. Output strictly a JSON list of strings. Do NOT output any markdown formatting, explanations, or introductory text."
        )

        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "format": "json"  # Змушуємо Ollama віддавати чистий JSON
        }

        req = urllib.request.Request(
            self.base_url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"}
        )

        try:
            with urllib.request.urlopen(req, timeout=30) as response:
                result = json.loads(response.read().decode("utf-8"))
                response_text = result.get("response", "[]").strip()
                
                parsed = json.loads(response_text)
                
                if isinstance(parsed, dict):
                    for val in parsed.values():
                        if isinstance(val, list):
                            return val
                    return []
                    
                if isinstance(parsed, list):
                    return parsed
                return []
                
        except Exception as e:
            print(f"[!] LLM Judge failed: {e}")
            return candidates  