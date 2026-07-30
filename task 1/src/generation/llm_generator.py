from typing import Any, Callable, Dict, Iterable, List, Optional
from .base import BaseTextGenerator
from .template_generator import TemplateGenerator

class LLMGenerator(BaseTextGenerator):
    """Optional generation layer that can enrich text via an LLM client."""

    def __init__(
        self,
        llm_client: Optional[Callable[[str], str]] = None,
        fallback_generator: Optional[BaseTextGenerator] = None,
    ) -> None:
        self._llm_client = llm_client
        self._fallback_generator = fallback_generator or TemplateGenerator()

    def generate(self, text: Optional[str] = None, **context: Any) -> str:
        if self._llm_client is None:
            return self._fallback_generator.generate(**context)

        prompt = text or self._build_prompt(context)
        return self._llm_client(prompt)

    def generate_batch(
        self,
        batch_size: int = 1,
        texts: Optional[Iterable[str]] = None,
        **context: Any,
    ) -> List[str]:
        if texts is not None:
            return [self.generate(text=text, **context) for text in texts]
        return super().generate_batch(batch_size=batch_size, **context)

    def _build_prompt(self, context: Dict[str, Any]) -> str:
        if not context:
            return "Create a polished text from the given context."

        details = ", ".join(f"{key}={value}" for key, value in sorted(context.items()))
        return f"Create a polished paraphrase or richer wording using these details: {details}"