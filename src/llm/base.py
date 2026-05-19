from abc import ABC, abstractmethod


class LLM(ABC):
    @abstractmethod
    def generate(
        self,
        prompt: str,
        system: str = "Ты — полезный ассистент.",
        *,
        task: str | None = None,
        temperature: float | None = None,
    ) -> str:
        raise NotImplementedError