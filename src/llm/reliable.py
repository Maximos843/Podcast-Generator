from dataclasses import dataclass
from typing import Type, TypeVar
from pydantic import BaseModel, ValidationError

from src.generation.json_extract import extract_json_object
from src.llm.base import LLM

T = TypeVar("T", bound=BaseModel)


@dataclass(frozen=True)
class LLMJsonPolicy:
    max_attempts: int = 2
    system_hint_on_retry: str = "Верни СТРОГО валидный JSON без пояснений и без текста вокруг."


class ReliableLLM(LLM):
    def __init__(self, llm: LLM, policy: LLMJsonPolicy | None = None):
        self.llm = llm
        self.policy = policy or LLMJsonPolicy()

    def generate(
        self,
        prompt: str,
        system: str | None = None,
        *,
        task: str | None = None,
        temperature: float | None = None,
    ) -> str:
        return self.llm.generate(
            prompt,
            system=system or "Ты — полезный ассистент.",
            task=task,
            temperature=temperature,
        )

    def generate_json(
        self,
        prompt: str,
        model: Type[T],
        system: str | None = None,
        *,
        task: str | None = None,
        temperature: float | None = None,
    ) -> T:
        last_err: Exception | None = None
        sys = system

        for _ in range(1, self.policy.max_attempts + 1):
            out = self.llm.generate(
                prompt,
                system=sys or "Ты — полезный ассистент.",
                task=task,
                temperature=temperature,
            )
            try:
                obj = extract_json_object(out)
                return model.model_validate(obj)
            except (ValueError, ValidationError) as e:
                last_err = e
                sys = (system or "") + "\n" + self.policy.system_hint_on_retry

        raise RuntimeError(f"LLM JSON generation failed for {model.__name__}: {last_err}")
