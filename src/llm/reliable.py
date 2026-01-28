from __future__ import annotations

from dataclasses import dataclass
from typing import Type, TypeVar, Optional

from pydantic import BaseModel, ValidationError

from src.generation.json_extract import extract_json_object
from src.llm.base import LLM


T = TypeVar("T", bound=BaseModel)


@dataclass(frozen=True)
class LLMJsonPolicy:
    max_attempts: int = 2  # 1 + 1 retry
    system_hint_on_retry: str = "Верни СТРОГО валидный JSON без пояснений и без текста вокруг."


class ReliableLLM:
    def __init__(self, llm: LLM, policy: Optional[LLMJsonPolicy] = None):
        self.llm = llm
        self.policy = policy or LLMJsonPolicy()

    def generate(self, prompt: str, system: str | None = None) -> str:
        return self.llm.generate(prompt, system=system)

    def generate_json(self, prompt: str, model: Type[T], system: str | None = None) -> T:
        last_err: Exception | None = None
        sys = system
        for attempt in range(1, self.policy.max_attempts + 1):
            out = self.llm.generate(prompt, system=sys)
            try:
                obj = extract_json_object(out)
                return model.model_validate(obj)
            except (ValueError, ValidationError) as e:
                last_err = e
                sys = (system or "") + "\n" + self.policy.system_hint_on_retry
        # если совсем плохо — поднимем читаемую ошибку
        raise RuntimeError(f"LLM JSON generation failed for {model.__name__}: {last_err}")
