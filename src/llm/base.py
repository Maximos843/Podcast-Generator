from __future__ import annotations

from typing import Optional


class LLM:
    def generate(
        self,
        prompt: str,
        system: str = "Ты — полезный ассистент.",
        *,
        task: Optional[str] = None,
        temperature: Optional[float] = None,
    ) -> str:
        raise NotImplementedError