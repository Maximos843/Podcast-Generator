# src/pipeline/steps/generate_step.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List

from src.domain.contracts import FactCard, Outline, PipelineRequest
from src.generation.script_generation import generate_outline, generate_script


@dataclass(frozen=True)
class GenerateStep:
    llm: Any

    def run(self, req: PipelineRequest, fact_cards: List[FactCard]) -> tuple[Outline, str]:
        outline = generate_outline(self.llm, req.query, fact_cards)
        script = generate_script(self.llm, req.query, outline, fact_cards)
        return outline, script
