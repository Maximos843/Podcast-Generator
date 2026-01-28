# src/pipeline/steps/fact_check_step.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List

from src.domain.contracts import FactCard, FactCheckReport
from src.generation.fact_checking import fact_check_script


@dataclass(frozen=True)
class FactCheckStep:
    llm: Any

    def run(self, script: str, fact_cards: List[FactCard]) -> FactCheckReport:
        return fact_check_script(self.llm, script, fact_cards)
