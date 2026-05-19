from dataclasses import dataclass, field
from typing import Any

from src.types import PipelineRequest, RetrievedArticleHit, FactCard, Outline, FactCheckReport


@dataclass
class PipelineContext:
    request_id: str
    request: PipelineRequest

    raw_hits: list[dict[str, Any]] = field(default_factory=list)
    hits: list[RetrievedArticleHit] = field(default_factory=list)

    fact_cards: list[FactCard] = field(default_factory=list)
    outline: Outline | None = None
    script: str = ""
    fact_check: FactCheckReport | None = None

    debug: dict[str, Any] = field(default_factory=dict)
