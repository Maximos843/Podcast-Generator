# src/pipeline/context.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, List, Dict

from src.types import PipelineRequest, RetrievedArticleHit, FactCard, Outline, FactCheckReport


@dataclass
class PipelineContext:
    request_id: str
    request: PipelineRequest

    raw_hits: List[Dict[str, Any]] = field(default_factory=list)  # как возвращает retrieve_articles()
    hits: List[RetrievedArticleHit] = field(default_factory=list)

    fact_cards: List[FactCard] = field(default_factory=list)
    outline: Optional[Outline] = None
    script: str = ""
    fact_check: Optional[FactCheckReport] = None

    debug: Dict[str, Any] = field(default_factory=dict)
