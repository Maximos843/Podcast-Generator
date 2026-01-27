# src/service/schemas.py
from __future__ import annotations

from typing import Optional, Literal, Any, List, Dict
from pydantic import BaseModel, Field


class GenerateRequest(BaseModel):
    query: str
    year: Optional[int] = None

    mode: Literal["fast", "quality"] = "quality"
    retrieval: Literal["dense", "hybrid"] = "hybrid"

    max_articles_for_facts: int = 7
    include_debug: bool = False


class RetrievedChunkHitOut(BaseModel):
    chunk_id: Optional[int] = None
    text: str
    score: float
    year: Optional[int] = None


class RetrievedArticleHitOut(BaseModel):
    article_id: str
    score: float
    year: Optional[int] = None
    best_chunks: List[RetrievedChunkHitOut] = Field(default_factory=list)


class TimingsOut(BaseModel):
    retrieval_ms: int
    fact_cards_ms: int
    generation_ms: int
    fact_check_ms: int
    total_ms: int


class GenerateResponse(BaseModel):
    request_id: str
    hits: List[RetrievedArticleHitOut] = Field(default_factory=list)

    # На первой итерации оставляем Any, чтобы не трогать генерацию/фактчек
    fact_cards: Any
    outline: Any
    script: Any
    fact_check: Any

    timings: TimingsOut
    debug: Optional[Dict[str, Any]] = None
