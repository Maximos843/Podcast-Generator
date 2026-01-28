# src/domain/contracts.py
from __future__ import annotations

from typing import List, Optional, Literal
from pydantic import BaseModel, Field


class PipelineRequest(BaseModel):
    query: str
    year: Optional[int] = None

    mode: Literal["fast", "quality"] = "quality"
    retrieval: Literal["dense", "hybrid"] = "hybrid"

    max_articles_for_facts: int = 7
    include_debug: bool = False


class RetrievedChunkHit(BaseModel):
    chunk_id: Optional[int] = None
    text: str
    score: float
    year: Optional[int] = None


class RetrievedArticleHit(BaseModel):
    article_id: str  # full_article_id
    score: float
    year: Optional[int] = None
    best_chunks: List[RetrievedChunkHit] = Field(default_factory=list)


class Fact(BaseModel):
    fact_id: str
    statement: str
    evidence_quote: str
    article_id: str


class FactCard(BaseModel):
    article_id: str
    year: Optional[int] = None
    title_guess: Optional[str] = None
    facts: List[Fact] = Field(default_factory=list)


class OutlineBlock(BaseModel):
    title: str
    goal: str
    facts_used: List[str] = Field(default_factory=list)


class Outline(BaseModel):
    outline: List[OutlineBlock]


class UnsupportedClaim(BaseModel):
    claim: str
    why_unsupported: str
    suggested_fix: Optional[str] = None


class FactCheckReport(BaseModel):
    unsupported: List[UnsupportedClaim] = Field(default_factory=list)


class PipelineTimingsMs(BaseModel):
    retrieval_ms: int = 0
    fact_cards_ms: int = 0
    generation_ms: int = 0
    fact_check_ms: int = 0
    total_ms: int = 0


class PipelineResponse(BaseModel):
    request_id: str
    hits: List[RetrievedArticleHit] = Field(default_factory=list)
    fact_cards: List[FactCard] = Field(default_factory=list)
    outline: Optional[Outline] = None
    script: str = ""
    fact_check: Optional[FactCheckReport] = None
    timings: PipelineTimingsMs = Field(default_factory=PipelineTimingsMs)
    debug: Optional[dict] = None
