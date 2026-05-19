from typing import Literal
from pydantic import BaseModel, Field
from dataclasses import dataclass


class PipelineRequest(BaseModel):
    query: str
    year: int | None = None
    mode: Literal["fast", "quality"] = "quality"
    retrieval: Literal["dense", "hybrid"] = "hybrid"
    max_articles_for_facts: int = 7
    include_debug: bool = False


class RetrievedChunkHit(BaseModel):
    chunk_id: int | None = None
    text: str
    score: float
    year: int | None = None


class RetrievedArticleHit(BaseModel):
    article_id: str
    score: float
    year: int | None = None
    best_chunks: list[RetrievedChunkHit] = Field(default_factory=list)


class Fact(BaseModel):
    fact_id: str
    statement: str
    evidence_quote: str
    article_id: str


class FactCard(BaseModel):
    article_id: str
    year: int | None = None
    title_guess: str | None = None
    facts: list[Fact] = Field(default_factory=list)


class OutlineBlock(BaseModel):
    title: str
    goal: str
    facts_used: list[str] = Field(default_factory=list)
    transition: str | None = None


class Outline(BaseModel):
    outline: list[OutlineBlock] = Field(default_factory=list)


class UnsupportedClaim(BaseModel):
    claim: str
    why_unsupported: str
    suggested_fix: str | None = None


class FactCheckReport(BaseModel):
    unsupported: list[UnsupportedClaim] = Field(default_factory=list)


class PipelineTimingsMs(BaseModel):
    retrieval_ms: int = 0
    fact_cards_ms: int = 0
    generation_ms: int = 0
    fact_check_ms: int = 0
    total_ms: int = 0


class PipelineResponse(BaseModel):
    request_id: str
    hits: list[RetrievedArticleHit] = Field(default_factory=list)
    fact_cards: list[FactCard] = Field(default_factory=list)
    outline: Outline | None = None
    script: str = ""
    fact_check: FactCheckReport | None = None
    timings: PipelineTimingsMs = Field(default_factory=PipelineTimingsMs)
    debug: dict | None = None


@dataclass(frozen=True)
class Article:
    page_path: str
    page_idx: int
    article_id: str
    year: int | None
    original_text: str
    cleaned_text: str


@dataclass(frozen=True)
class Chunk:
    article: Article
    chunk_id: int
    text: str
