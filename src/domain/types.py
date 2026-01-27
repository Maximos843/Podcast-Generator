# src/domain/types.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Literal, List, Dict, Any


# --- existing domain models (оставляем совместимость с текущим retrieval/chunking) ---

@dataclass(frozen=True)
class Article:
    page_path: str
    page_idx: int
    article_id: str
    year: Optional[int]
    original_text: str
    cleaned_text: str


@dataclass(frozen=True)
class Chunk:
    article: Article
    chunk_id: int
    text: str


# --- new service-level types (для API/контрактов) ---

@dataclass(frozen=True)
class RetrievedChunkHit:
    chunk_id: Optional[int]
    text: str
    score: float
    year: Optional[int] = None


@dataclass(frozen=True)
class RetrievedArticleHit:
    article_id: str
    score: float
    year: Optional[int]
    best_chunks: List[RetrievedChunkHit]


@dataclass(frozen=True)
class PipelineRequest:
    query: str
    year: Optional[int] = None

    # режимы твоего retrieval.py
    mode: Literal["fast", "quality"] = "quality"
    retrieval: Literal["dense", "hybrid"] = "hybrid"

    # knobs пайплайна
    max_articles_for_facts: int = 7

    # debug
    include_debug: bool = False


@dataclass(frozen=True)
class PipelineTimingsMs:
    retrieval_ms: int
    fact_cards_ms: int
    generation_ms: int
    fact_check_ms: int
    total_ms: int


@dataclass(frozen=True)
class PipelineResponse:
    request_id: str

    hits: List[RetrievedArticleHit]
    fact_cards: Any
    outline: Any
    script: Any
    fact_check: Any

    timings: PipelineTimingsMs
    debug: Optional[Dict[str, Any]] = None


def coerce_hits(raw_hits: List[Dict[str, Any]]) -> List[RetrievedArticleHit]:
    """
    Преобразуем текущий формат retrieve_articles() (dict-ы)
    в типизированный список для ответа API.
    """
    out: List[RetrievedArticleHit] = []
    for row in raw_hits or []:
        best_chunks = []
        for ch in row.get("best_chunks", []) or []:
            best_chunks.append(
                RetrievedChunkHit(
                    chunk_id=ch.get("chunk_id"),
                    text=ch.get("text", ""),
                    score=float(ch.get("score", 0.0)),
                    year=ch.get("year"),
                )
            )
        out.append(
            RetrievedArticleHit(
                article_id=row.get("article_id", ""),
                score=float(row.get("score", 0.0)),
                year=row.get("year"),
                best_chunks=best_chunks,
            )
        )
    return out
