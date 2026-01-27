# src/pipeline/service.py
from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Optional, Dict, Any
import uuid

from src.domain.types import (
    PipelineRequest,
    PipelineResponse,
    PipelineTimingsMs,
    coerce_hits,
)
from src.retrieval.retrieval import retrieve_articles
from src.generation.fact_checking import build_fact_cards_for_retrieved, fact_check_script
from src.generation.script_generation import generate_outline, generate_script


@dataclass(frozen=True)
class PipelineDeps:
    client: Any
    embedder: Any
    collection_name: str
    reranker: Any
    article_store: Any
    llm: Any


class PipelineService:
    """
    Сервисная оболочка вокруг твоего текущего пайплайна.
    Логику не меняем: retrieval -> fact_cards -> outline/script -> fact_check.
    """

    def __init__(self, deps: PipelineDeps):
        self.deps = deps

    def generate(self, req: PipelineRequest, request_id: Optional[str] = None) -> PipelineResponse:
        rid = request_id or str(uuid.uuid4())
        t0 = perf_counter()

        # 1) retrieval
        t = perf_counter()
        raw_hits = retrieve_articles(
            client=self.deps.client,
            embedder=self.deps.embedder,
            query_text=req.query,
            collection_name=self.deps.collection_name,
            mode=req.mode,
            retrieval=req.retrieval,
            year=req.year,
            reranker=self.deps.reranker if req.mode == "quality" else None,
        )
        retrieval_ms = int((perf_counter() - t) * 1000)

        # 2) fact cards
        t = perf_counter()
        fact_cards = build_fact_cards_for_retrieved(
            llm=self.deps.llm,
            article_store=self.deps.article_store,
            retrieved_articles=raw_hits,  # IMPORTANT: пока оставляем как есть (dict), чтобы не трогать генерацию
            max_articles=req.max_articles_for_facts,
        )
        fact_cards_ms = int((perf_counter() - t) * 1000)

        # 3) generation
        t = perf_counter()
        outline = generate_outline(self.deps.llm, req.query, fact_cards)
        script = generate_script(self.deps.llm, req.query, outline, fact_cards)
        generation_ms = int((perf_counter() - t) * 1000)

        # 4) fact check
        t = perf_counter()
        report = fact_check_script(self.deps.llm, script, fact_cards)
        fact_check_ms = int((perf_counter() - t) * 1000)

        total_ms = int((perf_counter() - t0) * 1000)

        debug: Optional[Dict[str, Any]] = None
        if req.include_debug:
            debug = {
                "raw_hits_preview": raw_hits[:3],
            }

        return PipelineResponse(
            request_id=rid,
            hits=coerce_hits(raw_hits),
            fact_cards=fact_cards,
            outline=outline,
            script=script,
            fact_check=report,
            timings=PipelineTimingsMs(
                retrieval_ms=retrieval_ms,
                fact_cards_ms=fact_cards_ms,
                generation_ms=generation_ms,
                fact_check_ms=fact_check_ms,
                total_ms=total_ms,
            ),
            debug=debug,
        )
