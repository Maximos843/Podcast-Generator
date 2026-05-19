from __future__ import annotations

import uuid
from dataclasses import dataclass
from time import perf_counter
from typing import Any, Optional

from src.types import PipelineRequest, PipelineResponse, PipelineTimingsMs
from src.retrieval.qdrant_retriever import QdrantRetriever
from src.generation.fact_checking import (
    build_fact_cards_for_retrieved,
    fact_check_script,
)
from src.generation.fact_refs import check_fact_refs
from src.generation.prompts import SYSTEM_STRICT_REFS_GENERATION, SYSTEM_SCRIPT_GENERATION
from src.generation.script_generation import (
    build_outline_and_script_prompt_strict_refs,
    generate_outline_and_script,
)
from src.pipeline.policy import apply_policy


@dataclass(frozen=True)
class PipelineDeps:
    client: Any
    embedder: Any
    collection_name: str
    reranker: Any
    article_store: Any
    llm: Any


def _repair_with_factcheck(llm, query: str, script: str, report) -> str:
    if not report or not report.unsupported:
        return script

    unsupported_dump = [u.model_dump() for u in report.unsupported]
    prompt = f"""Запрос пользователя:
{query}

Ниже сценарий:
{script}

Ниже неподтверждённые утверждения:
{unsupported_dump}

Задача:
Перепиши только неподтверждённые места, сохрани стиль и структуру.
Удаляй или исправляй неподтверждённые утверждения.
Не добавляй новых исторических фактов.

Верни строго JSON:
{{
  "script": "..."
}}
"""
    out = llm.generate(prompt, system=SYSTEM_SCRIPT_GENERATION, task="repair")
    from src.generation.json_extract import extract_json_object
    obj = extract_json_object(out)
    repaired = str(obj.get("script", "")).strip()
    return repaired or script


class PipelineService:
    def __init__(self, deps: PipelineDeps):
        self.deps = deps
        self.retriever = QdrantRetriever(
            client=deps.client,
            embedder=deps.embedder,
            collection_name=deps.collection_name,
            reranker=deps.reranker,
        )

    def generate(self, req: PipelineRequest, request_id: Optional[str] = None) -> PipelineResponse:
        req = apply_policy(req)
        rid = request_id or str(uuid.uuid4())

        t0 = perf_counter()
        timings = PipelineTimingsMs()

        t = perf_counter()
        hits = self.retriever.retrieve(req)
        timings.retrieval_ms = int((perf_counter() - t) * 1000)

        t = perf_counter()
        fact_cards = build_fact_cards_for_retrieved(
            llm=self.deps.llm,
            article_store=self.deps.article_store,
            retrieved_articles=hits,
            max_articles=req.max_articles_for_facts,
            request=req,
        )
        timings.fact_cards_ms = int((perf_counter() - t) * 1000)

        t = perf_counter()
        outline, script = generate_outline_and_script(self.deps.llm, req.query, fact_cards)

        ref_check = check_fact_refs(script, fact_cards)
        if not ref_check.ok:
            strict_prompt = build_outline_and_script_prompt_strict_refs(req.query, fact_cards)
            strict_out = strict_out = self.deps.llm.generate(
                strict_prompt,
                system=SYSTEM_STRICT_REFS_GENERATION,
                task="strict_refs",
            )
            from src.generation.json_extract import extract_json_object
            strict_obj = extract_json_object(strict_out)
            strict_script = str(strict_obj.get("script", "")).strip() if isinstance(strict_obj, dict) else ""
            if strict_script:
                strict_ref_check = check_fact_refs(strict_script, fact_cards)
                if strict_ref_check.ok:
                    script = strict_script

        timings.generation_ms = int((perf_counter() - t) * 1000)

        t = perf_counter()
        report = None
        if req.mode == "quality":
            report = fact_check_script(self.deps.llm, script, fact_cards, req.query)

            if report and report.unsupported:
                repaired_script = _repair_with_factcheck(self.deps.llm, req.query, script, report)
                repaired_ref_check = check_fact_refs(repaired_script, fact_cards)

                if repaired_ref_check.ok:
                    script = repaired_script
                    report = fact_check_script(self.deps.llm, script, fact_cards, req.query)

        timings.fact_check_ms = int((perf_counter() - t) * 1000)
        timings.total_ms = int((perf_counter() - t0) * 1000)

        debug = None
        if req.include_debug:
            debug = {
                "hits_preview": [h.model_dump() for h in hits[:3]],
                "fact_cards_cnt": len(fact_cards),
            }

        return PipelineResponse(
            request_id=rid,
            hits=hits,
            fact_cards=fact_cards,
            outline=outline,
            script=script,
            fact_check=report,
            timings=timings,
            debug=debug,
        )