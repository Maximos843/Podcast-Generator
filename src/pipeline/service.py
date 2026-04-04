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
    repair_script_with_fact_check_report,
)
from src.generation.fact_refs import check_fact_refs
from src.generation.prompts import SYSTEM_STRICT_REFS_GENERATION
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

        # 1. проверка ссылок
        ref_check = check_fact_refs(script, fact_cards)
        if not ref_check.ok:
            strict_prompt = build_outline_and_script_prompt_strict_refs(req.query, fact_cards)
            script2 = self.deps.llm.generate(strict_prompt, system=SYSTEM_STRICT_REFS_GENERATION)
            ref_check2 = check_fact_refs(script2, fact_cards)
            if ref_check2.ok:
                script = script2
            else:
                script += "\n\n[system] Предупреждение: обнаружены неизвестные ссылки на факты: " + ", ".join(
                    sorted(ref_check2.unknown)
                )

        timings.generation_ms = int((perf_counter() - t) * 1000)

        # 2. фактчекинг
        t = perf_counter()
        report = None
        if req.mode == "quality":
            report = fact_check_script(self.deps.llm, script, fact_cards, req.query)

            # 3. если есть unsupported — ремонтируем текст
            if report and report.unsupported:
                repaired_script = repair_script_with_fact_check_report(
                    llm=self.deps.llm,
                    query=req.query,
                    script=script,
                    fact_cards=fact_cards,
                    report=report,
                )

                # повторная проверка после repair
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