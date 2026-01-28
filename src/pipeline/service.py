from __future__ import annotations

import uuid
from dataclasses import dataclass
from time import perf_counter
from typing import Any, Optional

from src.domain.contracts import PipelineRequest, PipelineResponse, PipelineTimingsMs
from src.retrieval.qdrant_retriever import QdrantRetriever
from src.generation.fact_checking import build_fact_cards_for_retrieved, fact_check_script
from src.generation.script_generation import generate_outline, generate_script
from src.generation.fact_refs import check_fact_refs
from src.generation.script_generation import build_script_prompt_strict_refs
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
        )
        timings.fact_cards_ms = int((perf_counter() - t) * 1000)

        t = perf_counter()
        outline = generate_outline(self.deps.llm, req.query, fact_cards)
        script = generate_script(self.deps.llm, req.query, outline, fact_cards)
        timings.generation_ms = int((perf_counter() - t) * 1000)
        ref_check = check_fact_refs(script, fact_cards)
        if not ref_check.ok:
            # 1 retry: перегенерить сценарий со строгим списком fact_id
            strict_prompt = build_script_prompt_strict_refs(req.query, outline.model_dump(), fact_cards)
            script2 = self.deps.llm.generate(strict_prompt, system="Ты строго следуешь списку fact_id, не выдумываешь.")
            ref_check2 = check_fact_refs(script2, fact_cards)
            if ref_check2.ok:
                script = script2
            else:
                # мягкая деградация: добавим предупреждение в конец
                script += "\n\n[system] Предупреждение: обнаружены неизвестные ссылки на факты: " + ", ".join(sorted(ref_check2.unknown))

        t = perf_counter()
        report = None
        if req.mode == "quality":
            report = fact_check_script(self.deps.llm, script, fact_cards)
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
