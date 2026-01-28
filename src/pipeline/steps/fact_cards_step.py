# src/pipeline/steps/fact_cards_step.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List

from src.domain.contracts import FactCard, PipelineRequest, RetrievedArticleHit
from src.generation.fact_checking import build_fact_cards_for_retrieved


@dataclass(frozen=True)
class FactCardsStep:
    llm: Any
    article_store: Any

    def run(self, req: PipelineRequest, hits: List[RetrievedArticleHit]) -> List[FactCard]:
        return build_fact_cards_for_retrieved(
            llm=self.llm,
            article_store=self.article_store,
            retrieved_articles=hits,
            max_articles=req.max_articles_for_facts,
        )
