# src/pipeline/steps/retrieve_step.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List

from src.domain.contracts import PipelineRequest, RetrievedArticleHit
from src.retrieval.retriever import Retriever


@dataclass(frozen=True)
class RetrieveStep:
    retriever: Retriever

    def run(self, req: PipelineRequest) -> List[RetrievedArticleHit]:
        return self.retriever.retrieve(req)
