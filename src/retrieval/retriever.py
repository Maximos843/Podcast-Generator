# src/retrieval/retriever.py
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, List

from src.domain.contracts import RetrievedArticleHit, PipelineRequest


class Retriever(ABC):
    @abstractmethod
    def retrieve(self, req: PipelineRequest) -> List[RetrievedArticleHit]:
        ...
