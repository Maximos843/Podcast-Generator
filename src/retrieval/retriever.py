# src/retrieval/retriever.py
from __future__ import annotations

from abc import ABC, abstractmethod

from src.types import RetrievedArticleHit, PipelineRequest


class Retriever(ABC):
    @abstractmethod
    def retrieve(self, req: PipelineRequest) -> list[RetrievedArticleHit]:
        raise NotImplementedError
