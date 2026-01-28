from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List

from src.domain.contracts import PipelineRequest, RetrievedArticleHit, RetrievedChunkHit
from src.retrieval.retrieval import retrieve_articles  # твоя текущая функция


@dataclass(frozen=True)
class QdrantRetriever:
    client: Any
    embedder: Any
    collection_name: str
    reranker: Any = None

    def retrieve(self, req: PipelineRequest) -> List[RetrievedArticleHit]:
        raw = retrieve_articles(
            client=self.client,
            embedder=self.embedder,
            query_text=req.query,
            collection_name=self.collection_name,
            mode=req.mode,
            retrieval=req.retrieval,
            year=req.year,
            reranker=self.reranker if req.mode == "quality" else None,
        )

        hits: List[RetrievedArticleHit] = []
        for row in raw:
            best_chunks = [
                RetrievedChunkHit(
                    chunk_id=ch.get("chunk_id"),
                    text=ch.get("text", ""),
                    score=float(ch.get("score", 0.0)),
                    year=ch.get("year"),
                )
                for ch in (row.get("best_chunks") or [])
            ]
            hits.append(
                RetrievedArticleHit(
                    article_id=row.get("article_id", ""),
                    score=float(row.get("score", 0.0)),
                    year=row.get("year"),
                    best_chunks=best_chunks,
                )
            )
        return hits
