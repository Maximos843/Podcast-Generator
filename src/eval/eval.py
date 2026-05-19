# eval.py
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import List, Dict, Any, Iterable, Optional, Tuple
from collections import defaultdict

from qdrant_client import QdrantClient

from src.retrieval.model import BaseEmbedder, BaseReranker
from src.retrieval.retrieval import retrieve_articles


@dataclass(frozen=True)
class EvalQuery:
    query_text: str
    relevant_articles: set[str]
    source: str
    relevance_grades: Optional[dict[str, int]] = None


@dataclass(frozen=True)
class EvalMetrics:
    mrr: float
    hit_at_k: float
    recall_at_k: float
    precision_at_k: float


def load_eval_queries_from_file(path: str) -> List[EvalQuery]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    out: List[EvalQuery] = []
    for row in data:
        out.append(EvalQuery(
            query_text=row["query"],
            relevant_articles=set(row["relevant_articles"]),
            source=row.get("source", "unknown"),
            relevance_grades=row.get("relevance_grades"),
        ))
    return out


def _compute_metrics_article_level(
    ranked_articles: List[str],
    relevant_articles: set[str],
    k: int,
) -> Tuple[float, float, float, float]:
    topk = ranked_articles[:k]
    if not relevant_articles:
        return 0.0, 0.0, 0.0, 0.0

    hit = 1.0 if any(a in relevant_articles for a in topk) else 0.0

    rr = 0.0
    for i, a in enumerate(topk, start=1):
        if a in relevant_articles:
            rr = 1.0 / i
            break

    found = sum(1 for a in topk if a in relevant_articles)
    recall = found / len(relevant_articles)
    precision = found / k
    return rr, hit, recall, precision


def evaluate_retriever_on_queries(
    client: QdrantClient,
    queries: Iterable[EvalQuery],
    embedder: BaseEmbedder,
    collection_name: str,
    *,
    retrieval: str = "hybrid",             # "dense" | "hybrid"
    mode: str = "fast",                    # "fast" | "quality"
    reranker: Optional[BaseReranker] = None,
    top_k_articles: int = 10,
    candidate_pool_chunks: int = 80,
    prefetch_k: int = 150,
    per_article_top_chunks: int = 3,
    rerank_batch_size: int = 16,
):
    queries = list(queries)
    if not queries:
        return EvalMetrics(0.0, 0.0, 0.0, 0.0)

    mrr_sum = hit_sum = recall_sum = precision_sum = 0.0

    for q in queries:
        hits = retrieve_articles(
            client=client,
            embedder=embedder,
            query_text=q.query_text,
            collection_name=collection_name,
            mode=mode,
            retrieval=retrieval,
            top_k_articles=top_k_articles,
            candidate_pool_chunks=candidate_pool_chunks,
            prefetch_k=prefetch_k,
            per_article_top_chunks=per_article_top_chunks,
            reranker=reranker,
            rerank_batch_size=rerank_batch_size,
        )
        ranked_ids = [h["article_id"] for h in hits]
        rr, hit, recall, precision = _compute_metrics_article_level(
            ranked_ids, q.relevant_articles, k=top_k_articles
        )
        mrr_sum += rr
        hit_sum += hit
        recall_sum += recall
        precision_sum += precision

    n = len(queries)
    return EvalMetrics(
        mrr=mrr_sum / n,
        hit_at_k=hit_sum / n,
        recall_at_k=recall_sum / n,
        precision_at_k=precision_sum / n,
    )


def evaluate_by_source(
    client: QdrantClient,
    eval_queries: List[EvalQuery],
    embedder: BaseEmbedder,
    collection_name: str,
    *,
    retrieval: str = "hybrid",
    mode: str = "fast",
    reranker: Optional[BaseReranker] = None,
    top_k_articles: int = 10,
    candidate_pool_chunks: int = 80,
    prefetch_k: int = 150,
    per_article_top_chunks: int = 3,
):
    overall = evaluate_retriever_on_queries(
        client, eval_queries, embedder, collection_name,
        retrieval=retrieval,
        mode=mode,
        reranker=reranker,
        top_k_articles=top_k_articles,
        candidate_pool_chunks=candidate_pool_chunks,
        prefetch_k=prefetch_k,
        per_article_top_chunks=per_article_top_chunks,
    )

    grouped: Dict[str, List[EvalQuery]] = defaultdict(list)
    for q in eval_queries:
        grouped[q.source].append(q)

    per_source = {
        src: evaluate_retriever_on_queries(
            client, qs, embedder, collection_name,
            retrieval=retrieval,
            mode=mode,
            reranker=reranker,
            top_k_articles=top_k_articles,
            candidate_pool_chunks=candidate_pool_chunks,
            prefetch_k=prefetch_k,
            per_article_top_chunks=per_article_top_chunks,
        )
        for src, qs in grouped.items()
    }
    return overall, per_source
