# chunks_utils.py
from __future__ import annotations

import gc
import logging
import uuid
from collections import defaultdict
from typing import Optional, Literal, List, Dict, Any

import torch
from tqdm.auto import tqdm
from qdrant_client import QdrantClient, models
from qdrant_client.models import (
    Distance, VectorParams, SparseVectorParams, Modifier,
    Filter, FieldCondition, Range,
)

from src.config import QDrantConfig
from src.storage.preprocessing import chunk_text_by_sentences
from src.retrieval.model import BaseEmbedder, BaseReranker
from src.types import Article, Chunk
from src.retrieval.payload_schema import FULL_ARTICLE_ID, CHUNK_ID, TEXT, YEAR, REQUIRED_KEYS


logger = logging.getLogger("rag-service")



def make_chunks(articles: List[Article]) -> List[Chunk]:
    chunks: List[Chunk] = []
    for art in tqdm(articles, total=len(articles), desc="chunking articles"):
        texts = chunk_text_by_sentences(art.cleaned_text)
        for idx, txt in enumerate(texts):
            if txt.strip():
                chunks.append(Chunk(article=art, chunk_id=idx, text=txt))
    return chunks



def _year_filter(year: Optional[int]) -> Optional[Filter]:
    if year is None:
        return None
    return Filter(must=[FieldCondition(key="year", range=Range(gte=year, lte=year))])


def search_chunks_dense(
    client: QdrantClient,
    embedder: BaseEmbedder,
    query_text: str,
    collection_name: str,
    top_k_chunks: int,
    year: Optional[int] = None,
    using: Optional[str] = None,
):
    qvec = embedder.embed_query(query_text).tolist()
    res = client.query_points(
        collection_name=collection_name,
        query=qvec,
        using=using,
        query_filter=_year_filter(year),
        limit=top_k_chunks,
        with_payload=True,
    )
    return res.points


def search_chunks_hybrid_rrf(
    client: QdrantClient,
    embedder: BaseEmbedder,
    query_text: str,
    collection_name: str,
    top_k_chunks: int,
    prefetch_k: int,
    year: Optional[int] = None,
):
    dense_q = embedder.embed_query(query_text).tolist()
    q_filter = _year_filter(year)

    res = client.query_points(
        collection_name=collection_name,
        prefetch=[
            models.Prefetch(
                using=QDrantConfig.BM25_VECTOR_NAME,
                query=models.Document(text=query_text, model=QDrantConfig.BM25_MODEL),
                limit=prefetch_k,
                filter=q_filter,
            ),
            models.Prefetch(
                using=QDrantConfig.DENSE_VECTOR_NAME,
                query=dense_q,
                limit=prefetch_k,
                filter=q_filter,
            ),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=top_k_chunks,
        with_payload=True,
    )
    return res.points



def _payload_has_required(payload: dict) -> bool:
    for k in REQUIRED_KEYS:
        v = payload.get(k)
        if v is None or (isinstance(v, str) and not v.strip()):
            return False
    return True


def _aggregate_points_to_articles(
    points: List[models.ScoredPoint],
    *,
    top_k_articles: int,
    per_article_top_chunks: int,
    score_agg: Literal["max", "sum"] = "max",
) -> List[Dict[str, Any]]:
    by_article: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    bad_payload_cnt = 0

    for p in points:
        payload = p.payload or {}
        if not _payload_has_required(payload):
            bad_payload_cnt += 1
            continue

        art_id = payload.get(FULL_ARTICLE_ID)
        by_article[art_id].append(
            {
                "chunk_id": payload.get(CHUNK_ID),
                "text": payload.get(TEXT, ""),
                "score": float(p.score),
                "year": payload.get(YEAR),
            }
        )

    if bad_payload_cnt:
        logger.warning("qdrant_payload_invalid", extra={"bad_payload_cnt": bad_payload_cnt})

    rows: List[Dict[str, Any]] = []
    for art_id, lst in by_article.items():
        scores = [x["score"] for x in lst]
        art_score = sum(scores) if score_agg == "sum" else (max(scores) if scores else 0.0)
        best_chunks = sorted(lst, key=lambda x: x["score"], reverse=True)[:per_article_top_chunks]
        year = best_chunks[0].get("year") if best_chunks else None

        rows.append(
            {
                "article_id": art_id,
                "score": float(art_score),
                "year": year,
                "best_chunks": best_chunks,
            }
        )

    rows.sort(key=lambda x: x["score"], reverse=True)
    return rows[:top_k_articles]



def _rerank_chunks(
    reranker: BaseReranker,
    query_text: str,
    points: List[models.ScoredPoint],
    *,
    batch_size: int = 16,
) -> List[models.ScoredPoint]:
    texts = [(p.payload or {}).get("text", "") for p in points]
    scores = reranker.score(query_text, texts, batch_size=batch_size)

    out: List[models.ScoredPoint] = []
    for p, s in zip(points, scores):
        out.append(models.ScoredPoint(
            id=p.id,
            version=p.version,
            score=float(s),
            payload=p.payload,
            vector=p.vector,
            shard_key=p.shard_key,
            order_value=p.order_value,
        ))
    out.sort(key=lambda x: x.score, reverse=True)
    return out


def retrieve_articles(
    client: QdrantClient,
    embedder: BaseEmbedder,
    query_text: str,
    collection_name: str,
    mode: Literal["fast", "quality"] = "quality",
    retrieval: Literal["dense", "hybrid"] = "hybrid",
    top_k_articles: Optional[int] = None,
    candidate_pool_chunks: Optional[int] = None,
    prefetch_k: Optional[int] = None,
    per_article_top_chunks: Optional[int] = None,
    year: Optional[int] = None,
    score_agg: Literal["max", "sum"] = "max",
    reranker: Optional[BaseReranker] = None,
    rerank_batch_size: int = 16,
):
    if prefetch_k is None:
        prefetch_k = QDrantConfig.PREFETCH_K
    if per_article_top_chunks is None:
        per_article_top_chunks = QDrantConfig.PER_ARTICLE_TOP_CHUNKS

    if mode == "fast":
        if candidate_pool_chunks is None:
            candidate_pool_chunks = QDrantConfig.CANDIDATE_POOL_CHUNKS_FAST
        if top_k_articles is None:
            top_k_articles = QDrantConfig.TOP_K_ARTICLES_FAST
    else:
        if candidate_pool_chunks is None:
            candidate_pool_chunks = QDrantConfig.CANDIDATE_POOL_CHUNKS_QUALITY
        if top_k_articles is None:
            top_k_articles = QDrantConfig.TOP_K_ARTICLES_QUALITY

    if retrieval == "dense":
        points = search_chunks_dense(
            client, embedder, query_text, collection_name,
            top_k_chunks=candidate_pool_chunks,
            year=year,
            using=QDrantConfig.DENSE_VECTOR_NAME,
        )
    else:
        points = search_chunks_hybrid_rrf(
            client, embedder, query_text, collection_name,
            top_k_chunks=candidate_pool_chunks,
            prefetch_k=prefetch_k,
            year=year,
        )

    if mode == "quality" and reranker is not None and points:
        points = _rerank_chunks(reranker, query_text, points, batch_size=rerank_batch_size)

    return _aggregate_points_to_articles(
        points,
        top_k_articles=top_k_articles,
        per_article_top_chunks=per_article_top_chunks,
        score_agg=score_agg,
    )
