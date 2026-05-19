import logging
import time
from dataclasses import dataclass
from typing import Literal

from qdrant_client import QdrantClient

from src.storage.preprocessing import load_articles_from_json
from src.retrieval.retrieval import make_chunks
from src.retrieval.model import BaseEmbedder
from src.retrieval.qdrant_index import (
    recreate_collection_dense,
    recreate_collection_hybrid,
    index_chunks_in_qdrant,
)
from src.storage.article_store import SQLiteArticleStore, ArticleRecord, make_full_article_id

logger = logging.getLogger("rag-indexer")


class StageTimer:
    def __init__(self, name: str):
        self.name = name
        self.t0 = None

    def __enter__(self):
        self.t0 = time.perf_counter()
        logger.info("stage_start", extra={"stage": self.name})
        return self

    def __exit__(self, exc_type, exc, tb):
        dt = time.perf_counter() - (self.t0 or time.perf_counter())
        logger.info("stage_end", extra={"stage": self.name, "ms": int(dt * 1000)})


@dataclass(frozen=True)
class IndexRunConfig:
    json_path: str
    sqlite_db_path: str
    collection_name: str
    mode: Literal["dense", "hybrid"] = "hybrid"
    recreate: bool = True
    batch_size: int = 32


def run_indexing(
    *,
    client: QdrantClient,
    embedder: BaseEmbedder,
    cfg: IndexRunConfig,
) -> dict:
    with StageTimer("load_articles"):
        articles = load_articles_from_json(cfg.json_path)
        logger.info("articles_loaded", extra={"count": len(articles)})

    with StageTimer("upsert_sqlite"):
        store = SQLiteArticleStore(cfg.sqlite_db_path)
        records = []
        for a in articles:
            full_id = make_full_article_id(a.page_path, a.article_id)
            records.append(
                ArticleRecord(
                    full_article_id=full_id,
                    page_path=a.page_path,
                    page_idx=a.page_idx,
                    article_id=a.article_id,
                    year=a.year,
                    cleaned_text=a.cleaned_text,
                    original_text=a.original_text,
                )
            )
        store.bulk_upsert(records)
        logger.info("sqlite_upsert_done", extra={"db": cfg.sqlite_db_path})

    with StageTimer("chunking"):
        chunks = make_chunks(articles)
        logger.info("chunks_created", extra={"count": len(chunks)})

    if cfg.recreate:
        with StageTimer("recreate_collection"):
            if cfg.mode == "dense":
                recreate_collection_dense(client, cfg.collection_name, embedder)
            else:
                recreate_collection_hybrid(client, cfg.collection_name, embedder)
            logger.info("collection_ready", extra={"collection": cfg.collection_name, "mode": cfg.mode})

    with StageTimer("qdrant_index"):
        index_chunks_in_qdrant(
            client=client,
            embedder=embedder,
            chunks=chunks,
            collection_name=cfg.collection_name,
            mode=cfg.mode,
            batch_size=cfg.batch_size,
        )
        logger.info("qdrant_index_done", extra={"collection": cfg.collection_name})

    return {
        "articles": len(articles),
        "chunks": len(chunks),
        "collection": cfg.collection_name,
        "mode": cfg.mode,
        "sqlite_db": cfg.sqlite_db_path,
    }
