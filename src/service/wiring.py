from __future__ import annotations

from typing import Any, Optional

from qdrant_client import QdrantClient

from src.config import AppConfig
from src.pipeline.service import PipelineDeps
from src.storage.article_store import SQLiteArticleStore, InMemoryArticleStore
from src.storage.preprocessing import extract_year_from_path, clean_article_text
from src.llm.yandex_llm import YandexLLMConfig, YandexGPT5Client
from src.llm.reliable import ReliableLLM, LLMJsonPolicy
from src.retrieval.model import SentenceTransformerEmbedder


def build_qdrant(cfg: AppConfig) -> QdrantClient:
    if cfg.qdrant_api_key:
        return QdrantClient(url=cfg.qdrant_url, api_key=cfg.qdrant_api_key)
    return QdrantClient(url=cfg.qdrant_url)


def build_article_store(cfg: AppConfig) -> Any:
    # Прод: всегда sqlite
    if cfg.env != "dev" or not cfg.json_articles_path:
        return SQLiteArticleStore(cfg.sqlite_db_path)

    # Dev: можно in-memory из JSON
    return InMemoryArticleStore.from_json(
        cfg.json_articles_path,
        clean=True,
        extract_year_fn=extract_year_from_path,
        clean_text_fn=clean_article_text,
        dedupe_adjacent=True,
    )


def build_llm(cfg: AppConfig) -> Any:
    if not cfg.yandex_api_key or not cfg.yandex_folder_id:
        raise RuntimeError("Missing YANDEX_API_KEY or YANDEX_FOLDER_ID for LLM.")

    base = YandexGPT5Client(
        YandexLLMConfig(
            api_key=cfg.yandex_api_key,
            folder_id=cfg.yandex_folder_id,
            model_uri=cfg.yandex_model_uri,
        )
    )
    # Оборачиваем для JSON-надёжности
    return ReliableLLM(base, policy=LLMJsonPolicy(max_attempts=2))


def build_embedder(cfg: AppConfig) -> Any:
    return SentenceTransformerEmbedder(
        model_name=cfg.embedder_model_name,
        device=cfg.embedder_device,
    )


def build_reranker(cfg: AppConfig) -> Optional[Any]:
    # Пока None: включим позже конфигом и конкретной реализацией
    return None


def build_deps(cfg: AppConfig) -> PipelineDeps:
    client = build_qdrant(cfg)
    embedder = build_embedder(cfg)
    reranker = build_reranker(cfg)
    article_store = build_article_store(cfg)
    llm = build_llm(cfg)

    return PipelineDeps(
        client=client,
        embedder=embedder,
        collection_name=cfg.qdrant_collection,
        reranker=reranker,
        article_store=article_store,
        llm=llm,
    )
