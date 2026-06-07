# src/service/wiring.py
from __future__ import annotations

from typing import Any

from qdrant_client import QdrantClient

from src.config import AppConfig
from src.pipeline.service import PipelineDeps
from src.storage.article_store import SQLiteArticleStore, InMemoryArticleStore
from src.storage.preprocessing import extract_year_from_path, clean_article_text
from src.llm.yandex_llm import YandexLLMConfig, YandexGPT5Client
from src.llm.reliable import ReliableLLM, LLMJsonPolicy
from src.llm.routed import RoutedLLM, LLMTaskRoutingConfig
from src.retrieval.model import SentenceTransformerEmbedder
from src.retrieval.model import SentenceTransformerEmbedder, BGEReranker


def build_qdrant(cfg: AppConfig) -> QdrantClient:
    #if cfg.qdrant_api_key:
    #    return QdrantClient(url=cfg.qdrant_url, api_key=cfg.qdrant_api_key)
    return QdrantClient(url=cfg.qdrant_url)


def build_article_store(cfg: AppConfig) -> Any:
    if cfg.env != "dev" or not cfg.json_articles_path:
        print(cfg.sqlite_db_path)
        return SQLiteArticleStore(cfg.sqlite_db_path)

    return InMemoryArticleStore.from_json(
        cfg.json_articles_path,
        clean=True,
        extract_year_fn=extract_year_from_path,
        clean_text_fn=clean_article_text,
        dedupe_adjacent=True,
    )


def _build_single_llm(
    *,
    api_key: str,
    folder_id: str,
    model_uri: str,
    base_url: str,
    temperature: float,
) -> Any:
    base = YandexGPT5Client(
        YandexLLMConfig(
            api_key=api_key,
            folder_id=folder_id,
            model_uri=model_uri,
            base_url=base_url,
            temperature=temperature,
        )
    )
    return ReliableLLM(base, policy=LLMJsonPolicy(max_attempts=2))  # type: ignore


def build_llm(cfg: AppConfig) -> Any:
    if not cfg.yandex_api_key or not cfg.yandex_folder_id:
        raise RuntimeError("Missing YANDEX_API_KEY or YANDEX_FOLDER_ID for weak LLM.")

    weak_llm = _build_single_llm(
        api_key=cfg.yandex_api_key,
        folder_id=cfg.yandex_folder_id,
        model_uri=f"gpt://{cfg.yandex_folder_id}/yandexgpt-5-lite",
        base_url=cfg.yandex_url,
        temperature=cfg.llm_default_temperature,
    )

    strong_llm = _build_single_llm(
        api_key=cfg.yandex_api_key,
        folder_id=cfg.yandex_folder_id,
        model_uri=f"gpt://{cfg.yandex_folder_id}/yandexgpt-5-pro",
        base_url=cfg.yandex_url,
        temperature=cfg.script_temperature,
    )

    routed = RoutedLLM(
        weak_llm=weak_llm,
        strong_llm=strong_llm,
        routing=LLMTaskRoutingConfig(
            facts_temperature=cfg.facts_temperature,
            factcheck_temperature=cfg.factcheck_temperature,
            strict_refs_temperature=cfg.strict_refs_temperature,
            script_temperature=cfg.script_temperature,
            repair_temperature=cfg.repair_temperature,
        ),
    )
    return routed


def build_embedder(cfg: AppConfig) -> Any:
    return SentenceTransformerEmbedder(
        model_name=cfg.embedder_model_name,
        device=cfg.embedder_device,
    )


def build_reranker(cfg: AppConfig) -> Any | None:
    return BGEReranker(
        model_name="BAAI/bge-reranker-v2-m3",
        device="cpu",
    )


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