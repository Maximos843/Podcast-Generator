# src/config.py
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Optional
from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class QDrantConfig:
    # --- hybrid (named vectors) ---
    DENSE_VECTOR_NAME: str = "dense"
    BM25_VECTOR_NAME: str = "bm25"
    BM25_MODEL: str = "Qdrant/bm25"

    # --- retrieve defaults ---
    PREFETCH_K: int = 150
    CANDIDATE_POOL_CHUNKS_FAST: int = 50
    CANDIDATE_POOL_CHUNKS_QUALITY: int = 75
    TOP_K_ARTICLES_FAST: int = 7
    TOP_K_ARTICLES_QUALITY: int = 7
    PER_ARTICLE_TOP_CHUNKS: int = 3


YEAR_RE = re.compile(r"_(\d{4})_")


@dataclass(frozen=True)
class AppConfig:
    # --- runtime ---
    env: str = "dev"
    host: str = "0.0.0.0"
    port: int = 8000

    # --- qdrant ---
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: Optional[str] = None
    qdrant_collection: str = "podcast_chunks"

    # --- article store ---
    sqlite_db_path: str = "articles.sqlite"
    json_articles_path: Optional[str] = None  # если хочешь in-memory из JSON

    # --- yandex llm ---
    yandex_api_key: Optional[str] = None
    yandex_folder_id: Optional[str] = None
    yandex_model_uri: Optional[str] = None



    yandex_url: str = "https://llm.api.cloud.yandex.net/v1"
    llm_default_temperature: float = 0.1

    facts_temperature: float = 0.1
    factcheck_temperature: float = 0.0
    strict_refs_temperature: float = 0.0
    script_temperature: float = 0.5
    repair_temperature: float = 0.2

    # --- embedder ---
    embedder_model_name: str = "intfloat/multilingual-e5-base"
    embedder_device: str = "cpu"

    @classmethod
    def from_env(cls) -> "AppConfig":
        def geti(name: str, default: int) -> int:
            v = os.getenv(name)
            return default if v is None else int(v)

        return cls(
            env=os.getenv("ENV", "dev"),
            host=os.getenv("HOST", "0.0.0.0"),
            port=geti("PORT", 8000),

            qdrant_url=os.getenv("QDRANT_URL", "http://localhost:6333"),
            qdrant_api_key=os.getenv("QDRANT_API_KEY"),

            qdrant_collection=os.getenv("QDRANT_COLLECTION", "podcast_chunks"),

            sqlite_db_path=os.getenv("SQLITE_DB_PATH", "data/articles.sqlite"),
            json_articles_path=os.getenv("JSON_ARTICLES_PATH"),

            yandex_api_key=os.getenv("YANDEX_API_KEY", '***'),
            yandex_folder_id=os.getenv("YANDEX_FOLDER_ID", '***'),
            yandex_model_uri=os.getenv("YANDEX_MODEL_URI", ''),

            embedder_model_name=os.getenv("EMBEDDER_MODEL_NAME", "intfloat/multilingual-e5-base"),
            embedder_device=os.getenv("EMBEDDER_DEVICE", "cpu"),
        )
