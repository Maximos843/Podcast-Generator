import re
from dataclasses import dataclass


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
