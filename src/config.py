import re
from dataclasses import dataclass
from typing import Optional


@dataclass
class Article:
    page_path: str
    page_idx: int
    article_id: str          # "0", "1", ...
    year: Optional[int]
    original_text: str       # как в JSON
    cleaned_text: str        # после предобработки


@dataclass
class Chunk:
    article: Article
    chunk_id: int
    text: str                # текст чанка (очищенный)


@dataclass
class QDrantConfig:
    COLLECTION_NAME: str = "newspapers_chunks"
    EMBEDDING_DIM: int = 768
    QDRANT_DATA_PATH: str = "qdrant_data"


@dataclass
class EvalQuery:
    query_text: str
    relevant_articles: set[str]
    source: str # "golden_set", "synthetic_simple", "synthetic_hard", "qa", ...
    relevance_grades: Optional[dict[str, int]] = None


@dataclass
class EvalMetrics:
    mrr: float
    hit_at_k: float
    recall_at_k: float
    precision_at_k: float


YEAR_RE = re.compile(r'_(\d{4})_')