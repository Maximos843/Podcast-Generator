from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class Article:
    page_path: str
    page_idx: int
    article_id: str
    year: Optional[int]
    original_text: str
    cleaned_text: str


@dataclass(frozen=True)
class Chunk:
    article: Article
    chunk_id: int
    text: str