# src/retrieval/payload_schema.py
from __future__ import annotations

FULL_ARTICLE_ID = "full_article_id"
CHUNK_ID = "chunk_id"
TEXT = "text"
YEAR = "year"

REQUIRED_KEYS = (FULL_ARTICLE_ID, TEXT)
OPTIONAL_KEYS = (CHUNK_ID, YEAR)
