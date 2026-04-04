# preprocessing.py
from __future__ import annotations

import json
import logging
import re
from typing import Optional, List

from src.config import YEAR_RE
from src.domain.types import Article

logger = logging.getLogger(__name__)


def clean_article_text(text: str, dedupe_adjacent: bool = True) -> str:
    if not text:
        return ""

    lines = [ln.strip() for ln in text.splitlines()]
    lines = [ln for ln in lines if ln]

    if dedupe_adjacent:
        deduped = []
        prev = None
        for ln in lines:
            if ln != prev:
                deduped.append(ln)
            prev = ln
        lines = deduped

    out = " ".join(lines)
    out = re.sub(r"\s+", " ", out).strip()
    return out


def chunk_text_by_sentences(
    text: str,
    max_chars: int = 1200,
    overlap_chars: int = 300,
) -> List[str]:
    if not text or not text.strip():
        return []

    sentences = re.split(r"(?<=[\.\?!»])\s+", text)
    chunks: List[List[str]] = []
    current: List[str] = []

    def current_len(parts: List[str]) -> int:
        return sum(len(s) + 1 for s in parts)

    i = 0
    while i < len(sentences):
        sent = sentences[i].strip()
        if not sent:
            i += 1
            continue

        if current_len(current) + len(sent) + 1 <= max_chars:
            current.append(sent)
            i += 1
            continue

        if current:
            prev_current = current
            chunks.append(prev_current)

            overlap: List[str] = []
            total = 0
            for s in reversed(prev_current):
                if total + len(s) + 1 <= overlap_chars:
                    overlap.append(s)
                    total += len(s) + 1
                else:
                    break
            current = list(reversed(overlap))

            if current and (current_len(current) + len(sent) + 1 > max_chars):
                logger.debug(
                    "Overlap blocks next sentence; dropping overlap. sent_len=%s overlap_len=%s",
                    len(sent), current_len(current),
                )
                current = []
            continue

        chunks.append([sent])
        i += 1

    if current:
        chunks.append(current)

    return [" ".join(ch).strip() for ch in chunks if ch and " ".join(ch).strip()]


def extract_year_from_path(path: str) -> Optional[int]:
    m = YEAR_RE.search(path)
    return int(m.group(1)) if m else None


def load_articles_from_json(json_path: str) -> List[Article]:
    with open(json_path, "r", encoding="utf-8") as f:
        pages = json.load(f)

    articles: List[Article] = []
    page_paths = list(pages.keys())

    for page_idx, page_path in enumerate(page_paths):
        year = extract_year_from_path(page_path)
        article_map: dict[str, str] = pages[page_path]

        for article_id, raw_text in article_map.items():
            cleaned = clean_article_text(raw_text, dedupe_adjacent=True)
            if not cleaned:
                continue

            articles.append(Article(
                page_path=page_path,
                page_idx=page_idx,
                article_id=article_id,
                year=year,
                original_text=raw_text,
                cleaned_text=cleaned,
            ))

    return articles
