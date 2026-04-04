from __future__ import annotations

from typing import List, Dict, Any, Optional, Tuple, Union
import logging
import re

from src.types import Fact, FactCard, FactCheckReport, RetrievedArticleHit, PipelineRequest
from src.generation.json_extract import extract_json_object
from src.generation.prompts import (
    SYSTEM_FACTCHECK,
    SYSTEM_MAKE_FACTS,
    build_fact_cards_batch_user_prompt,
    build_fact_check_user_prompt,
)
from src.llm.base import LLM

logger = logging.getLogger("rag-service")

# Поднимаем лимиты умеренно, а не бесконечно
FACT_CONTEXT_MAX_CHUNKS = 6
FACT_CONTEXT_MAX_CHARS_PER_ARTICLE = 4200
FACT_BATCH_MAX_ARTICLES = 6

ARTICLE_WINDOW_CHARS = 1800
MAX_WINDOWS_PER_ARTICLE = 2
MIN_TOTAL_FACTS = 18

HitT = Union[Dict[str, Any], RetrievedArticleHit]


def _extract_hit_fields(hit: HitT) -> Tuple[str, Optional[int], List[str]]:
    if isinstance(hit, RetrievedArticleHit):
        aid = hit.article_id
        year = hit.year
        best_chunks = [c.text for c in (hit.best_chunks or []) if c.text]
        return aid, year, best_chunks

    aid = hit.get("article_id")
    year = hit.get("year")
    best_chunks = [c.get("text", "") for c in hit.get("best_chunks", []) if c.get("text")]
    return aid, year, best_chunks


def _normalize_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def _safe_find_subtext(full_text: str, sub_text: str) -> int:
    full = _normalize_ws(full_text)
    sub = _normalize_ws(sub_text)

    if not full or not sub:
        return -1

    idx = full.find(sub)
    if idx != -1:
        return idx

    anchor = sub[: min(len(sub), 180)]
    if len(anchor) >= 40:
        idx = full.find(anchor)
        if idx != -1:
            return idx

    return -1


def _cut_window(full_text: str, center_start: int, center_len: int, window_chars: int) -> str:
    full = _normalize_ws(full_text)
    if not full:
        return ""

    half = window_chars // 2
    center_mid = center_start + center_len // 2
    left = max(0, center_mid - half)
    right = min(len(full), center_mid + half)

    while left > 0 and full[left] != " ":
        left -= 1
    while right < len(full) and full[right - 1] != " ":
        right += 1
        if right >= len(full):
            right = len(full)
            break

    return full[left:right].strip()


def _dedupe_texts(texts: List[str]) -> List[str]:
    seen = set()
    out = []

    for t in texts:
        norm = _normalize_ws(t)
        if not norm or norm in seen:
            continue
        seen.add(norm)
        out.append(norm)

    return out


def _build_expanded_context_for_hit(article_store, hit: HitT) -> str:
    aid, _, best_chunks = _extract_hit_fields(hit)
    best_chunks = [c for c in best_chunks if c and c.strip()][:FACT_CONTEXT_MAX_CHUNKS]

    fallback_ctx = ("\n\n".join(best_chunks))[:FACT_CONTEXT_MAX_CHARS_PER_ARTICLE]

    if not aid or article_store is None:
        return fallback_ctx

    rec = article_store.get(aid)
    if rec is None or not rec.cleaned_text:
        return fallback_ctx

    full_text = rec.cleaned_text
    windows = []

    for chunk in best_chunks:
        idx = _safe_find_subtext(full_text, chunk)
        if idx == -1:
            windows.append(_normalize_ws(chunk))
            continue

        window = _cut_window(
            full_text=full_text,
            center_start=idx,
            center_len=len(_normalize_ws(chunk)),
            window_chars=ARTICLE_WINDOW_CHARS,
        )
        if window:
            windows.append(window)

        if len(windows) >= MAX_WINDOWS_PER_ARTICLE:
            break

    windows = _dedupe_texts(windows)
    if not windows:
        return fallback_ctx

    parts = []
    for i, window in enumerate(windows, start=1):
        if i - 1 < len(best_chunks):
            parts.append(f"[ANCHOR_CHUNK_{i}]\n{_normalize_ws(best_chunks[i - 1])}")
        parts.append(f"[ARTICLE_WINDOW_{i}]\n{window}")

    merged = "\n\n".join(parts).strip()
    return merged[:FACT_CONTEXT_MAX_CHARS_PER_ARTICLE]


def build_fact_cards_batch_prompt(hits: List[HitT], request: PipelineRequest, article_store) -> str:
    blocks = []

    for i, h in enumerate(hits, start=1):
        aid, year, _ = _extract_hit_fields(h)
        ctx = _build_expanded_context_for_hit(article_store, h)

        if not aid or not ctx.strip():
            continue

        blocks.append(
            {
                "slot": f"A{i}",
                "article_id": aid,
                "year": year,
                "context": ctx,
            }
        )

    return build_fact_cards_batch_user_prompt(blocks, request.query)


def build_fact_cards_batch_prompt_fallback(hits: List[HitT], request: PipelineRequest, article_store) -> str:
    blocks = []

    for i, h in enumerate(hits, start=1):
        aid, year, _ = _extract_hit_fields(h)
        ctx = _build_expanded_context_for_hit(article_store, h)

        if not aid or not ctx.strip():
            continue

        blocks.append(
            {
                "slot": f"A{i}",
                "article_id": aid,
                "year": year,
                "context": ctx,
            }
        )

    return build_fact_cards_batch_user_prompt_fallback(blocks, request.query)


def _parse_fact_cards(obj: dict) -> List[FactCard]:
    cards_in = obj.get("cards", [])
    fact_cards: List[FactCard] = []

    for c in cards_in:
        aid = c.get("article_id")
        if not aid:
            continue

        facts: List[Fact] = []
        seen = set()

        for f in c.get("facts", []) or []:
            st = str(f.get("statement", "")).strip()
            ev = str(f.get("evidence_quote", "")).strip()
            fid = str(f.get("fact_id", "")).strip()
            if not (st and ev and fid):
                continue

            key = st.lower()
            if key in seen:
                continue
            seen.add(key)

            facts.append(
                Fact(
                    fact_id=fid,
                    statement=st,
                    evidence_quote=ev,
                    article_id=aid,
                )
            )

        if facts:
            fact_cards.append(
                FactCard(
                    article_id=aid,
                    year=c.get("year"),
                    title_guess=c.get("title_guess"),
                    facts=facts,
                )
            )

    return fact_cards


def build_fact_cards_for_retrieved(
    llm: LLM,
    article_store,
    request: PipelineRequest,
    retrieved_articles: List[HitT],
    max_articles: int = 7,
) -> List[FactCard]:
    hits = retrieved_articles[: min(max_articles, FACT_BATCH_MAX_ARTICLES)]

    prompt = build_fact_cards_batch_prompt(hits, request, article_store)
    out = llm.generate(prompt, system=SYSTEM_MAKE_FACTS)
    obj = extract_json_object(out)
    fact_cards = _parse_fact_cards(obj)

    total_facts = sum(len(c.facts) for c in fact_cards)
    if total_facts >= MIN_TOTAL_FACTS:
        return fact_cards

    # fallback: пытаемся дожать больше фактов
    fallback_prompt = build_fact_cards_batch_prompt(hits, request, article_store)
    fallback_out = llm.generate(fallback_prompt + '\n\nВ прошлый раз было слишком мало фактов, нужно больше релеватных фактов для статей.', system=SYSTEM_MAKE_FACTS)
    fallback_obj = extract_json_object(fallback_out)
    fallback_cards = _parse_fact_cards(fallback_obj)

    fallback_total_facts = sum(len(c.facts) for c in fallback_cards)
    if fallback_total_facts > total_facts:
        return fallback_cards

    return fact_cards


def build_fact_check_prompt(script: str, fact_cards: List[FactCard], request: str) -> str:
    facts_flat = []
    for card in fact_cards:
        for f in card.facts:
            facts_flat.append(
                {
                    "fact_id": f.fact_id,
                    "article_id": f.article_id,
                    "statement": f.statement,
                    "evidence_quote": f.evidence_quote,
                }
            )

    return build_fact_check_user_prompt(facts_flat, script, request)


def fact_check_script(llm: LLM, script: str, fact_cards: List[FactCard], request: str) -> FactCheckReport:
    prompt = build_fact_check_prompt(script, fact_cards, request)
    out = llm.generate(prompt, system=SYSTEM_FACTCHECK)
    obj = extract_json_object(out)
    return FactCheckReport.model_validate(obj)