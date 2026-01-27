# fact_checking.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import json
import re

from src.llm.base import LLM


@dataclass
class Fact:
    fact_id: str
    statement: str
    evidence_quote: str
    article_id: str


@dataclass
class FactCard:
    article_id: str
    year: Optional[int]
    title_guess: Optional[str]
    facts: List[Fact]


def _safe_json_extract(text: str) -> Dict[str, Any]:
    m = re.search(r"\{.*\}", text, flags=re.S)
    if not m:
        raise ValueError(f"LLM output does not contain JSON object. Output head: {text[:200]}")
    return json.loads(m.group(0))


def build_fact_card_prompt(article_id: str, year: Optional[int], text: str, best_chunks: List[str]) -> str:
    return f"""Извлеки факты из газетной статьи (1930-е, русский язык).
Правила:
1) НЕ добавляй фактов/деталей, которых нет в тексте.
2) Каждый факт должен иметь доказательство: короткую цитату (<= 25 слов) из текста.
3) Факт должен быть атомарным (одно утверждение).
4) Если в тексте нет явной даты/места — не выдумывай.

Верни строго ОДИН JSON:

{{
  "article_id": "{article_id}",
  "year": {year if year is not None else "null"},
  "title_guess": null,
  "facts": [
    {{
      "fact_id": "A1-F1",
      "statement": "атомарный факт",
      "evidence_quote": "точная цитата из текста"
    }}
  ]
}}

ARTICLE_ID: {article_id}
YEAR: {year}

Подсказка по теме (самые релевантные фрагменты из retrieval):
{chr(10).join([f"- {c}" for c in best_chunks if c.strip()])}

Текст статьи:
{text}
"""


def build_fact_card(
    llm: LLM,
    article_id: str,
    year: Optional[int],
    full_text: str,
    best_chunks_texts: List[str],
    fact_id_prefix: str,
) -> FactCard:
    prompt = build_fact_card_prompt(article_id, year, full_text, best_chunks_texts)
    out = llm.generate(prompt, system="Ты — строгий факт-экстрактор. Никаких выдумок.")
    obj = _safe_json_extract(out)

    facts: List[Fact] = []
    for i, f in enumerate(obj.get("facts", []), start=1):
        fid = f.get("fact_id") or f"{fact_id_prefix}-F{i}"
        facts.append(Fact(
            fact_id=fid,
            statement=f["statement"].strip(),
            evidence_quote=f["evidence_quote"].strip(),
            article_id=article_id,
        ))

    return FactCard(
        article_id=article_id,
        year=obj.get("year", year),
        title_guess=obj.get("title_guess"),
        facts=facts,
    )


def build_fact_cards_for_retrieved(
    llm: LLM,
    article_store,
    retrieved_articles: List[Dict[str, Any]],
    max_articles: int = 7,
) -> List[FactCard]:
    fact_cards: List[FactCard] = []
    for idx, hit in enumerate(retrieved_articles[:max_articles], start=1):
        aid = hit["article_id"]
        rec = article_store.get(aid)
        if rec is None:
            continue

        best_chunks = [c["text"] for c in hit.get("best_chunks", []) if c.get("text")]
        card = build_fact_card(
            llm=llm,
            article_id=aid,
            year=rec.year,
            full_text=rec.cleaned_text,
            best_chunks_texts=best_chunks,
            fact_id_prefix=f"A{idx}",
        )
        fact_cards.append(card)
    return fact_cards


def build_fact_check_prompt(script: str, fact_cards: List[FactCard]) -> str:
    facts_flat = []
    for card in fact_cards:
        for f in card.facts:
            facts_flat.append({
                "fact_id": f.fact_id,
                "article_id": f.article_id,
                "statement": f.statement,
                "evidence_quote": f.evidence_quote,
            })

    return f"""Проверь сценарий подкаста на неподтверждённые утверждения.
Правила:
- Сценарий может быть стилизован, но факты должны опираться на список фактов ниже.
- Если утверждение НЕ следует из фактов — пометь как unsupported.
- Предложи исправление, которое ОПИРАЕТСЯ НА ФАКТЫ (или null).

Верни строго JSON:
{{
  "unsupported": [
    {{
      "claim": "...",
      "why_unsupported": "...",
      "suggested_fix": "..."  // или null
    }}
  ]
}}

ФАКТЫ:
{json.dumps(facts_flat, ensure_ascii=False)}

СЦЕНАРИЙ:
{script}
"""


def fact_check_script(llm: LLM, script: str, fact_cards: List[FactCard]) -> Dict[str, Any]:
    prompt = build_fact_check_prompt(script, fact_cards)
    out = llm.generate(prompt, system="Ты — строгий фактчекер. Минимум фантазии.")
    return _safe_json_extract(out)
