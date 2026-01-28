# src/generation/fact_checking.py
from __future__ import annotations

from typing import List, Dict, Any, Optional, Tuple, Union
import json
import logging

from src.domain.contracts import Fact, FactCard, FactCheckReport, RetrievedArticleHit
from src.generation.json_extract import extract_json_object
from src.llm.base import LLM

logger = logging.getLogger("rag-service")

# --- Настройки батча ---
FACT_CONTEXT_MAX_CHUNKS = 4
FACT_CONTEXT_MAX_CHARS_PER_ARTICLE = 2200   # ↓ меньше, чтобы батч не раздувался
FACT_BATCH_MAX_ARTICLES = 4                 # сколько статей пакуем в батч (обычно = max_articles)

HitT = Union[Dict[str, Any], RetrievedArticleHit]


def _extract_hit_fields(hit: HitT) -> Tuple[str, Optional[int], List[str]]:
    """
    Возвращает: (article_id, year, best_chunks_texts)
    Поддерживает dict и RetrievedArticleHit.
    """
    if isinstance(hit, RetrievedArticleHit):
        aid = hit.article_id
        year = hit.year
        best_chunks = [c.text for c in (hit.best_chunks or []) if c.text]
        return aid, year, best_chunks

    aid = hit.get("article_id")
    year = hit.get("year")
    best_chunks = [c.get("text", "") for c in hit.get("best_chunks", []) if c.get("text")]
    return aid, year, best_chunks


def _normalize_context(best_chunks_texts: List[str]) -> str:
    chunks = [c.strip() for c in best_chunks_texts if c and c.strip()][:FACT_CONTEXT_MAX_CHUNKS]
    return ("\n\n".join(chunks))[:FACT_CONTEXT_MAX_CHARS_PER_ARTICLE]


def build_fact_cards_batch_prompt(hits: List[HitT]) -> str:
    """
    В prompt чётко разделяем статьи, запрещаем смешивать источники.
    """
    blocks = []
    for i, h in enumerate(hits, start=1):
        aid, year, best_chunks = _extract_hit_fields(h)
        ctx = _normalize_context(best_chunks)
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

    return f"""Ты извлекаешь факты из фрагментов газетных статей (1930-е, русский язык).
ВАЖНО:
- НЕЛЬЗЯ смешивать источники: факты внутри карточки должны быть ТОЛЬКО из её контекста.
- НЕЛЬЗЯ добавлять детали, которых нет в контексте.
- Каждый факт обязан иметь короткую дословную цитату (<= 25 слов) из соответствующего контекста.
- fact_id должен быть вида A#-F# (например A1-F1, A1-F2...) где A# соответствует номеру блока.

Верни строго JSON (без текста вокруг):
{{
  "cards": [
    {{
      "slot": "A1",
      "article_id": "...",
      "year": 1934,
      "title_guess": null,
      "facts": [
        {{
          "fact_id": "A1-F1",
          "statement": "атомарный факт",
          "evidence_quote": "короткая цитата из A1"
        }}
      ]
    }}
  ]
}}

БЛОКИ (каждый блок — отдельная статья, используй факты только из её контекста):
{json.dumps(blocks, ensure_ascii=False)}
"""


def build_fact_cards_for_retrieved(
    llm: LLM,
    article_store,  # оставляем для совместимости; не используем
    retrieved_articles: List[HitT],
    max_articles: int = 7,
) -> List[FactCard]:
    # для батча обычно хватает 3-4 статей, но берём min(max_articles, FACT_BATCH_MAX_ARTICLES)
    hits = retrieved_articles[: min(max_articles, FACT_BATCH_MAX_ARTICLES)]
    prompt = build_fact_cards_batch_prompt(hits)

    out = llm.generate(
        prompt,
        system="STAGE:FACTS\nТы — строгий факт-экстрактор. Никаких выдумок. Только JSON.",
    )
    obj = extract_json_object(out)
    cards_in = obj.get("cards", [])

    fact_cards: List[FactCard] = []
    for c in cards_in:
        aid = c.get("article_id")
        if not aid:
            continue

        facts: List[Fact] = []
        for f in c.get("facts", []) or []:
            # минимальная защита от пустых полей
            st = str(f.get("statement", "")).strip()
            ev = str(f.get("evidence_quote", "")).strip()
            fid = str(f.get("fact_id", "")).strip()
            if not (st and ev and fid):
                continue
            facts.append(Fact(
                fact_id=fid,
                statement=st,
                evidence_quote=ev,
                article_id=aid,
            ))

        fact_cards.append(FactCard(
            article_id=aid,
            year=c.get("year"),
            title_guess=c.get("title_guess"),
            facts=facts,
        ))

    return fact_cards


# --- fact check (оставляем как было, только stage полезно) ---
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


def fact_check_script(llm: LLM, script: str, fact_cards: List[FactCard]) -> FactCheckReport:
    prompt = build_fact_check_prompt(script, fact_cards)
    out = llm.generate(prompt, system="STAGE:FACTCHECK\nТы — строгий фактчекер. Минимум фантазии.")
    obj = extract_json_object(out)
    return FactCheckReport.model_validate(obj)
