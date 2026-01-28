# src/generation/script_generation.py
from __future__ import annotations

from typing import List, Dict, Any, Tuple
import json
import re

from src.domain.contracts import FactCard
from src.llm.base import LLM


def _extract_json_object(text: str) -> Dict[str, Any]:
    fenced = re.search(r"```json\s*(\{.*?\})\s*```", text, flags=re.S | re.I)
    if fenced:
        return json.loads(fenced.group(1))

    l = text.find("{")
    r = text.rfind("}")
    if l == -1 or r == -1 or r <= l:
        raise ValueError(f"Cannot find JSON object in LLM output. Head: {text[:200]}")
    return json.loads(text[l:r+1])


def build_outline_and_script_prompt(query: str, fact_cards: List[FactCard]) -> str:
    # compact facts for outline
    compact = []
    facts_flat = []
    for c in fact_cards:
        compact.append({
            "article_id": c.article_id,
            "year": c.year,
            "facts": [f.statement for f in c.facts],
        })
        for f in c.facts:
            facts_flat.append({"fact_id": f.fact_id, "statement": f.statement})

    style = """Стиль:
- диктор 1930-х, официально-газетная интонация, но живо
- связки и переходы допустимы, но НЕ добавляй новых фактов
- можно использовать “сегодня в нашей передаче…”, “как сообщают газеты…”, но без выдуманных деталей
"""

    return f"""{style}

Тема: {query}

Требования:
1) Сначала составь OUTLINE (5–8 блоков).
2) Затем напиши SCRIPT (4–7 минут текста).
3) Используй только факты из FACTS.
4) Вставляй ссылки на факты в квадратных скобках прямо в тексте, например: [A1-F3].
5) Не выдумывай детали. Если фактов не хватает — формулируй осторожно и без новых утверждений.

Верни строго JSON:
{{
  "outline": [
    {{"title": "...", "goal": "...", "facts_used": ["A1-F1", "A2-F3"]}}
  ],
  "script": "..."
}}

FACTS_FOR_OUTLINE (сгруппировано):
{json.dumps(compact, ensure_ascii=False)}

FACTS (плоский список для ссылок):
{json.dumps(facts_flat, ensure_ascii=False)}
"""


def generate_outline_and_script(llm: LLM, query: str, fact_cards: List[FactCard]) -> Tuple[Dict[str, Any], str]:
    prompt = build_outline_and_script_prompt(query, fact_cards)
    out = llm.generate(
        prompt,
        system="STAGE:SCRIPT\nТы — редактор и диктор. Сначала план, затем сценарий. Строго по фактам. Только JSON.",
    )
    obj = _extract_json_object(out)

    outline = obj.get("outline") or []
    script = obj.get("script") or ""
    if not script.strip():
        raise ValueError("LLM returned empty script")

    return {"outline": outline}, str(script)


def build_outline_and_script_prompt_strict_refs(query: str, fact_cards: List[FactCard]) -> str:
    base = build_outline_and_script_prompt(query, fact_cards)

    known = []
    for c in fact_cards:
        for f in c.facts:
            known.append(f.fact_id)

    return (
        base
        + "\n\nСТРОГОЕ ПРАВИЛО:\n"
          "Используй ТОЛЬКО эти fact_id в ссылках вида [A#-F#]. Никаких других:\n"
        + ", ".join(known)
    )
