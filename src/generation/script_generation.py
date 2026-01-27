# script_generation.py
from __future__ import annotations

from typing import List, Dict, Any
import json
import re

from src.generation.fact_checking import FactCard
from src.llm.base import LLM



def _extract_json_object(text: str) -> Dict[str, Any]:
    """
    Более устойчиво, чем r"\{.*\}":
    - сначала ищем fenced ```json
    - иначе берём от первой '{' до последней '}'
    """
    fenced = re.search(r"```json\s*(\{.*?\})\s*```", text, flags=re.S | re.I)
    if fenced:
        return json.loads(fenced.group(1))

    l = text.find("{")
    r = text.rfind("}")
    if l == -1 or r == -1 or r <= l:
        raise ValueError(f"Cannot find JSON object in LLM output. Head: {text[:200]}")
    return json.loads(text[l:r+1])


def build_outline_prompt(query: str, fact_cards: List[FactCard]) -> str:
    compact = []
    for c in fact_cards:
        compact.append({
            "article_id": c.article_id,
            "year": c.year,
            "facts": [f.statement for f in c.facts],
        })

    return f"""Составь план выпуска подкаста в стиле ведущего 1930-х годов.
Тема запроса: {query}

Правила:
- Используй только факты из списка.
- План должен быть на 5–8 блоков с короткими названиями и 1–2 предложениями описания.
- Верни строго JSON:
{{
  "outline": [
    {{"title": "...", "goal": "...", "facts_used": ["A1-F1", "A2-F3"]}}
  ]
}}

ФАКТЫ (без цитат):
{json.dumps(compact, ensure_ascii=False)}
"""


def build_script_prompt(query: str, outline_obj: Dict[str, Any], fact_cards: List[FactCard]) -> str:
    facts_flat = []
    for c in fact_cards:
        for f in c.facts:
            facts_flat.append({"fact_id": f.fact_id, "statement": f.statement})

    style = """Стиль:
- диктор 1930-х, официально-газетная интонация, но живо
- связки и переходы допустимы, но НЕ добавляй новых фактов
- можно использовать “сегодня в нашей передаче…”, “как сообщают газеты…”, но без выдуманных деталей
"""

    return f"""{style}

Тема: {query}

Используй только факты ниже. Вставляй ссылки на факты в квадратных скобках прямо в тексте, например: [A1-F3].

OUTLINE (JSON):
{json.dumps(outline_obj, ensure_ascii=False)}

ФАКТЫ:
{json.dumps(facts_flat, ensure_ascii=False)}

Напиши цельный сценарий подкаста (примерно 4–7 минут текста)."""


def generate_outline(llm: LLM, query: str, fact_cards: List[FactCard]) -> Dict[str, Any]:
    prompt = build_outline_prompt(query, fact_cards)
    out = llm.generate(prompt, system="Ты — редактор, который строит план выпуска.")
    return _extract_json_object(out)


def generate_script(llm: LLM, query: str, outline_obj: Dict[str, Any], fact_cards: List[FactCard]) -> str:
    prompt = build_script_prompt(query, outline_obj, fact_cards)
    return llm.generate(prompt, system="Ты — диктор 1930-х. Но ты строго следуешь фактам.")
