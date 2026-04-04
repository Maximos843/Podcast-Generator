# src/generation/script_generation.py
from __future__ import annotations

from typing import List, Dict, Any, Tuple

from src.types import FactCard
from src.generation.json_extract import extract_json_object
from src.generation.prompts import (
    SYSTEM_SCRIPT_GENERATION,
    build_outline_and_script_user_prompt,
    build_outline_and_script_user_prompt_strict_refs,
)
from src.llm.base import LLM


def build_outline_and_script_prompt(query: str, fact_cards: List[FactCard]) -> str:
    compact = []
    facts_flat = []

    for c in fact_cards:
        compact.append(
            {
                "article_id": c.article_id,
                "year": c.year,
                "facts": [f.statement for f in c.facts],
            }
        )
        for f in c.facts:
            facts_flat.append({"fact_id": f.fact_id, "statement": f.statement})

    return build_outline_and_script_user_prompt(
        query=query,
        compact=compact,
        facts_flat=facts_flat,
    )


def generate_outline_and_script(llm: LLM, query: str, fact_cards: List[FactCard]) -> Tuple[Dict[str, Any], str]:
    prompt = build_outline_and_script_prompt(query, fact_cards)
    out = llm.generate(prompt, system=SYSTEM_SCRIPT_GENERATION)
    print(out)
    obj = extract_json_object(out)

    outline = obj.get("outline") or []
    script = obj.get("script") or ""
    if not script.strip():
        raise ValueError("LLM returned empty script")

    return {"outline": outline}, str(script)


def build_outline_and_script_prompt_strict_refs(query: str, fact_cards: List[FactCard]) -> str:
    compact = []
    facts_flat = []
    known = []

    for c in fact_cards:
        compact.append(
            {
                "article_id": c.article_id,
                "year": c.year,
                "facts": [f.statement for f in c.facts],
            }
        )
        for f in c.facts:
            facts_flat.append({"fact_id": f.fact_id, "statement": f.statement})
            known.append(f.fact_id)

    return build_outline_and_script_user_prompt_strict_refs(
        query=query,
        compact=compact,
        facts_flat=facts_flat,
        known_fact_ids=known,
    )