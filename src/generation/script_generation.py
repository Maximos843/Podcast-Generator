from typing import Any
import re

from src.types import FactCard
from src.generation.json_extract import extract_json_object
from src.generation.prompts import (
    SYSTEM_SCRIPT_GENERATION,
    build_outline_and_script_user_prompt,
    build_outline_and_script_user_prompt_strict_refs,
)
from src.llm.base import LLM


FACT_REF_RE = re.compile(r"\[A\d+-F\d+\]")


def build_outline_and_script_prompt(query: str, fact_cards: list[FactCard]) -> str:
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


def _script_needs_retry(script: str) -> bool:
    words = len(script.split())
    refs = len(FACT_REF_RE.findall(script))
    return words < 1500 or refs < 10


def generate_outline_and_script(llm: LLM, query: str, fact_cards: list[FactCard]) -> tuple[dict[str, Any], str]:
    prompt = build_outline_and_script_prompt(query, fact_cards)
    out = llm.generate(prompt, system=SYSTEM_SCRIPT_GENERATION, task="script")
    obj = extract_json_object(out)

    outline = obj.get("outline") or []
    script = str(obj.get("script") or "").strip()
    if not script:
        raise ValueError("LLM returned empty script")

    if _script_needs_retry(script):
        retry_prompt = prompt + "\n\nДополнительное требование: сценарий получился слишком кратким. Сделай его более насыщенным фактами и развернутым, но без выдумки."
        retry_out = out = llm.generate(retry_prompt, system=SYSTEM_SCRIPT_GENERATION, task="script")
        retry_obj = extract_json_object(retry_out)

        retry_outline = retry_obj.get("outline") or outline
        retry_script = str(retry_obj.get("script") or "").strip()

        if retry_script and not _script_needs_retry(retry_script):
            return {"outline": retry_outline}, retry_script

    return {"outline": outline}, script


def build_outline_and_script_prompt_strict_refs(query: str, fact_cards: list[FactCard]) -> str:
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
