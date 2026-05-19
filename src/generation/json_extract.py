# src/generation/json_extract.py
from __future__ import annotations

import json
import re
from typing import Any, Dict


def extract_json_object(text: str) -> Dict[str, Any]:
    """
    Устойчивое извлечение JSON:
    1) fenced ```json {...}```
    2) иначе от первой '{' до последней '}'
    """
    fenced = re.search(r"```json\s*(\{.*?\})\s*```", text, flags=re.S | re.I)
    if fenced:
        return json.loads(fenced.group(1))

    l = text.find("{")
    r = text.rfind("}")
    if l == -1 or r == -1 or r <= l:
        raise ValueError(f"Cannot find JSON object in LLM output. Head: {text[:200]}")
    return json.loads(text[l : r + 1])
