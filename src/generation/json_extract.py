# src/generation/json_extract.py

from __future__ import annotations

import json
import re
from json import JSONDecodeError
from typing import Any


_JSON_FENCE_RE = re.compile(
    r"```(?:json|JSON)?\s*(.*?)\s*```",
    flags=re.S,
)


def _try_parse_first_json_object(s: str) -> dict[str, Any] | None:
    """
    Пытается распарсить первый JSON-объект из строки.

    Важно:
    json.JSONDecoder().raw_decode умеет распарсить первый JSON
    и вернуть позицию, на которой он закончился. Это спасает от
    ошибки "Extra data", когда LLM после JSON добавил текст или второй JSON.
    """
    decoder = json.JSONDecoder()

    for i, ch in enumerate(s):
        if ch != "{":
            continue

        try:
            obj, _end = decoder.raw_decode(s[i:])
        except JSONDecodeError:
            continue

        if isinstance(obj, dict):
            return obj

    return None


def extract_json_object(text: str) -> dict[str, Any]:
    """
    Устойчивое извлечение JSON из ответа LLM.

    Поддерживает:
    1. ```json {...}```
    2. ``` {...}```
    3. обычный текст + JSON
    4. JSON + хвостовой текст
    5. несколько JSON-блоков подряд — берём первый валидный объект
    """
    if not text or not text.strip():
        raise ValueError("Empty LLM output, cannot extract JSON object.")

    # 1. Сначала пробуем fenced code blocks.
    for match in _JSON_FENCE_RE.finditer(text):
        block = match.group(1).strip()
        obj = _try_parse_first_json_object(block)
        if obj is not None:
            return obj

    # 2. Потом пробуем весь текст.
    obj = _try_parse_first_json_object(text)
    if obj is not None:
        return obj

    head = text[:500].replace("\n", "\\n")
    raise ValueError(f"Cannot find valid JSON object in LLM output. Head: {head}")
