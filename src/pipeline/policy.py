# src/pipeline/policy.py
from __future__ import annotations

from src.types import PipelineRequest


def apply_policy(req: PipelineRequest) -> PipelineRequest:
    """
    Минимальная политика без усложнений:
    - fast: меньше статей для fact cards, отключаем фактчек (пайплайн это уже сделает),
            и предпочитаем dense, если хочешь (можно убрать).
    - quality: как есть.
    """
    if req.mode == "fast":
        # меньше статей => быстрее и дешевле LLM
        max_articles = min(req.max_articles_for_facts, 4)
        # можешь оставить retrieval как есть; я предлагаю dense как дефолт для скорости
        retrieval = "dense" if req.retrieval == "hybrid" else req.retrieval
        return req.model_copy(update={"max_articles_for_facts": max_articles, "retrieval": retrieval})

    return req
