# pipeline.py
from __future__ import annotations

from typing import Dict, Any, Optional
from time import perf_counter

from src.retrieval.retrieval import retrieve_articles
from src.generation.fact_checking import build_fact_cards_for_retrieved, fact_check_script
from src.generation.script_generation import generate_outline, generate_script


def run_pipeline(
    *,
    client,
    embedder,
    collection_name: str,
    reranker,
    article_store,
    llm,
    query: str,
    year: Optional[int] = None,
    max_articles_for_facts: int = 7,
) -> Dict[str, Any]:
    t0 = perf_counter()

    hits = retrieve_articles(
        client=client,
        embedder=embedder,
        query_text=query,
        collection_name=collection_name,
        mode="quality",
        retrieval="hybrid",
        year=year,
        reranker=reranker,
    )
    t1 = perf_counter()

    fact_cards = build_fact_cards_for_retrieved(
        llm=llm,
        article_store=article_store,
        retrieved_articles=hits,
        max_articles=max_articles_for_facts,
    )
    t2 = perf_counter()

    outline = generate_outline(llm, query, fact_cards)
    script = generate_script(llm, query, outline, fact_cards)
    t3 = perf_counter()

    report = fact_check_script(llm, script, fact_cards)
    t4 = perf_counter()

    return {
        "hits": hits,
        "fact_cards": fact_cards,
        "outline": outline,
        "script": script,
        "fact_check": report,
        "timings_sec": {
            "retrieval": t1 - t0,
            "fact_cards": t2 - t1,
            "generation": t3 - t2,
            "fact_check": t4 - t3,
            "total": t4 - t0,
        },
    }
