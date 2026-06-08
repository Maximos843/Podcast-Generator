# tests/bench_retrieval.py

import asyncio
import time
import random
import traceback
from collections import Counter

from src.config import AppConfig
from src.service.wiring import build_deps
from src.types import PipelineRequest
from src.retrieval.qdrant_retriever import QdrantRetriever


QUERIES = [
    "Индустриализация 1930-х годов",
    "Жизнь рабочих в советских газетах",
    "Сельское хозяйство в СССР",
    "Культурная жизнь 1930-х",
    "Политическая повестка советской прессы",
    "Образование и грамотность в СССР",
    "Ударники труда и заводская жизнь",
    "Международная повестка в советской прессе",
    "Быт людей по материалам газет",
    "Роль партии в советской прессе",
]


def percentile(sorted_values: list[float], p: float) -> float:
    if not sorted_values:
        raise ValueError("empty values")

    idx = int((len(sorted_values) - 1) * p)
    return sorted_values[idx]


async def bench_retrieval(n_requests: int = 100, concurrency: int = 10):
    cfg = AppConfig.from_env()
    deps = build_deps(cfg)

    retriever = QdrantRetriever(
        client=deps.client,
        embedder=deps.embedder,
        collection_name=deps.collection_name,
        reranker=deps.reranker,
    )

    print(f"Тест retrieval: {n_requests} запросов, concurrency={concurrency}", flush=True)
    print(f"Коллекция: {deps.collection_name}", flush=True)

    collection_info = deps.client.get_collection(deps.collection_name)
    points_count = collection_info.points_count
    print(f"Точек в коллекции: {points_count}", flush=True)

    error_types = Counter()
    error_examples: list[str] = []

    async def single_request(i: int, query: str):
        req = PipelineRequest(
            query=query,
            mode="fast",
            retrieval="hybrid",
            max_articles_for_facts=5,
            include_debug=False,
        )

        t0 = time.perf_counter()

        try:
            hits = await asyncio.to_thread(retriever.retrieve, req)
            latency_ms = (time.perf_counter() - t0) * 1000
            return {
                "ok": True,
                "latency_ms": latency_ms,
                "hits": len(hits),
                "error": None,
            }

        except Exception as e:
            error_types[type(e).__name__] += 1

            if len(error_examples) < 3:
                error_examples.append(
                    f"[request={i}] {type(e).__name__}: {e}\n"
                    + traceback.format_exc(limit=5)
                )

            return {
                "ok": False,
                "latency_ms": None,
                "hits": 0,
                "error": str(e),
            }

    semaphore = asyncio.Semaphore(concurrency)

    async def limited_request(i: int):
        async with semaphore:
            query = random.choice(QUERIES)
            return await single_request(i, query)

    wall_t0 = time.perf_counter()

    tasks = [limited_request(i) for i in range(n_requests)]
    results = await asyncio.gather(*tasks)

    wall_seconds = time.perf_counter() - wall_t0

    ok_results = [r for r in results if r["ok"]]
    latencies = sorted(float(r["latency_ms"]) for r in ok_results)
    hit_counts = [int(r["hits"]) for r in ok_results]

    print("\nРезультаты:", flush=True)
    print(f"  Всего запросов: {n_requests}", flush=True)
    print(f"  Успешных: {len(ok_results)}", flush=True)
    print(f"  Ошибок: {n_requests - len(ok_results)}", flush=True)
    print(f"  Wall time: {wall_seconds:.2f} sec", flush=True)
    print(f"  RPS: {len(ok_results) / wall_seconds:.2f}", flush=True)

    if error_types:
        print("\nОшибки:", flush=True)
        for name, cnt in error_types.most_common():
            print(f"  {name}: {cnt}", flush=True)

        print("\nПримеры ошибок:", flush=True)
        for err in error_examples:
            print(err, flush=True)

    if not latencies:
        print("\nНет успешных запросов, latency считать нельзя.", flush=True)
        return

    print("\nLatency:", flush=True)
    print(f"  Avg: {sum(latencies) / len(latencies):.2f} ms", flush=True)
    print(f"  P50: {percentile(latencies, 0.50):.2f} ms", flush=True)
    print(f"  P95: {percentile(latencies, 0.95):.2f} ms", flush=True)
    print(f"  P99: {percentile(latencies, 0.99):.2f} ms", flush=True)
    print(f"  Min: {min(latencies):.2f} ms", flush=True)
    print(f"  Max: {max(latencies):.2f} ms", flush=True)

    print("\nHits:", flush=True)
    print(f"  Avg hits: {sum(hit_counts) / len(hit_counts):.2f}", flush=True)
    print(f"  Min hits: {min(hit_counts)}", flush=True)
    print(f"  Max hits: {max(hit_counts)}", flush=True)


if __name__ == "__main__":
    asyncio.run(bench_retrieval(n_requests=500, concurrency=10))