import json
from config import EvalQuery, EvalMetrics
from embedder import Embedder
from typing import Iterable
from collections import defaultdict
from qdrant_client.models import Filter, FieldCondition, Range
from qdrant_client import QdrantClient


def load_eval_queries_from_file(path: str) -> list[EvalQuery]:
    with open(path, "r", encoding="utf-8") as f:
        items = json.load(f)

    eval_queries: list[EvalQuery] = []
    for obj in items:
        # поддерживаем и "query", и "question"
        if "query" in obj:
            query_text = obj["query"]
        elif "question" in obj:
            query_text = obj["question"]
        else:
            continue  # или raise

        relevant_articles = set(obj.get("relevant_articles", []))
        source = obj.get("source", "unknown")

        # если хочешь graded relevance:
        relevance_grades = obj.get("relevance_grades")  # {"article_id": 2, ...} или None

        eval_queries.append(
            EvalQuery(
                query_text=query_text,
                relevant_articles=relevant_articles,
                source=source,
                relevance_grades=relevance_grades,
            )
        )

    return eval_queries


def evaluate_retriever_on_queries(
    client: QdrantClient,
    queries: Iterable[EvalQuery],
    embedder,
    COLLECTION_NAME,
    top_k_articles: int = 10,
    top_k_chunks: int = 10,
) -> EvalMetrics:
    queries = list(queries)
    if not queries:
        return EvalMetrics(0.0, 0.0, 0.0, 0.0)

    mrr_sum = 0.0
    hit_sum = 0.0
    recall_sum = 0.0
    precision_sum = 0.0

    for q in queries:
        # 1. получаем ранжированный список статей из системы
        ranked_articles, _scores = retrieve_ranked_articles(
            client,
            embedder,
            q.query_text,
            COLLECTION_NAME,
            top_k_articles=top_k_articles,
            top_k_chunks=top_k_chunks,
        )

        relevant = q.relevant_articles
        if not relevant:
            continue  # или считать, что по нему нечего мерить

        # 2. MRR: позиция первой релевантной
        rank = None
        for idx, art_id in enumerate(ranked_articles, start=1):
            if art_id in relevant:
                rank = idx
                break

        if rank is not None:
            mrr_sum += 1.0 / rank
            hit_sum += 1.0
        else:
            # hit остаётся 0
            pass

        # 3. Recall@k и Precision@k
        top_k_list = ranked_articles[:top_k_articles]
        found_relevant = sum(1 for art_id in top_k_list if art_id in relevant)

        recall = found_relevant / len(relevant)
        precision = found_relevant / top_k_articles

        recall_sum += recall
        precision_sum += precision

    n = len(queries)
    return EvalMetrics(
        mrr=mrr_sum / n,
        hit_at_k=hit_sum / n,
        recall_at_k=recall_sum / n,
        precision_at_k=precision_sum / n,
    )


def evaluate_by_source(
    client: QdrantClient,
    eval_queries: list[EvalQuery],
    embedder: Embedder,
    COLLECTION_NAME: str,
    top_k_articles: int = 10,
    top_k_chunks: int = 10,
):
    # общие метрики
    overall = evaluate_retriever_on_queries(
        client,
        eval_queries,
        embedder,
        COLLECTION_NAME,
        top_k_articles=top_k_articles,
        top_k_chunks=top_k_chunks,
    )

    # по source
    grouped: dict[str, list[EvalQuery]] = defaultdict(list)
    for q in eval_queries:
        grouped[q.source].append(q)

    per_source = {}
    for source, qs in grouped.items():
        per_source[source] = evaluate_retriever_on_queries(
            client,
            qs,
            embedder,
            COLLECTION_NAME,
            top_k_articles=top_k_articles,
            top_k_chunks=top_k_chunks,
        )

    return overall, per_source


def retrieve_ranked_articles(
    client: QdrantClient,
    embedder: Embedder,
    query_text: str,
    COLLECTION_NAME: str,
    top_k_articles: int = 10,
    top_k_chunks: int = 10,
    year: int | None = None,
):
    query_vec = embedder.embed_query(query_text).tolist()

    qdrant_filter = None
    if year is not None:
        qdrant_filter = Filter(
            must=[
                FieldCondition(
                    key="year",
                    range=Range(gte=year, lte=year)
                )
            ]
        )

    # 1. ищем чанки
    chunk_results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vec,
        query_filter=qdrant_filter,
        limit=top_k_chunks,
        with_payload=True,
    )

    # 2. агрегируем по статьям
    article_scores = {}
    #print(chunk_results.points)

    for r in chunk_results.points:
        art_id = r.payload["full_article_id"]
        score = r.score
        # агрегатор: max или sum
        if art_id not in article_scores:
            article_scores[art_id] = score
        else:
            article_scores[art_id] = max(article_scores[art_id], score)

    # 3. сортируем статьи
    ranked_articles = sorted(
        article_scores.items(),
        key=lambda x: x[1],
        reverse=True,
    )

    # 4. ограничиваем top_k_articles
    ranked_articles = ranked_articles[:top_k_articles]

    # возвращаем список article_id и, при желании, скоры
    article_ids = [a for a, s in ranked_articles]
    return article_ids, article_scores

