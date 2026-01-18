from qdrant_client.models import Filter, PointStruct, FieldCondition, Range
from embedder import Embedder
from config import QDrantConfig, Chunk, Article
from qdrant_client import QdrantClient
from preprocessing import chunk_text_by_sentences
from tqdm.auto import tqdm
import gc
import torch
import uuid


def make_chunks(articles: list[Article]) -> list[Chunk]:
    """Разбиваем статьи на чанки с прогресс-баром, пригодным для Jupyter/CLI."""
    chunks: list[Chunk] = []
    for art in tqdm(articles, total=len(articles), desc="chunking articles"):
        texts = chunk_text_by_sentences(art.cleaned_text)
        for idx, txt in enumerate(texts):
            if not txt.strip():
                continue
            chunks.append(Chunk(article=art, chunk_id=idx, text=txt))
    return chunks


def chunk_to_point(chunk: Chunk, vector: list[float]) -> PointStruct:
    art = chunk.article
    full_article_id = f"{art.page_path}#article_{art.article_id}"

    return PointStruct(
        id=str(uuid.uuid4()),
        vector=vector,
        payload={
            "page_path": art.page_path,
            "page_idx": art.page_idx,
            "article_id": art.article_id,
            "chunk_id": chunk.chunk_id,
            "year": art.year,
            "full_article_id": full_article_id,
            "text": chunk.text,
            # можно добавить при желании:
            # "original_article_text": art.original_text,
        },
    )


def index_chunks_in_qdrant(
    client,
    embedder: Embedder,
    chunks: list,
    collection_name: str,
    batch_size: int = 32,
):
    pbar = tqdm(total=len(chunks), desc=f"indexing chunks ({embedder.device})")
    for i in range(0, len(chunks), batch_size):
        batch_chunks = chunks[i:i + batch_size]
        texts = [ch.text for ch in batch_chunks]

        emb = embedder.embed_passages(texts)

        points = [
            chunk_to_point(ch, emb[j].tolist())
            for j, ch in enumerate(batch_chunks)
        ]

        client.upsert(
            collection_name=collection_name,
            points=points,
        )

        del emb, points, texts, batch_chunks
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

        pbar.update(min(batch_size, len(chunks) - i))
    pbar.close()



def search_chunks(
    client: QdrantClient,
    embedder: Embedder,
    query_text: str,
    collection_name: str,
    top_k: int = 10,
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

    results = client.query_points(
        collection_name=collection_name,
        query=query_vec,
        query_filter=qdrant_filter,
        limit=top_k,
        with_payload=True,
    )

    return results


def retrieve_ranked_articles(
    client: QdrantClient,
    embedder: Embedder,
    query_text: str,
    top_k_articles: int = 10,
    top_k_chunks: int = 50,
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
        collection_name=QDrantConfig.COLLECTION_NAME,
        query=query_vec,
        query_filter=qdrant_filter,
        limit=top_k_chunks,
        with_payload=True,
    )

    # 2. агрегируем по статьям
    article_scores = {}
    print(chunk_results)

    for r in chunk_results[0]:
        #print(r)
        art_id = r[1].payload["full_article_id"]
        score = r[1].score
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
