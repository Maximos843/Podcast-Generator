import gc
from typing import List, Literal
import uuid

import torch
from tqdm.auto import tqdm
from qdrant_client import QdrantClient, models
from qdrant_client.models import (
    Distance, VectorParams, SparseVectorParams, Modifier,
    Filter, FieldCondition, Range,
)
from src.retrieval.model import BaseEmbedder
from src.config import QDrantConfig
from src.domain.types import Chunk


def index_chunks_in_qdrant(
    client: QdrantClient,
    embedder: BaseEmbedder,
    chunks: List[Chunk],
    collection_name: str,
    mode: Literal["dense", "hybrid"] = "hybrid",
    batch_size: int = 32,
):
    """
    mode="dense": vector=list[float]
    mode="hybrid": vector={"dense": [...], "bm25": Document(...)}
    """
    if mode not in ("dense", "hybrid"):
        raise ValueError("mode must be 'dense' or 'hybrid'")

    avg_len = None
    if mode == "hybrid":
        avg_len = sum(len(ch.text.split()) for ch in chunks) / max(1, len(chunks))

    pbar = tqdm(total=len(chunks), desc=f"indexing chunks [{mode}] ({getattr(embedder, 'device', 'n/a')})")
    for i in range(0, len(chunks), batch_size):
        batch_chunks = chunks[i:i + batch_size]
        texts = [ch.text for ch in batch_chunks]
        dense_emb = embedder.embed_passages(texts)

        points = []
        for j, ch in enumerate(batch_chunks):
            art = ch.article
            full_article_id = f"{art.page_path}#article_{art.article_id}"

            payload = {
                "page_path": art.page_path,
                "page_idx": art.page_idx,
                "article_id": art.article_id,
                "chunk_id": ch.chunk_id,
                "year": art.year,
                "full_article_id": full_article_id,
                "text": ch.text,
            }

            if mode == "dense":
                vec = dense_emb[j].tolist()
            else:
                vec = {
                    QDrantConfig.DENSE_VECTOR_NAME: dense_emb[j].tolist(),
                    QDrantConfig.BM25_VECTOR_NAME: models.Document(
                        text=ch.text,
                        model=QDrantConfig.BM25_MODEL,
                        options={"avg_len": avg_len},
                    )
                }

            points.append(models.PointStruct(
                id=str(uuid.uuid4()),
                vector=vec,
                payload=payload,
            ))

        client.upsert(collection_name=collection_name, points=points)

        del dense_emb, points, texts, batch_chunks
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

        pbar.update(min(batch_size, len(chunks) - i))
    pbar.close()


def recreate_collection_dense(client: QdrantClient, collection_name: str, embedder: BaseEmbedder):
    client.recreate_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=embedder.get_dim(), distance=Distance.COSINE),
    )


def recreate_collection_hybrid(client: QdrantClient, collection_name: str, embedder: BaseEmbedder):
    client.recreate_collection(
        collection_name=collection_name,
        vectors_config={
            QDrantConfig.DENSE_VECTOR_NAME: VectorParams(
                size=embedder.get_dim(),
                distance=Distance.COSINE,
            )
        },
        sparse_vectors_config={
            QDrantConfig.BM25_VECTOR_NAME: SparseVectorParams(modifier=Modifier.IDF)
        },
    )