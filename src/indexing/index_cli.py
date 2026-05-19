# src/indexing/index_cli.py
from __future__ import annotations

import argparse

from qdrant_client import QdrantClient

from src.retrieval.model import SentenceTransformerEmbedder
from src.indexing.index_pipeline import IndexRunConfig, run_indexing
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--json", required=True, help="Path to articles JSON")
    p.add_argument("--sqlite", required=True, help="Path to sqlite db for ArticleStore")
    p.add_argument("--qdrant-url", default="http://localhost:6333")
    p.add_argument("--qdrant-api-key", default=None)
    p.add_argument("--collection", required=True)
    p.add_argument("--mode", choices=["dense", "hybrid"], default="hybrid")
    p.add_argument("--recreate", action="store_true", help="Recreate collection before indexing")
    p.add_argument("--batch-size", type=int, default=32)

    p.add_argument("--embedder-model", default="intfloat/multilingual-e5-base")
    p.add_argument("--device", default="cpu")

    args = p.parse_args()

    client = QdrantClient(url=args.qdrant_url, api_key=args.qdrant_api_key) if args.qdrant_api_key else QdrantClient(url=args.qdrant_url)
    embedder = SentenceTransformerEmbedder(model_name=args.embedder_model, device=args.device)

    cfg = IndexRunConfig(
        json_path=args.json,
        sqlite_db_path=args.sqlite,
        collection_name=args.collection,
        mode=args.mode,
        recreate=args.recreate,
        batch_size=args.batch_size,
    )

    result = run_indexing(client=client, embedder=embedder, cfg=cfg)
    print("DONE:", result)


if __name__ == "__main__":
    main()
