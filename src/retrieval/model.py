from dataclasses import dataclass
from typing import Protocol
from abc import abstractmethod

import numpy as np
from sentence_transformers import SentenceTransformer


class BaseEmbedder(Protocol):
    device: str

    @abstractmethod
    def get_dim(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def embed_passages(self, texts: list[str]) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def embed_query(self, text: str) -> np.ndarray:
        raise NotImplementedError


class BaseReranker(Protocol):
    @abstractmethod
    def score(self, query: str, passages: list[str], batch_size: int = 16) -> list[float]:
        raise NotImplementedError


@dataclass
class SentenceTransformerEmbedder:
    model_name: str = "intfloat/multilingual-e5-base"
    query_prefix: str = "query: "
    passage_prefix: str = "passage: "
    device: str = "cpu"
    encode_batch_size: int = 8
    normalize: bool = True

    def __post_init__(self):
        self.model = SentenceTransformer(self.model_name, device=self.device)
        self._dim = self.model.get_sentence_embedding_dimension()

    def get_dim(self) -> int | None:
        return self._dim

    def embed_passages(self, texts: list[str]) -> np.ndarray:
        inputs = [f"{self.passage_prefix}{t}" for t in texts]
        emb = self.model.encode(
            inputs,
            batch_size=self.encode_batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize,
        )
        return emb.astype(np.float32, copy=False)

    def embed_query(self, text: str) -> np.ndarray:
        inp = f"{self.query_prefix}{text}"
        emb = self.model.encode(
            [inp],
            batch_size=1,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize,
        )[0]
        return emb.astype(np.float32, copy=False)


class BGEReranker:

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-v2-m3",
        device: str = "cpu",
        use_fp16: bool = False,
        max_length: int = 512,
    ):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.model_name = model_name
        self.device = device
        self.max_length = max_length

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        if device == "cuda" and use_fp16:
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
            )
        else:
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name)

        self.model.to(device)
        self.model.eval()

    def score(self, query: str, passages: list[str], batch_size: int = 16) -> list[float]:
        import torch

        if not passages:
            return []

        scores: list[float] = []

        for i in range(0, len(passages), batch_size):
            batch_passages = passages[i : i + batch_size]
            pairs = [[query, passage] for passage in batch_passages]

            inputs = self.tokenizer(
                pairs,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=self.max_length,
            )

            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                logits = self.model(**inputs, return_dict=True).logits
                batch_scores = logits.view(-1).float().cpu().tolist()

            scores.extend(float(s) for s in batch_scores)

        return scores
