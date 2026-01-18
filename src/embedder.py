from sentence_transformers import SentenceTransformer
import numpy as np


class Embedder:
    def __init__(
        self,
        model_name: str = "intfloat/multilingual-e5-base",
        query_prefix: str = "query: ",
        passage_prefix: str = "passage: ",
        device: str = "cpu",
        encode_batch_size: int = 8,   # <-- внутренний batch_size
    ):
        self.model_name = model_name
        self.query_prefix = query_prefix
        self.passage_prefix = passage_prefix
        self.device = device
        self.encode_batch_size = encode_batch_size

        self.model = SentenceTransformer(model_name, device=device)
        self.dim = self.model.get_sentence_embedding_dimension()
    
    def get_dim(self):
        return self.dim

    def embed_passages(self, texts: list[str]) -> np.ndarray:
        inputs = [f"{self.passage_prefix}{t}" for t in texts]
        emb = self.model.encode(
            inputs,
            batch_size=self.encode_batch_size,
            show_progress_bar=False,   # <-- внешний tqdm уже есть
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        # чуть стабильнее по памяти и скорости
        return emb.astype(np.float32, copy=False)

    def embed_query(self, text: str) -> np.ndarray:
        inp = f"{self.query_prefix}{text}"
        emb = self.model.encode(
            [inp],
            batch_size=1,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )[0]
        return emb.astype(np.float32, copy=False)