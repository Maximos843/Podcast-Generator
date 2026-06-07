from abc import ABC, abstractmethod
from typing import Iterable
from dataclasses import dataclass



@dataclass(frozen=True)
class ArticleRecord:
    full_article_id: str
    page_path: str
    page_idx: int
    article_id: str
    year: int | None
    cleaned_text: str
    original_text: str | None = None


class BaseArticleStore(ABC):
    @abstractmethod
    def get(self, full_article_id: str) -> ArticleRecord | None:
        raise NotImplementedError

    def __contains__(self, full_article_id: str) -> bool:
        return self.get(full_article_id) is not None

    def get_many(self, full_article_ids: Iterable[str]) -> dict[str, ArticleRecord]:
        """
        По умолчанию — наивно через get() (подойдет для InMemory).
        SQLite переопределит на один SQL-запрос.
        """
        out: dict[str, ArticleRecord] = {}
        for fid in full_article_ids:
            rec = self.get(fid)
            if rec is not None:
                out[fid] = rec
        return out
