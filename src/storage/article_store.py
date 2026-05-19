# article_store.py
from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from typing import Optional, Dict, Iterable, Tuple
from functools import lru_cache

# Если у тебя уже есть extract_year_from_path / clean_article_text — импортни их:
# from preprocessing import extract_year_from_path, clean_article_text


@dataclass(frozen=True)
class ArticleRecord:
    full_article_id: str
    page_path: str
    page_idx: int
    article_id: str
    year: Optional[int]
    cleaned_text: str
    original_text: Optional[str] = None


def make_full_article_id(page_path: str, article_id: str) -> str:
    return f"{page_path}#article_{article_id}"


class BaseArticleStore:
    def get(self, full_article_id: str) -> Optional[ArticleRecord]:
        raise NotImplementedError

    def __contains__(self, full_article_id: str) -> bool:
        return self.get(full_article_id) is not None

    def get_many(self, full_article_ids: Iterable[str]) -> Dict[str, ArticleRecord]:
        """
        По умолчанию — наивно через get() (подойдет для InMemory).
        SQLite переопределит на один SQL-запрос.
        """
        out: Dict[str, ArticleRecord] = {}
        for fid in full_article_ids:
            rec = self.get(fid)
            if rec is not None:
                out[fid] = rec
        return out


class InMemoryArticleStore(BaseArticleStore):
    def __init__(self, records: Dict[str, ArticleRecord]):
        self._records = records

    @lru_cache(maxsize=10_000)
    def get(self, full_article_id: str) -> Optional[ArticleRecord]:
        return self._records.get(full_article_id)

    @classmethod
    def from_json(
        cls,
        json_path: str,
        *,
        clean: bool = False,
        # hooks:
        extract_year_fn=None,
        clean_text_fn=None,
        dedupe_adjacent: bool = True,
    ) -> "InMemoryArticleStore":
        """
        Ожидаемый формат json:
        {
          "page_path_1": {"0": "...", "1": "..."},
          "page_path_2": {"0": "..."}
        }
        """
        with open(json_path, "r", encoding="utf-8") as f:
            pages = json.load(f)

        page_paths = list(pages.keys())
        records: Dict[str, ArticleRecord] = {}

        for page_idx, page_path in enumerate(page_paths):
            year = extract_year_fn(page_path) if extract_year_fn else None
            article_map: Dict[str, str] = pages[page_path]

            for article_id, raw_text in article_map.items():
                cleaned_text = raw_text
                if clean and clean_text_fn is not None:
                    cleaned_text = clean_text_fn(raw_text, dedupe_adjacent=dedupe_adjacent)

                if not cleaned_text or not cleaned_text.strip():
                    continue

                fid = make_full_article_id(page_path, article_id)
                records[fid] = ArticleRecord(
                    full_article_id=fid,
                    page_path=page_path,
                    page_idx=page_idx,
                    article_id=article_id,
                    year=year,
                    cleaned_text=cleaned_text,
                    original_text=raw_text,
                )

        return cls(records)


class SQLiteArticleStore(BaseArticleStore):
    """
    Для масштаба: 5000 страниц и выше.
    Плюс: быстрый get по ключу, не требует держать всё в памяти.
    """
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._init_db()

    def _connect(self):
        con = sqlite3.connect(self.db_path)
        con.execute("PRAGMA journal_mode=WAL;")
        con.execute("PRAGMA synchronous=NORMAL;")
        return con

    def _init_db(self):
        con = self._connect()
        try:
            con.execute("""
            CREATE TABLE IF NOT EXISTS articles (
                full_article_id TEXT PRIMARY KEY,
                page_path TEXT NOT NULL,
                page_idx INTEGER NOT NULL,
                article_id TEXT NOT NULL,
                year INTEGER,
                cleaned_text TEXT NOT NULL,
                original_text TEXT
            );
            """)
            con.execute("CREATE INDEX IF NOT EXISTS idx_articles_year ON articles(year);")
            con.commit()
        finally:
            con.close()

    def bulk_upsert(self, records: Iterable[ArticleRecord], chunk_size: int = 2000):
        con = self._connect()
        try:
            buf = []
            for r in records:
                buf.append((
                    r.full_article_id, r.page_path, r.page_idx, r.article_id,
                    r.year, r.cleaned_text, r.original_text
                ))
                if len(buf) >= chunk_size:
                    con.executemany("""
                    INSERT INTO articles(full_article_id, page_path, page_idx, article_id, year, cleaned_text, original_text)
                    VALUES(?,?,?,?,?,?,?)
                    ON CONFLICT(full_article_id) DO UPDATE SET
                        page_path=excluded.page_path,
                        page_idx=excluded.page_idx,
                        article_id=excluded.article_id,
                        year=excluded.year,
                        cleaned_text=excluded.cleaned_text,
                        original_text=excluded.original_text;
                    """, buf)
                    con.commit()
                    buf.clear()

            if buf:
                con.executemany("""
                INSERT INTO articles(full_article_id, page_path, page_idx, article_id, year, cleaned_text, original_text)
                VALUES(?,?,?,?,?,?,?)
                ON CONFLICT(full_article_id) DO UPDATE SET
                    page_path=excluded.page_path,
                    page_idx=excluded.page_idx,
                    article_id=excluded.article_id,
                    year=excluded.year,
                    cleaned_text=excluded.cleaned_text,
                    original_text=excluded.original_text;
                """, buf)
                con.commit()

        finally:
            con.close()

    @lru_cache(maxsize=10_000)
    def get(self, full_article_id: str) -> Optional[ArticleRecord]:
        con = self._connect()
        try:
            cur = con.execute("""
                SELECT full_article_id, page_path, page_idx, article_id, year, cleaned_text, original_text
                FROM articles
                WHERE full_article_id = ?
            """, (full_article_id,))
            row = cur.fetchone()
            if not row:
                return None
            return ArticleRecord(
                full_article_id=row[0],
                page_path=row[1],
                page_idx=int(row[2]),
                article_id=row[3],
                year=int(row[4]) if row[4] is not None else None,
                cleaned_text=row[5],
                original_text=row[6],
            )
        finally:
            con.close()

    @classmethod
    def build_from_json(
        cls,
        db_path: str,
        json_path: str,
        *,
        clean: bool = False,
        extract_year_fn=None,
        clean_text_fn=None,
        dedupe_adjacent: bool = True,
    ) -> "SQLiteArticleStore":
        store = cls(db_path)

        # загружаем JSON и стримим в bulk_upsert
        with open(json_path, "r", encoding="utf-8") as f:
            pages = json.load(f)

        page_paths = list(pages.keys())
        def gen_records():
            for page_idx, page_path in enumerate(page_paths):
                year = extract_year_fn(page_path) if extract_year_fn else None
                article_map: Dict[str, str] = pages[page_path]
                for article_id, raw_text in article_map.items():
                    cleaned_text = raw_text
                    if clean and clean_text_fn is not None:
                        cleaned_text = clean_text_fn(raw_text, dedupe_adjacent=dedupe_adjacent)
                    if not cleaned_text or not cleaned_text.strip():
                        continue
                    fid = make_full_article_id(page_path, article_id)
                    yield ArticleRecord(
                        full_article_id=fid,
                        page_path=page_path,
                        page_idx=page_idx,
                        article_id=article_id,
                        year=year,
                        cleaned_text=cleaned_text,
                        original_text=raw_text,
                    )

        store.bulk_upsert(gen_records())
        return store

    def get_many(self, full_article_ids: Iterable[str]) -> Dict[str, ArticleRecord]:
        ids = [x for x in full_article_ids if x]
        if not ids:
            return {}

        # SQLite ограничивает количество параметров в IN (...).
        # 500 — безопасно для разных сборок.
        CHUNK = 500
        out: Dict[str, ArticleRecord] = {}

        con = self._connect()
        try:
            for i in range(0, len(ids), CHUNK):
                part = ids[i : i + CHUNK]
                placeholders = ",".join(["?"] * len(part))
                cur = con.execute(
                    f"""
                    SELECT full_article_id, page_path, page_idx, article_id, year, cleaned_text, original_text
                    FROM articles
                    WHERE full_article_id IN ({placeholders})
                    """,
                    tuple(part),
                )
                for row in cur.fetchall():
                    out[row[0]] = ArticleRecord(
                        full_article_id=row[0],
                        page_path=row[1],
                        page_idx=int(row[2]),
                        article_id=row[3],
                        year=int(row[4]) if row[4] is not None else None,
                        cleaned_text=row[5],
                        original_text=row[6],
                    )
        finally:
            con.close()

        return out
