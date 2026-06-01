import json
import sqlite3
from typing import Iterable
from functools import lru_cache
from src.storage.base import ArticleRecord, BaseArticleStore


class InMemoryArticleStore(BaseArticleStore):
    def __init__(self, records: dict[str, ArticleRecord]):
        self._records = records

    @lru_cache(maxsize=10_000)
    def get(self, full_article_id: str) -> ArticleRecord | None:  # type: ignore
        return self._records.get(full_article_id)

    @classmethod
    def from_json(
        cls,
        json_path: str,
        *,
        clean: bool = False,
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
        records: dict[str, ArticleRecord] = {}

        for page_idx, page_path in enumerate(page_paths):
            year = extract_year_fn(page_path) if extract_year_fn else None
            article_map: dict[str, str] = pages[page_path]

            for article_id, raw_text in article_map.items():
                cleaned_text = raw_text
                if clean and clean_text_fn is not None:
                    cleaned_text = clean_text_fn(raw_text, dedupe_adjacent=dedupe_adjacent)

                if not cleaned_text or not cleaned_text.strip():
                    continue

                fid = cls.make_full_article_id(page_path, article_id)
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
    def get(self, full_article_id: str) -> ArticleRecord | None:  # type: ignore
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

        with open(json_path, "r", encoding="utf-8") as f:
            pages = json.load(f)

        page_paths = list(pages.keys())
        def gen_records():
            for page_idx, page_path in enumerate(page_paths):
                year = extract_year_fn(page_path) if extract_year_fn else None
                article_map: dict[str, str] = pages[page_path]
                for article_id, raw_text in article_map.items():
                    cleaned_text = raw_text
                    if clean and clean_text_fn is not None:
                        cleaned_text = clean_text_fn(raw_text, dedupe_adjacent=dedupe_adjacent)
                    if not cleaned_text or not cleaned_text.strip():
                        continue
                    fid = cls.make_full_article_id(page_path, article_id)
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

    def get_many(self, full_article_ids: Iterable[str]) -> dict[str, ArticleRecord]:
        ids = [x for x in full_article_ids if x]
        if not ids:
            return {}

        CHUNK = 500
        out: dict[str, ArticleRecord] = {}

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
