import re
import json
from tqdm import tqdm
from config import Article, Chunk, YEAR_RE


def clean_article_text(text: str, dedupe_adjacent: bool = True) -> str:
    # Унификация переносов строк
    text = text.replace('\r\n', '\n')

    # 1. Склейка переносов по слогам: "ра-\nбочих" → "рабочих"
    lines = text.split('\n')
    merged_lines = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.endswith('-') and i + 1 < len(lines):
            next_line = lines[i + 1].lstrip()
            # если продолжение начинается со строчной буквы — это перенос
            if next_line and re.match(r'^[а-яёa-z]', next_line):
                # склеиваем и записываем в следующую строку
                lines[i + 1] = line[:-1] + next_line
                i += 1
                continue
        merged_lines.append(line)
        i += 1

    text = '\n'.join(merged_lines)

    # 2. Помечаем абзацные разрывы (если есть) и убираем одинарные переносы
    text = text.replace('\n\n', '<PARA_BREAK>')
    text = text.replace('\n', ' ')
    text = text.replace('<PARA_BREAK>', '\n\n')

    # 3. Чистим подчёркивания, лишние пробелы
    text = re.sub(r'\s*_\s*', ' ', text)     # одиночные "_" → пробел
    text = re.sub(r'[ \t]+', ' ', text)      # несколько пробелов → один
    text = text.strip()

    if not dedupe_adjacent or not text:
        return text

    # 4. Удаляем подряд идущие дублирующиеся предложения
    sentences = re.split(r'(?<=[\.\?!»])\s+', text)
    deduped = []
    prev = None
    for s in sentences:
        s_clean = s.strip()
        if not s_clean:
            continue
        if s_clean == prev:
            # подряд дубль — выкидываем
            continue
        deduped.append(s_clean)
        prev = s_clean

    return ' '.join(deduped)


def chunk_text_by_sentences(
    text: str,
    max_chars: int = 1200,
    overlap_chars: int = 300,
) -> list[str]:
    sentences = re.split(r'(?<=[\.\?!»])\s+', text)
    chunks: list[list[str]] = []
    current: list[str] = []

    def current_len(parts: list[str]) -> int:
        return sum(len(s) + 1 for s in parts)

    i = 0
    while i < len(sentences):
        sent = sentences[i].strip()
        if not sent:
            i += 1
            continue

        # если sentence сам по себе больше max_chars — кладем отдельно и двигаемся дальше
        if len(sent) + 1 > max_chars:
            if current:
                chunks.append(current)
                current = []
            chunks.append([sent])
            i += 1
            continue

        if current_len(current) + len(sent) + 1 <= max_chars:
            current.append(sent)
            i += 1
            continue

        # sent не помещается в текущий чанк -> фиксируем current
        if current:
            prev_current = current
            chunks.append(current)

            # считаем overlap
            overlap: list[str] = []
            total = 0
            for s in reversed(prev_current):
                if total + len(s) + 1 <= overlap_chars:
                    overlap.append(s)
                    total += len(s) + 1
                else:
                    break
            current = list(reversed(overlap))

            # ---- КРИТИЧЕСКАЯ ЗАЩИТА ОТ БЕСКОНЕЧНОГО ЦИКЛА ----
            # Если overlap остался таким, что с ним sent всё равно не влезет,
            # сбрасываем overlap, чтобы следующая итерация смогла положить sent и увеличить i.
            if current and (current_len(current) + len(sent) + 1 > max_chars):
                print("OVERLAP BLOCKS SENT:", len(sent), "overlap_len=", current_len(current))
                current = []
            # ----------------------------------------------------
        else:
            # current пуст, а sent <= max_chars -> кладём sent отдельно
            chunks.append([sent])
            i += 1

    if current:
        chunks.append(current)

    return [' '.join(ch) for ch in chunks if ch]


def extract_year_from_path(path: str) -> int | None:
    m = YEAR_RE.search(path)
    return int(m.group(1)) if m else None


def load_articles_from_json(json_path: str) -> list[Article]:
    with open(json_path, "r", encoding="utf-8") as f:
        pages = json.load(f)

    articles: list[Article] = []
    page_paths = list(pages.keys())

    for page_idx, page_path in enumerate(page_paths):
        year = extract_year_from_path(page_path)
        article_map: dict[str, str] = pages[page_path]

        for article_id, raw_text in article_map.items():
            cleaned = clean_article_text(raw_text, dedupe_adjacent=True)
            if not cleaned:
                continue

            art = Article(
                page_path=page_path,
                page_idx=page_idx,
                article_id=article_id,
                year=year,
                original_text=raw_text,
                cleaned_text=cleaned,
            )
            articles.append(art)

    return articles

