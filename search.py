"""
RAG-поиск по базе PRD через ChromaDB.

Запуск:
    python search.py --index                        # индексировать data/prd_synthetic/
    python search.py --index --dir data/prd_output  # индексировать другую папку
    python search.py "как реализовать онбординг"    # найти похожие PRD
    python search.py "виш-лист" --top 3             # топ-3 результата
"""
import argparse
import os
from pathlib import Path

import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from dotenv import load_dotenv

load_dotenv(override=True)

SYNTHETIC_DIR = Path(__file__).parent / "data" / "prd_synthetic"
CHROMA_DIR = Path(__file__).parent / "chroma_db"
COLLECTION_NAME = "prd_documents"
EMBEDDING_MODEL = "text-embedding-3-small"

# Цены на эмбеддинг-модели OpenAI ($ за 1M токенов)
EMBEDDING_PRICES = {
    "text-embedding-3-small": 0.02,
    "text-embedding-3-large": 0.13,
    "text-embedding-ada-002": 0.10,
}


def _get_collection() -> chromadb.Collection:
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    ef = OpenAIEmbeddingFunction(
        api_key=os.environ["OPENAI_API_KEY"],
        model_name=EMBEDDING_MODEL,
    )
    return client.get_or_create_collection(name=COLLECTION_NAME, embedding_function=ef)


def add_prd(filepath: str | Path) -> bool:
    """Добавляет один PRD-файл в коллекцию. Возвращает True если добавлен, False если уже есть."""
    path = Path(filepath)
    doc_id = path.stem
    collection = _get_collection()

    existing = collection.get(ids=[doc_id])
    if existing["ids"]:
        return False

    text = path.read_text(encoding="utf-8")
    collection.add(
        ids=[doc_id],
        documents=[text],
        metadatas=[{"filename": path.name, "source": str(path)}],
    )
    return True


def index_directory(directory: str | Path = SYNTHETIC_DIR) -> tuple[int, int]:
    """Индексирует все .md файлы в директории. Возвращает (добавлено, пропущено)."""
    directory = Path(directory)
    files = sorted(directory.glob("*.md"))
    if not files:
        print(f"  Файлов .md не найдено в {directory}")
        return 0, 0

    added = skipped = 0
    for f in files:
        if add_prd(f):
            print(f"  + {f.name}")
            added += 1
        else:
            skipped += 1

    return added, skipped


def search(query: str, n: int = 5) -> list[dict]:
    """Ищет похожие PRD по запросу. Возвращает список результатов с текстом и метаданными."""
    collection = _get_collection()
    results = collection.query(query_texts=[query], n_results=min(n, collection.count()))
    output = []
    for i, (doc, meta, distance) in enumerate(zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    )):
        output.append({
            "rank": i + 1,
            "filename": meta["filename"],
            "score": round(1 - distance, 4),
            "text": doc,
        })
    return output


def print_results(results: list[dict]):
    if not results:
        print("Ничего не найдено.")
        return
    for r in results:
        print(f"\n{'='*55}")
        print(f"#{r['rank']}  {r['filename']}  (релевантность: {r['score']})")
        print(f"{'='*55}")
        # показываем первые ~400 символов документа
        preview = r["text"][:400].strip()
        if len(r["text"]) > 400:
            preview += "..."
        print(preview)


def estimate_index_cost(directory: str | Path = SYNTHETIC_DIR) -> None:
    """Считает токены и стоимость индексации без обращения к API."""
    import tiktoken
    directory = Path(directory)
    files = sorted(directory.glob("*.md"))
    if not files:
        print(f"Файлов .md не найдено в {directory}")
        return

    enc = tiktoken.get_encoding("cl100k_base")
    total_tokens = 0
    for f in files:
        text = f.read_text(encoding="utf-8")
        total_tokens += len(enc.encode(text))

    price_per_1m = EMBEDDING_PRICES.get(EMBEDDING_MODEL)
    cost = (total_tokens / 1_000_000) * price_per_1m if price_per_1m else None

    print(f"\n{'='*50}")
    print(f"Модель эмбеддингов:  {EMBEDDING_MODEL}")
    print(f"Файлов к индексации: {len(files)}")
    print(f"Токенов суммарно:    {total_tokens:,}")
    if cost is not None:
        print(f"Стоимость:           ${cost:.6f}")
    else:
        print(f"Стоимость:           н/д (модель не в таблице цен)")
    print(f"{'='*50}\n")


def parse_args():
    parser = argparse.ArgumentParser(description="Поиск по базе PRD (ChromaDB)")
    parser.add_argument("query", nargs="?", default=None,
                        help="Поисковый запрос на естественном языке")
    parser.add_argument("--index", action="store_true",
                        help="Индексировать PRD-файлы в ChromaDB")
    parser.add_argument("--dir", type=str, default=None,
                        help=f"Папка для индексации (по умолчанию {SYNTHETIC_DIR})")
    parser.add_argument("--top", type=int, default=5,
                        help="Количество результатов (по умолчанию 5)")
    parser.add_argument("--estimate-index", action="store_true",
                        help="Оценить стоимость индексации без обращения к API")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.estimate_index:
        directory = Path(args.dir) if args.dir else SYNTHETIC_DIR
        estimate_index_cost(directory)
        return

    if args.index:
        directory = Path(args.dir) if args.dir else SYNTHETIC_DIR
        print(f"Индексируем {directory}...")
        added, skipped = index_directory(directory)
        print(f"\nГотово: добавлено {added}, пропущено {skipped} (уже в базе)")
        collection = _get_collection()
        print(f"Всего документов в базе: {collection.count()}")
        return

    if args.query:
        print(f'Ищем: "{args.query}" (топ {args.top})\n')
        results = search(args.query, n=args.top)
        print_results(results)
        return

    parser.print_help()


if __name__ == "__main__":
    main()
