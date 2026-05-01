"""
Сохранение готового PRD в файл и индексация в ChromaDB.
"""
import datetime
from pathlib import Path

from search import add_prd
from logger import get_logger

log = get_logger("save")

OUTPUT_DIR = Path(__file__).parent / "data" / "prd_output"


def save_prd(content: str, feature_description: str) -> str:
    """Сохраняет PRD в .md файл и добавляет в ChromaDB. Возвращает путь к файлу."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    slug = feature_description[:40].lower().replace(" ", "_")
    slug = "".join(c for c in slug if c.isalnum() or c == "_")
    date = datetime.date.today().strftime("%Y%m%d")
    path = OUTPUT_DIR / f"{date}_{slug}.md"

    path.write_text(content, encoding="utf-8")
    indexed = add_prd(path)

    log.info("save_prd_done", extra={
        "path": str(path),
        "feature": feature_description,
        "indexed": indexed,
        "size_bytes": len(content.encode()),
    })

    return str(path)
