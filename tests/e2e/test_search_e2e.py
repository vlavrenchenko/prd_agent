"""
E2e тесты search.py — использует реальную ChromaDB (без OpenAI API).
Требует: проиндексированная база (python search.py --index).
"""
import pytest
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent


def test_chroma_db_exists():
    """ChromaDB директория создана после индексации."""
    chroma_dir = PROJECT_ROOT / "chroma_db"
    assert chroma_dir.exists(), "chroma_db/ не найдена — запусти: python search.py --index"


def test_search_returns_results():
    """Реальный поиск возвращает результаты из проиндексированной базы."""
    import search
    results = search.search("программа лояльности для покупателей", n=3)
    assert len(results) > 0


def test_search_result_structure():
    """Результаты поиска содержат все нужные поля."""
    import search
    results = search.search("онбординг нового пользователя", n=1)
    assert len(results) > 0
    r = results[0]
    assert "rank" in r
    assert "filename" in r
    assert "score" in r
    assert "text" in r


def test_search_score_is_valid():
    """Оценка релевантности в диапазоне 0–1."""
    import search
    results = search.search("аналитика расходов", n=3)
    for r in results:
        assert 0 <= r["score"] <= 1, f"score вне диапазона: {r['score']}"


def test_search_returns_markdown():
    """Текст результата содержит markdown PRD."""
    import search
    results = search.search("виш-лист", n=1)
    assert len(results) > 0
    assert "#" in results[0]["text"]


def test_add_and_search_new_prd(tmp_path):
    """Добавленный PRD находится в поиске."""
    import search

    prd_file = tmp_path / "e2e_test_prd.md"
    prd_file.write_text(
        "# PRD: Уникальная e2e тестовая фича xyzabc123\n\n"
        "## Проблема\nТестовая проблема для e2e проверки поиска.",
        encoding="utf-8",
    )

    added = search.add_prd(prd_file)
    assert added is True

    results = search.search("Уникальная e2e тестовая фича xyzabc123", n=1)
    assert len(results) > 0
    assert "e2e_test_prd" in results[0]["filename"]
