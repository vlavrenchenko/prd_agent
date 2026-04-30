"""Тесты search.py — ChromaDB мокируется, API не вызывается."""
import pytest
from importlib import reload
from unittest.mock import patch, MagicMock
from pathlib import Path


def make_mock_collection(existing_ids=None):
    """Создаёт мок ChromaDB коллекции."""
    collection = MagicMock()
    collection.get.return_value = {"ids": existing_ids or []}
    collection.count.return_value = 3
    collection.query.return_value = {
        "documents": [["Текст PRD 1", "Текст PRD 2", "Текст PRD 3"]],
        "metadatas": [
            [
                {"filename": "001_test.md", "source": "/path/001_test.md"},
                {"filename": "002_test.md", "source": "/path/002_test.md"},
                {"filename": "003_test.md", "source": "/path/003_test.md"},
            ]
        ],
        "distances": [[0.1, 0.2, 0.3]],
    }
    return collection


def test_add_prd_new_file(tmp_prd_file):
    import search
    reload(search)
    mock_collection = make_mock_collection(existing_ids=[])
    with patch("search._get_collection", return_value=mock_collection):
        result = search.add_prd(tmp_prd_file)
    assert result is True
    mock_collection.add.assert_called_once()


def test_add_prd_duplicate(tmp_prd_file):
    import search
    reload(search)
    mock_collection = make_mock_collection(existing_ids=["test_prd"])
    with patch("search._get_collection", return_value=mock_collection):
        result = search.add_prd(tmp_prd_file)
    assert result is False
    mock_collection.add.assert_not_called()


def test_add_prd_uses_stem_as_id(tmp_prd_file):
    import search
    reload(search)
    mock_collection = make_mock_collection(existing_ids=[])
    with patch("search._get_collection", return_value=mock_collection):
        search.add_prd(tmp_prd_file)
    call_kwargs = mock_collection.add.call_args
    assert call_kwargs[1]["ids"] == ["test_prd"]


def test_search_returns_list():
    import search
    reload(search)
    mock_collection = make_mock_collection()
    with patch("search._get_collection", return_value=mock_collection):
        results = search.search("программа лояльности", n=3)
    assert isinstance(results, list)
    assert len(results) == 3


def test_search_result_fields():
    import search
    reload(search)
    mock_collection = make_mock_collection()
    with patch("search._get_collection", return_value=mock_collection):
        results = search.search("тест")
    for r in results:
        assert "rank" in r
        assert "filename" in r
        assert "score" in r
        assert "text" in r


def test_search_score_between_0_and_1():
    import search
    reload(search)
    mock_collection = make_mock_collection()
    with patch("search._get_collection", return_value=mock_collection):
        results = search.search("тест")
    for r in results:
        assert 0 <= r["score"] <= 1


def test_index_directory_adds_files(tmp_prd_dir):
    import search
    reload(search)
    mock_collection = make_mock_collection(existing_ids=[])
    with patch("search._get_collection", return_value=mock_collection):
        added, skipped = search.index_directory(tmp_prd_dir)
    assert added == 3
    assert skipped == 0


def test_index_directory_skips_existing(tmp_prd_dir):
    import search
    reload(search)
    # все файлы уже есть в коллекции
    mock_collection = make_mock_collection(existing_ids=["exists"])
    mock_collection.get.return_value = {"ids": ["exists"]}
    with patch("search._get_collection", return_value=mock_collection):
        added, skipped = search.index_directory(tmp_prd_dir)
    assert added == 0
    assert skipped == 3


def test_index_directory_empty(tmp_path):
    import search
    reload(search)
    added, skipped = search.index_directory(tmp_path)
    assert added == 0
    assert skipped == 0


def test_estimate_index_cost(tmp_prd_dir):
    import search
    reload(search)
    # не должен бросать исключений и не вызывает API
    search.estimate_index_cost(tmp_prd_dir)
