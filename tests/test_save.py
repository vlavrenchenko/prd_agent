"""Тесты save.py — ChromaDB мокируется."""
import datetime
import pytest
from importlib import reload
from unittest.mock import patch
from pathlib import Path


def test_save_prd_creates_file(tmp_path):
    import save
    reload(save)
    with patch("save.OUTPUT_DIR", tmp_path), patch("save.add_prd", return_value=True):
        path = save.save_prd("# PRD контент", "добавить виш-лист")
    assert Path(path).exists()
    assert Path(path).read_text(encoding="utf-8") == "# PRD контент"


def test_save_prd_filename_contains_date(tmp_path):
    import save
    reload(save)
    today = datetime.date.today().strftime("%Y%m%d")
    with patch("save.OUTPUT_DIR", tmp_path), patch("save.add_prd", return_value=True):
        path = save.save_prd("содержимое", "виш-лист")
    assert today in Path(path).name


def test_save_prd_filename_contains_slug(tmp_path):
    import save
    reload(save)
    with patch("save.OUTPUT_DIR", tmp_path), patch("save.add_prd", return_value=True):
        path = save.save_prd("содержимое", "добавить виш-лист")
    assert "виш_лист" in Path(path).name or "добавить" in Path(path).name


def test_save_prd_calls_add_prd(tmp_path):
    import save
    reload(save)
    with patch("save.OUTPUT_DIR", tmp_path), patch("save.add_prd", return_value=True) as mock_add:
        save.save_prd("содержимое", "тестовая фича")
    mock_add.assert_called_once()


def test_save_prd_returns_string(tmp_path):
    import save
    reload(save)
    with patch("save.OUTPUT_DIR", tmp_path), patch("save.add_prd", return_value=True):
        result = save.save_prd("содержимое", "фича")
    assert isinstance(result, str)


def test_save_prd_creates_output_dir(tmp_path):
    import save
    reload(save)
    new_dir = tmp_path / "nested" / "output"
    with patch("save.OUTPUT_DIR", new_dir), patch("save.add_prd", return_value=True):
        save.save_prd("содержимое", "фича")
    assert new_dir.exists()
