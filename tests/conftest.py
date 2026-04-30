import pytest
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent


@pytest.fixture
def tmp_prd_file(tmp_path):
    """Временный .md файл с содержимым PRD."""
    f = tmp_path / "test_prd.md"
    f.write_text("# PRD: Тестовая фича\n\n## Проблема\nТестовое содержимое.", encoding="utf-8")
    return f


@pytest.fixture
def tmp_prd_dir(tmp_path):
    """Временная директория с несколькими .md файлами."""
    for i in range(3):
        f = tmp_path / f"prd_{i:03d}.md"
        f.write_text(f"# PRD #{i}\n\nТестовый документ номер {i}.", encoding="utf-8")
    return tmp_path
