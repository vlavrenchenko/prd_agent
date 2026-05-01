import pytest
from pathlib import Path
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).parent.parent


@pytest.fixture(autouse=True)
def suppress_cost_logging():
    """Заглушает запись в costs.log во всех unit-тестах."""
    with patch("agent._log_cost", lambda *a, **kw: None), \
         patch("generate_synthetic._cost_log") as mock_log:
        mock_log.info = lambda *a, **kw: None
        yield


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
