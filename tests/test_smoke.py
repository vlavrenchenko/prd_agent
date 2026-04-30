"""Smoke-тесты — не требуют внешних API."""
import pytest
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent


def test_imports():
    import generate_synthetic
    import search
    import save
    import agent


def test_prd_template_exists():
    path = PROJECT_ROOT / "config" / "prd_template.md"
    assert path.exists(), "config/prd_template.md не найден"
    assert path.stat().st_size > 0


def test_prd_template_has_all_sections():
    path = PROJECT_ROOT / "config" / "prd_template.md"
    content = path.read_text(encoding="utf-8")
    sections = [
        "{problem}", "{goal}", "{users}", "{user_stories}",
        "{functional_requirements}", "{out_of_scope}",
        "{success_metrics}", "{open_questions}", "{dependencies}",
    ]
    for section in sections:
        assert section in content, f"Секция {section} не найдена в шаблоне"


def test_models_pricing_exists():
    path = PROJECT_ROOT / "config" / "models_pricing.json"
    assert path.exists(), "config/models_pricing.json не найден"


def test_models_pricing_valid():
    import json
    path = PROJECT_ROOT / "config" / "models_pricing.json"
    data = json.loads(path.read_text())
    assert "models" in data
    assert len(data["models"]) > 0


def test_synthetic_dir_exists():
    path = PROJECT_ROOT / "data" / "prd_synthetic"
    assert path.exists(), "data/prd_synthetic/ не найдена"


def test_synthetic_prd_files_exist():
    path = PROJECT_ROOT / "data" / "prd_synthetic"
    files = list(path.glob("*.md"))
    assert len(files) > 0, "Нет синтетических PRD файлов"


def test_synthetic_prd_not_empty():
    path = PROJECT_ROOT / "data" / "prd_synthetic"
    for f in list(path.glob("*.md"))[:3]:
        assert f.stat().st_size > 0, f"{f.name} пустой"


def test_output_dir_exists():
    path = PROJECT_ROOT / "data" / "prd_output"
    assert path.exists(), "data/prd_output/ не найдена"


def test_env_example_exists():
    path = PROJECT_ROOT / ".env.example"
    assert path.exists()
    content = path.read_text()
    assert "OPENAI_API_KEY" in content
    assert "TELEGRAM_BOT_TOKEN" in content
