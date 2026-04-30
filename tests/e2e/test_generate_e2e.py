"""
E2e тесты generate_synthetic.py — реальный вызов OpenAI API.
Требует: OPENAI_API_KEY в .env
"""
import pytest
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent


def test_generate_one_returns_valid_json():
    """Один реальный вызов API возвращает валидный JSON с нужными полями."""
    import json
    from openai import OpenAI
    from dotenv import load_dotenv
    import generate_synthetic

    load_dotenv(override=True)
    client = OpenAI()

    raw, usage = generate_synthetic.generate_one(client, "fintech", "P2P переводы", "gpt-4o-mini")
    fields = json.loads(raw)

    required = ["title", "problem", "goal", "users", "user_stories",
                "functional_requirements", "out_of_scope", "success_metrics",
                "open_questions", "dependencies"]
    for field in required:
        assert field in fields, f"Поле '{field}' отсутствует в ответе"


def test_generate_one_returns_usage():
    """API возвращает информацию о потреблении токенов."""
    from openai import OpenAI
    from dotenv import load_dotenv
    import generate_synthetic

    load_dotenv(override=True)
    client = OpenAI()

    _, usage = generate_synthetic.generate_one(client, "edtech", "AI-репетитор", "gpt-4o-mini")
    assert usage["input"] > 0
    assert usage["output"] > 0


def test_render_and_save_prd(tmp_path):
    """Полный цикл: генерация → рендер → сохранение файла."""
    import json
    from openai import OpenAI
    from dotenv import load_dotenv
    import generate_synthetic

    load_dotenv(override=True)
    client = OpenAI()
    template = (PROJECT_ROOT / "config" / "prd_template.md").read_text(encoding="utf-8")

    raw, _ = generate_synthetic.generate_one(client, "productivity", "time tracking", "gpt-4o-mini")
    fields = json.loads(raw)
    prd_text = generate_synthetic.render_prd(template, fields)

    output = tmp_path / "test_prd.md"
    output.write_text(prd_text, encoding="utf-8")

    assert output.exists()
    assert len(prd_text) > 500
    assert "# PRD:" in prd_text
    assert "{" not in prd_text, "В PRD остались незаполненные плейсхолдеры"
