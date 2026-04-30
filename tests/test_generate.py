"""Тесты generate_synthetic.py — без вызовов API."""
import pytest
from importlib import reload
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent


def test_to_markdown_list():
    import generate_synthetic
    result = generate_synthetic._to_markdown(["пункт 1", "пункт 2", "пункт 3"])
    assert result == "- пункт 1\n- пункт 2\n- пункт 3"


def test_to_markdown_string():
    import generate_synthetic
    result = generate_synthetic._to_markdown("просто строка")
    assert result == "просто строка"


def test_render_prd_fills_placeholders():
    import generate_synthetic
    template = "# PRD: {title}\n\n## Проблема\n{problem}"
    fields = {"title": "Тестовая фича", "problem": "Описание проблемы"}
    result = generate_synthetic.render_prd(template, fields)
    assert "Тестовая фича" in result
    assert "Описание проблемы" in result
    assert "{title}" not in result
    assert "{problem}" not in result


def test_render_prd_list_field():
    import generate_synthetic
    template = "{user_stories}"
    fields = {"user_stories": ["История 1", "История 2"]}
    result = generate_synthetic.render_prd(template, fields)
    assert "- История 1" in result
    assert "- История 2" in result


def test_make_filename_format():
    import generate_synthetic
    name = generate_synthetic.make_filename("e-commerce", "виш-лист", 5)
    assert name.startswith("005_")
    assert name.endswith(".md")
    assert "e_commerce" in name


def test_make_filename_truncates_long_feature():
    import generate_synthetic
    long_feature = "очень длинное название фичи которое превышает допустимый лимит символов"
    name = generate_synthetic.make_filename("fintech", long_feature, 1)
    assert len(name) < 100


def test_build_task_list_count():
    import generate_synthetic
    tasks = generate_synthetic.build_task_list(None, 10)
    assert len(tasks) == 10


def test_build_task_list_domain_filter():
    import generate_synthetic
    tasks = generate_synthetic.build_task_list("fintech", 5)
    assert len(tasks) == 5
    for domain, _ in tasks:
        assert domain == "fintech"


def test_build_task_list_repeats_if_needed():
    import generate_synthetic
    # fintech содержит 8 фич, просим 20 — должен повторить
    tasks = generate_synthetic.build_task_list("fintech", 20)
    assert len(tasks) == 20


def test_load_model_prices_returns_dict():
    import generate_synthetic
    prices = generate_synthetic.load_model_prices()
    assert isinstance(prices, dict)
    assert len(prices) > 0


def test_load_model_prices_has_input_output():
    import generate_synthetic
    prices = generate_synthetic.load_model_prices()
    for model_id, price in prices.items():
        assert "input" in price, f"{model_id}: нет поля input"
        assert "output" in price, f"{model_id}: нет поля output"
