"""
E2e тесты agent.py — реальный вызов OpenAI API + ChromaDB.
Требует: OPENAI_API_KEY в .env, проиндексированная база.
"""
import uuid
import pytest
from dotenv import load_dotenv
from langgraph.types import Command

load_dotenv(override=True)


def test_agent_generates_prd_with_skip(tmp_path):
    """Полный прогон агента с /skip — возвращает готовый PRD."""
    import agent

    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    initial_state = {
        "feature_description": "добавить виш-лист в мобильное приложение",
        "rag_context": [],
        "questions": [],
        "answers": "",
        "skipped": False,
        "prd": "",
        "output_path": "",
    }

    from unittest.mock import patch
    fake_path = str(tmp_path / "output.md")
    (tmp_path / "output.md").write_text("", encoding="utf-8")

    with patch("agent.save_prd", return_value=fake_path):
        graph = agent.build_graph()
        graph.invoke(initial_state, config=config)
        result = graph.invoke(Command(resume="/skip"), config=config)

    assert result["prd"], "PRD не сгенерирован"
    assert len(result["prd"]) > 200
    assert result["skipped"] is True


def test_agent_prd_contains_template_sections(tmp_path):
    """Сгенерированный PRD содержит ключевые секции шаблона."""
    import agent

    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    initial_state = {
        "feature_description": "система уведомлений о снижении цены",
        "rag_context": [],
        "questions": [],
        "answers": "",
        "skipped": False,
        "prd": "",
        "output_path": "",
    }

    from unittest.mock import patch
    fake_path = str(tmp_path / "output.md")
    (tmp_path / "output.md").write_text("", encoding="utf-8")

    with patch("agent.save_prd", return_value=fake_path):
        graph = agent.build_graph()
        graph.invoke(initial_state, config=config)
        result = graph.invoke(Command(resume="/skip"), config=config)

    prd = result["prd"]
    for section in ["Проблема", "Цель", "Пользователи", "Метрики"]:
        assert section in prd, f"Секция '{section}' не найдена в PRD"


def test_agent_ask_questions_then_answer(tmp_path):
    """Агент задаёт вопросы и генерирует PRD с учётом ответа."""
    import agent
    from langgraph.types import Command

    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    initial_state = {
        "feature_description": "добавить трекер привычек",
        "rag_context": [],
        "questions": [],
        "answers": "",
        "skipped": False,
        "prd": "",
        "output_path": "",
    }

    from unittest.mock import patch
    fake_path = str(tmp_path / "output.md")
    (tmp_path / "output.md").write_text("", encoding="utf-8")

    with patch("agent.save_prd", return_value=fake_path):
        graph = agent.build_graph()

        # Первый invoke — агент задаёт вопросы и останавливается
        graph.invoke(initial_state, config=config)

        snapshot = graph.get_state(config)
        assert snapshot.next, "Граф должен ждать ответа пользователя"

        # Отвечаем на вопросы
        answer = "Целевые пользователи — люди 25-35 лет. Привычки: спорт, чтение, вода."
        result = graph.invoke(Command(resume=answer), config=config)

    assert result["prd"], "PRD не сгенерирован"
    assert result["skipped"] is False
    assert result["answers"] == answer
