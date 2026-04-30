"""Тесты agent.py — OpenAI и ChromaDB замокированы."""
import pytest
from importlib import reload
from unittest.mock import patch, MagicMock
from pathlib import Path


def make_llm_response(content: str):
    response = MagicMock()
    response.choices[0].message.content = content
    return response


def test_search_context_node():
    """search_context заполняет rag_context в state."""
    import agent
    reload(agent)
    mock_results = [{"rank": 1, "filename": "test.md", "score": 0.9, "text": "PRD текст"}]
    with patch("agent.rag_search", return_value=mock_results):
        result = agent.search_context({"feature_description": "виш-лист"})
    assert result["rag_context"] == mock_results


def test_generate_node_calls_llm():
    """generate вызывает OpenAI и кладёт PRD в state."""
    import agent
    reload(agent)
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = make_llm_response("# Сгенерированный PRD")

    state = {
        "feature_description": "виш-лист",
        "rag_context": [],
        "questions": ["Вопрос 1"],
        "answers": "Ответ 1",
        "skipped": False,
        "prd": "",
        "output_path": "",
    }

    with patch("agent.OpenAI", return_value=mock_client):
        result = agent.generate(state)

    assert result["prd"] == "# Сгенерированный PRD"
    mock_client.chat.completions.create.assert_called_once()


def test_generate_node_skip_adds_draft_note():
    """generate с skipped=True добавляет пометку о неполноте."""
    import agent
    reload(agent)
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = make_llm_response("# PRD Draft")

    state = {
        "feature_description": "виш-лист",
        "rag_context": [],
        "questions": [],
        "answers": "",
        "skipped": True,
        "prd": "",
        "output_path": "",
    }

    with patch("agent.OpenAI", return_value=mock_client):
        agent.generate(state)

    call_messages = mock_client.chat.completions.create.call_args[1]["messages"]
    system_content = call_messages[0]["content"]
    assert "требует доработки" in system_content


def test_save_node(tmp_path):
    """save сохраняет файл и кладёт путь в state."""
    import agent
    reload(agent)
    fake_path = str(tmp_path / "20260430_виш_лист.md")

    state = {
        "feature_description": "виш-лист",
        "rag_context": [],
        "questions": [],
        "answers": "",
        "skipped": False,
        "prd": "# PRD содержимое",
        "output_path": "",
    }

    with patch("agent.save_prd", return_value=fake_path):
        result = agent.save(state)

    assert result["output_path"] == fake_path


def test_full_graph_skip(tmp_path):
    """Полный прогон графа с /skip — генерирует PRD без вопросов."""
    import agent
    reload(agent)

    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = make_llm_response("# Финальный PRD")
    fake_path = str(tmp_path / "output.md")

    import uuid
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    initial_state = {
        "feature_description": "добавить виш-лист",
        "rag_context": [],
        "questions": [],
        "answers": "",
        "skipped": False,
        "prd": "",
        "output_path": "",
    }

    with patch("agent.rag_search", return_value=[]), \
         patch("agent.OpenAI", return_value=mock_client), \
         patch("agent.save_prd", return_value=fake_path):

        graph = agent.build_graph()

        # Первый invoke — остановится на interrupt
        graph.invoke(initial_state, config=config)

        from langgraph.types import Command
        result = graph.invoke(Command(resume="/skip"), config=config)

    assert result["prd"] == "# Финальный PRD"
    assert result["output_path"] == fake_path
    assert result["skipped"] is True
