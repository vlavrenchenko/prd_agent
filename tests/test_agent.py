"""Тесты agent.py — OpenAI и ChromaDB замокированы."""
import pytest
from importlib import reload
from unittest.mock import patch, MagicMock
from pathlib import Path


def make_llm_response(content: str):
    response = MagicMock()
    response.choices[0].message.content = content
    return response


def make_critique_response(scores: dict, issues: list):
    import json
    return make_llm_response(json.dumps({"scores": scores, "issues": issues}))


def base_state(**overrides) -> dict:
    state = {
        "feature_description": "виш-лист",
        "rag_context": [],
        "questions": [],
        "answers": "",
        "skipped": False,
        "prd": "",
        "output_path": "",
        "critique_issues": [],
        "critique_score": 0,
        "prev_critique_score": 0,
        "critique_passed": False,
    }
    state.update(overrides)
    return state


# --- search_context ---

def test_search_context_node():
    import agent
    reload(agent)
    mock_results = [{"rank": 1, "filename": "test.md", "score": 0.9, "text": "PRD текст"}]
    with patch("agent.rag_search", return_value=mock_results):
        result = agent.search_context({"feature_description": "виш-лист"})
    assert result["rag_context"] == mock_results


# --- generate ---

def test_generate_node_calls_llm():
    import agent
    reload(agent)
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = make_llm_response("# Сгенерированный PRD")

    with patch("agent.OpenAI", return_value=mock_client):
        result = agent.generate(base_state(questions=["Вопрос 1"], answers="Ответ 1"))

    assert result["prd"] == "# Сгенерированный PRD"
    mock_client.chat.completions.create.assert_called_once()


def test_generate_node_skip_adds_draft_note():
    import agent
    reload(agent)
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = make_llm_response("# PRD Draft")

    with patch("agent.OpenAI", return_value=mock_client):
        agent.generate(base_state(skipped=True))

    call_messages = mock_client.chat.completions.create.call_args[1]["messages"]
    assert "требует доработки" in call_messages[0]["content"]


def test_generate_includes_critique_issues():
    """При повторной генерации промпт содержит замечания критика."""
    import agent
    reload(agent)
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = make_llm_response("# Улучшенный PRD")

    state = base_state(
        prd="# Старый PRD",
        critique_issues=["Нет JTBD", "Метрики без цифр"],
        critique_score=6,
    )

    with patch("agent.OpenAI", return_value=mock_client):
        agent.generate(state)

    call_messages = mock_client.chat.completions.create.call_args[1]["messages"]
    user_content = call_messages[1]["content"]
    assert "Нет JTBD" in user_content
    assert "Метрики без цифр" in user_content


# --- critique ---

def test_critique_passed_when_score_above_threshold():
    import agent
    reload(agent)
    mock_client = MagicMock()
    high_scores = {k: 2 for k in ["metrics", "segment", "requirements", "out_of_scope",
                                    "open_questions", "no_fluff", "jtbd", "business_metric"]}
    mock_client.chat.completions.create.return_value = make_critique_response(high_scores, [])

    with patch("agent.OpenAI", return_value=mock_client):
        result = agent.critique(base_state(prd="# PRD"))

    assert result["critique_passed"] is True
    assert result["critique_score"] == 16
    assert result["critique_issues"] == []


def test_critique_fails_when_score_below_threshold():
    import agent
    reload(agent)
    mock_client = MagicMock()
    low_scores = {k: 0 for k in ["metrics", "segment", "requirements", "out_of_scope",
                                   "open_questions", "no_fluff", "jtbd", "business_metric"]}
    issues = ["Нет JTBD", "Метрики расплывчатые", "Out of scope пустой"]
    mock_client.chat.completions.create.return_value = make_critique_response(low_scores, issues)

    with patch("agent.OpenAI", return_value=mock_client):
        result = agent.critique(base_state(prd="# Плохой PRD"))

    assert result["critique_passed"] is False
    assert result["critique_score"] == 0
    assert len(result["critique_issues"]) == 3


def test_critique_saves_prev_score():
    import agent
    reload(agent)
    mock_client = MagicMock()
    scores = {k: 1 for k in ["metrics", "segment", "requirements", "out_of_scope",
                               "open_questions", "no_fluff", "jtbd", "business_metric"]}
    mock_client.chat.completions.create.return_value = make_critique_response(scores, ["замечание"])

    with patch("agent.OpenAI", return_value=mock_client):
        result = agent.critique(base_state(prd="# PRD", critique_score=5))

    assert result["prev_critique_score"] == 5


# --- route_after_critique ---

def test_route_saves_when_passed():
    import agent
    result = agent.route_after_critique(base_state(critique_passed=True, critique_score=12))
    assert result == "save"


def test_route_regenerates_when_score_improves():
    import agent
    state = base_state(critique_passed=False, critique_score=8, prev_critique_score=5)
    result = agent.route_after_critique(state)
    assert result == "generate"


def test_route_saves_when_no_progress():
    import agent
    state = base_state(critique_passed=False, critique_score=7, prev_critique_score=7)
    result = agent.route_after_critique(state)
    assert result == "save"


# --- save ---

def test_save_node(tmp_path):
    import agent
    reload(agent)
    fake_path = str(tmp_path / "20260430_виш_лист.md")

    with patch("agent.save_prd", return_value=fake_path):
        result = agent.save(base_state(prd="# PRD содержимое"))

    assert result["output_path"] == fake_path


# --- full graph ---

def test_full_graph_skip(tmp_path):
    import agent
    reload(agent)
    import uuid
    from langgraph.types import Command

    mock_client = MagicMock()
    high_scores = {k: 2 for k in ["metrics", "segment", "requirements", "out_of_scope",
                                    "open_questions", "no_fluff", "jtbd", "business_metric"]}
    # LangGraph переигрывает ask_questions при resume — итого 4 вызова:
    # ask_questions (первый invoke) → ask_questions (второй invoke, replay)
    # → generate → critique
    responses = [
        make_llm_response("Вопрос 1\nВопрос 2"),  # ask_questions, первый invoke
        make_llm_response("Вопрос 1\nВопрос 2"),  # ask_questions, replay при resume
        make_llm_response("# Финальный PRD"),       # generate
        make_critique_response(high_scores, []),    # critique
    ]
    call_count = {"n": 0}

    def side_effect(*args, **kwargs):
        r = responses[min(call_count["n"], len(responses) - 1)]
        call_count["n"] += 1
        return r

    mock_client.chat.completions.create.side_effect = side_effect

    fake_path = str(tmp_path / "output.md")
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}

    initial_state = base_state(feature_description="добавить виш-лист")

    with patch("agent.rag_search", return_value=[]), \
         patch("agent.OpenAI", return_value=mock_client), \
         patch("agent.save_prd", return_value=fake_path):

        graph = agent.build_graph()
        graph.invoke(initial_state, config=config)
        result = graph.invoke(Command(resume="/skip"), config=config)

    assert result["prd"] == "# Финальный PRD"
    assert result["output_path"] == fake_path
    assert result["skipped"] is True
    assert result["critique_passed"] is True
