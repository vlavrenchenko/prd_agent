"""
LangGraph агент для генерации PRD с критиком.

Запуск:
    python agent.py "добавить виш-лист в мобильное приложение"
"""
import json
import sys
import uuid
from pathlib import Path
from typing import TypedDict

from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt, Command
from openai import OpenAI

from search import search as rag_search
from save import save_prd

load_dotenv(override=True)

TEMPLATE_PATH = Path(__file__).parent / "config" / "prd_template.md"
CRITERIA_PATH = Path(__file__).parent / "config" / "critique_criteria.json"


def _load_criteria() -> dict:
    return json.loads(CRITERIA_PATH.read_text(encoding="utf-8"))


class AgentState(TypedDict):
    feature_description: str
    rag_context: list[dict]
    questions: list[str]
    answers: str
    skipped: bool
    prd: str
    output_path: str
    critique_issues: list[str]
    critique_score: int
    prev_critique_score: int
    critique_passed: bool


# --- Узлы графа ---

def search_context(state: AgentState) -> dict:
    """Ищет похожие PRD в ChromaDB."""
    results = rag_search(state["feature_description"], n=3)
    return {"rag_context": results}


def ask_questions(state: AgentState) -> dict:
    """Генерирует уточняющие вопросы и ждёт ответа от PM."""
    client = OpenAI()

    context_preview = "\n\n".join(
        f"[{r['filename']}]\n{r['text'][:400]}"
        for r in state["rag_context"]
    )

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "Ты опытный продуктовый менеджер. "
                    "На основе описания фичи задай 3-5 коротких уточняющих вопроса, "
                    "которые помогут написать качественный PRD. "
                    "Обязательно спроси про JTBD пользователя и ключевую бизнес-метрику. "
                    "Верни только список вопросов — по одному на строку, без нумерации и маркеров."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Фича: {state['feature_description']}\n\n"
                    f"Похожие PRD из базы:\n{context_preview}"
                ),
            },
        ],
    )

    questions = [
        q.strip()
        for q in response.choices[0].message.content.strip().splitlines()
        if q.strip()
    ]

    user_response = interrupt({"questions": questions})

    if str(user_response).strip().lower() == "/skip":
        return {"questions": questions, "skipped": True, "answers": ""}

    return {"questions": questions, "skipped": False, "answers": str(user_response)}


def generate(state: AgentState) -> dict:
    """Генерирует PRD по шаблону. При повторной генерации учитывает замечания критика."""
    client = OpenAI()
    template = TEMPLATE_PATH.read_text(encoding="utf-8")

    context_preview = "\n\n".join(
        f"[{r['filename']}]\n{r['text'][:600]}"
        for r in state["rag_context"]
    )

    if state["skipped"]:
        status_note = 'Статус документа: "Draft (требует доработки)" — пользователь пропустил уточняющие вопросы.'
        answers_block = ""
    else:
        status_note = 'Статус документа: "Draft".'
        questions_text = "\n".join(f"- {q}" for q in state["questions"])
        answers_block = (
            f"\n\nУточняющие вопросы:\n{questions_text}"
            f"\n\nОтветы PM:\n{state['answers']}"
        )

    critique_block = ""
    if state.get("critique_issues"):
        issues_text = "\n".join(f"- {issue}" for issue in state["critique_issues"])
        critique_block = (
            f"\n\nПредыдущая версия PRD получила оценку {state['critique_score']}/16. "
            f"Исправь следующие замечания:\n{issues_text}"
        )

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "Ты опытный продуктовый менеджер. "
                    "Напиши полный PRD на русском языке строго следуя структуре шаблона. "
                    "Все разделы заполняй конкретно и профессионально. "
                    "Метрики должны быть измеримыми с цифрами и сроками. "
                    "Обязательно включи JTBD в формате: 'Когда [ситуация], я хочу [действие], чтобы [результат]'. "
                    "Обязательно укажи ключевую бизнес-метрику с текущим и целевым значением. "
                    f"{status_note}"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Шаблон PRD:\n{template}\n\n"
                    f"Описание фичи: {state['feature_description']}"
                    f"{answers_block}"
                    f"{critique_block}\n\n"
                    f"Похожие PRD для контекста:\n{context_preview}"
                ),
            },
        ],
    )

    return {"prd": response.choices[0].message.content}


def critique(state: AgentState) -> dict:
    """Оценивает качество PRD по 8 критериям и возвращает замечания."""
    client = OpenAI()
    criteria_config = _load_criteria()

    criteria_text = "\n".join(
        f"{i+1}. {c['name']} (0-{c['max']} баллов):\n"
        + "\n".join(f"   {score}: {desc}" for score, desc in c["levels"].items())
        for i, c in enumerate(criteria_config["criteria"])
    )

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": (
                    "Ты строгий рецензент PRD. "
                    "Оцени документ по каждому критерию и верни результат строго в JSON. "
                    "Будь конкретен в замечаниях — указывай какой раздел и что именно нужно исправить."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Оцени PRD по следующим критериям:\n\n{criteria_text}\n\n"
                    f"PRD для оценки:\n{state['prd']}\n\n"
                    "Верни JSON:\n"
                    "{\n"
                    '  "scores": {"metrics": 0, "segment": 0, "requirements": 0, "out_of_scope": 0, '
                    '"open_questions": 0, "no_fluff": 0, "jtbd": 0, "business_metric": 0},\n'
                    '  "issues": ["конкретное замечание 1", "конкретное замечание 2"]\n'
                    "}"
                ),
            },
        ],
    )

    result = json.loads(response.choices[0].message.content)
    scores = result.get("scores", {})
    total_score = sum(scores.values())
    issues = result.get("issues", [])

    threshold = criteria_config["threshold"]
    passed = total_score >= threshold

    return {
        "critique_score": total_score,
        "prev_critique_score": state.get("critique_score", 0),
        "critique_issues": issues if not passed else [],
        "critique_passed": passed,
    }


def save(state: AgentState) -> dict:
    """Сохраняет PRD в файл и индексирует в ChromaDB."""
    path = save_prd(state["prd"], state["feature_description"])
    return {"output_path": path}


# --- Условное ветвление ---

def route_after_critique(state: AgentState) -> str:
    if state["critique_passed"]:
        return "save"
    delta = state["critique_score"] - state.get("prev_critique_score", 0)
    if delta < 1:
        return "save"
    return "generate"


# --- Сборка графа ---

def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("search_context", search_context)
    graph.add_node("ask_questions", ask_questions)
    graph.add_node("generate", generate)
    graph.add_node("critique", critique)
    graph.add_node("save", save)

    graph.add_edge(START, "search_context")
    graph.add_edge("search_context", "ask_questions")
    graph.add_edge("ask_questions", "generate")
    graph.add_edge("generate", "critique")
    graph.add_conditional_edges(
        "critique",
        route_after_critique,
        {"save": "save", "generate": "generate"},
    )
    graph.add_edge("save", END)

    return graph.compile(checkpointer=MemorySaver())


# --- CLI ---

def run_cli(description: str):
    graph = build_graph()
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}

    initial_state: AgentState = {
        "feature_description": description,
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

    criteria_config = _load_criteria()
    print(f'\n🔍 Ищем похожие PRD для: "{description}"\n')

    graph.invoke(initial_state, config=config)

    snapshot = graph.get_state(config)

    if snapshot.next:
        interrupts = snapshot.tasks[0].interrupts if snapshot.tasks else []
        questions = interrupts[0].value.get("questions", []) if interrupts else []

        print("❓ Уточняющие вопросы:")
        for q in questions:
            print(f"   • {q}")
        print("\nОтветь на вопросы одним сообщением или напиши /skip:\n")

        user_input = input("➤ ").strip()
        result = graph.invoke(Command(resume=user_input), config=config)
    else:
        result = graph.get_state(config).values

    score = result.get("critique_score", 0)
    passed = result.get("critique_passed", False)
    issues = result.get("critique_issues", [])
    max_score = criteria_config["max_score"]
    threshold = criteria_config["threshold"]

    print(f"\n{'='*55}")
    print(result["prd"])
    print(f"{'='*55}")
    print(f"\n📊 Оценка PRD: {score}/{max_score} (порог: {threshold})")
    if passed:
        print("✅ PRD прошёл проверку качества")
    else:
        print("⚠️  PRD сохранён с замечаниями:")
        for issue in issues:
            print(f"   • {issue}")
    print(f"\n💾 PRD сохранён: {result['output_path']}\n")


def main():
    if len(sys.argv) < 2:
        print("Использование: python agent.py \"описание фичи\"")
        print('Пример: python agent.py "добавить виш-лист в мобильное приложение"')
        sys.exit(1)

    run_cli(" ".join(sys.argv[1:]))


if __name__ == "__main__":
    main()
