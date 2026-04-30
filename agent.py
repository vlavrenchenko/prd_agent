"""
LangGraph агент для генерации PRD.

Запуск:
    python agent.py "добавить виш-лист в мобильное приложение"
"""
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


class AgentState(TypedDict):
    feature_description: str
    rag_context: list[dict]
    questions: list[str]
    answers: str
    skipped: bool
    prd: str
    output_path: str


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

    # Граф приостанавливается здесь и ждёт ввода от пользователя.
    # user_response — это то, что будет передано через Command(resume=...).
    user_response = interrupt({"questions": questions})

    if str(user_response).strip().lower() == "/skip":
        return {"questions": questions, "skipped": True, "answers": ""}

    return {"questions": questions, "skipped": False, "answers": str(user_response)}


def generate(state: AgentState) -> dict:
    """Генерирует PRD по шаблону с учётом контекста и ответов PM."""
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

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "Ты опытный продуктовый менеджер. "
                    "Напиши полный PRD на русском языке строго следуя структуре шаблона. "
                    "Все разделы заполняй конкретно и профессионально. "
                    "Метрики должны быть измеримыми. "
                    f"{status_note}"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Шаблон PRD:\n{template}\n\n"
                    f"Описание фичи: {state['feature_description']}"
                    f"{answers_block}\n\n"
                    f"Похожие PRD для контекста:\n{context_preview}"
                ),
            },
        ],
    )

    return {"prd": response.choices[0].message.content}


def save(state: AgentState) -> dict:
    """Сохраняет PRD в файл и индексирует в ChromaDB."""
    path = save_prd(state["prd"], state["feature_description"])
    return {"output_path": path}


# --- Сборка графа ---

def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("search_context", search_context)
    graph.add_node("ask_questions", ask_questions)
    graph.add_node("generate", generate)
    graph.add_node("save", save)

    graph.add_edge(START, "search_context")
    graph.add_edge("search_context", "ask_questions")
    graph.add_edge("ask_questions", "generate")
    graph.add_edge("generate", "save")
    graph.add_edge("save", END)

    return graph.compile(checkpointer=MemorySaver())


# --- CLI ---

def run_cli(description: str):
    graph = build_graph()
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}

    initial_state = {
        "feature_description": description,
        "rag_context": [],
        "questions": [],
        "answers": "",
        "skipped": False,
        "prd": "",
        "output_path": "",
    }

    print(f'\n🔍 Ищем похожие PRD для: "{description}"\n')

    # Первый запуск — граф остановится на interrupt внутри ask_questions
    graph.invoke(initial_state, config=config)

    snapshot = graph.get_state(config)

    if snapshot.next:
        # Извлекаем вопросы из interrupt
        interrupts = snapshot.tasks[0].interrupts if snapshot.tasks else []
        questions = interrupts[0].value.get("questions", []) if interrupts else []

        print("❓ Уточняющие вопросы:")
        for q in questions:
            print(f"   • {q}")
        print("\nОтветь на вопросы одним сообщением или напиши /skip:\n")

        user_input = input("➤ ").strip()

        # Возобновляем граф с ответом пользователя
        result = graph.invoke(Command(resume=user_input), config=config)
    else:
        result = graph.get_state(config).values

    output_path = result.get("output_path", "")
    prd_text = result.get("prd", "")

    print(f"\n{'='*55}")
    print(prd_text)
    print(f"{'='*55}")
    print(f"\n✅ PRD сохранён: {output_path}\n")


def main():
    if len(sys.argv) < 2:
        print("Использование: python agent.py \"описание фичи\"")
        print('Пример: python agent.py "добавить виш-лист в мобильное приложение"')
        sys.exit(1)

    run_cli(" ".join(sys.argv[1:]))


if __name__ == "__main__":
    main()
