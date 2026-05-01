"""
Telegram-бот для генерации PRD.

Запуск:
    python bot.py
"""
import asyncio
import logging
import os

from aiogram import Bot, Dispatcher, F
from aiogram.filters import Command, CommandStart
from aiogram.types import Message, FSInputFile
from dotenv import load_dotenv
from langgraph.types import Command as LGCommand

from agent import build_graph, AgentState, critique_prd
from search import search as rag_search
import uuid

load_dotenv(override=True)

logging.basicConfig(level=logging.WARNING)

dp = Dispatcher()
graph = build_graph()

# chat_id → thread_id активных сессий генерации
active_sessions: dict[int, str] = {}

# chat_id → список результатов последнего поиска
search_cache: dict[int, list[dict]] = {}

# инициализируется в main()
bot: Bot = None  # type: ignore


def _config(thread_id: str) -> dict:
    return {"configurable": {"thread_id": thread_id}}


def _get_questions(graph, config: dict) -> list[str]:
    """Извлекает вопросы из interrupt в текущем снэпшоте графа."""
    snapshot = graph.get_state(config)
    if not snapshot.next or not snapshot.tasks:
        return []
    interrupts = snapshot.tasks[0].interrupts
    if not interrupts:
        return []
    return interrupts[0].value.get("questions", [])


def _format_questions(questions: list[str]) -> str:
    lines = ["❓ *Уточняющие вопросы:*\n"]
    for q in questions:
        lines.append(f"• {q}")
    lines.append("\nОтветь одним сообщением или напиши /skip чтобы пропустить.")
    return "\n".join(lines)


@dp.message(CommandStart())
async def cmd_start(message: Message):
    await message.answer(
        "Привет! Я помогу написать и оценить PRD.\n\n"
        "*Создать PRD:*\n"
        "`/new <описание фичи>` — запустить генерацию\n"
        "`/skip` — пропустить уточняющие вопросы\n\n"
        "*Поиск и критика:*\n"
        "`/search <запрос>` — найти PRD в базе\n"
        "`/critique <номер>` — оценить PRD из результатов поиска\n\n"
        "Пример:\n"
        "`/new добавить виш-лист`\n"
        "`/search онбординг пользователя`",
        parse_mode="Markdown",
    )


@dp.message(Command("new"))
async def cmd_new(message: Message):
    description = message.text.removeprefix("/new").strip()
    if not description:
        await message.answer("Напиши описание фичи после команды.\nПример: `/new добавить виш-лист`", parse_mode="Markdown")
        return

    chat_id = message.chat.id
    thread_id = str(uuid.uuid4())
    active_sessions[chat_id] = thread_id
    config = _config(thread_id)

    initial_state: AgentState = {
        "feature_description": description,
        "rag_context": [],
        "questions": [],
        "answers": "",
        "skipped": False,
        "prd": "",
        "output_path": "",
    }

    await message.answer(f"🔍 Ищем похожие PRD для: _{description}_", parse_mode="Markdown")

    await asyncio.get_event_loop().run_in_executor(
        None, lambda: graph.invoke(initial_state, config)
    )

    questions = _get_questions(graph, config)
    if questions:
        await message.answer(_format_questions(questions), parse_mode="Markdown")
    else:
        await _finish_session(message, config, chat_id)


@dp.message(Command("skip"))
async def cmd_skip(message: Message):
    chat_id = message.chat.id
    if chat_id not in active_sessions:
        await message.answer("Нет активной сессии. Начни с `/new <описание фичи>`.", parse_mode="Markdown")
        return

    config = _config(active_sessions[chat_id])
    await message.answer("⏭ Пропускаем вопросы, генерируем PRD...")
    await _resume_session(message, config, chat_id, "/skip")


@dp.message(Command("search"))
async def cmd_search(message: Message):
    query = message.text.removeprefix("/search").strip()
    if not query:
        await message.answer("Напиши запрос после команды.\nПример: `/search онбординг пользователя`", parse_mode="Markdown")
        return

    await message.answer(f"🔍 Ищем: _{query}_...", parse_mode="Markdown")

    results = await asyncio.get_event_loop().run_in_executor(
        None, lambda: rag_search(query, n=5)
    )

    if not results:
        await message.answer("Ничего не найдено. Попробуй другой запрос.")
        return

    search_cache[message.chat.id] = results

    lines = ["📋 *Найденные PRD:*\n"]
    for r in results:
        lines.append(f"{r['rank']}. `{r['filename']}`  (релевантность: {r['score']})")
    lines.append("\nЧтобы покритиковать — напиши `/critique <номер>`")
    await message.answer("\n".join(lines), parse_mode="Markdown")


@dp.message(Command("critique"))
async def cmd_critique(message: Message):
    chat_id = message.chat.id
    arg = message.text.removeprefix("/critique").strip()

    if not arg.isdigit():
        await message.answer("Напиши номер PRD из последнего поиска.\nПример: `/critique 1`", parse_mode="Markdown")
        return

    idx = int(arg) - 1
    results = search_cache.get(chat_id, [])

    if not results:
        await message.answer("Сначала выполни поиск: `/search <запрос>`", parse_mode="Markdown")
        return

    if idx < 0 or idx >= len(results):
        await message.answer(f"Номер должен быть от 1 до {len(results)}.")
        return

    selected = results[idx]
    await message.answer(f"🔎 Критикуем: `{selected['filename']}`...", parse_mode="Markdown")

    critique_result = await asyncio.get_event_loop().run_in_executor(
        None, lambda: critique_prd(selected["text"])
    )

    score = critique_result["score"]
    max_score = critique_result["max_score"]
    threshold = critique_result["threshold"]
    passed = critique_result["passed"]
    issues = critique_result["issues"]
    scores = critique_result["scores"]

    criteria_labels = {
        "metrics": "Метрики измеримы",
        "segment": "Сегмент пользователей",
        "requirements": "Функц. требования",
        "out_of_scope": "Out of scope",
        "open_questions": "Открытые вопросы",
        "no_fluff": "Отсутствие воды",
        "jtbd": "JTBD описан",
        "business_metric": "Бизнес-метрика",
    }

    score_lines = [f"📊 *Оценка PRD: {score}/{max_score}* (порог: {threshold})\n"]
    score_lines.append("*Детали:*")
    for key, label in criteria_labels.items():
        val = scores.get(key, 0)
        emoji = "✅" if val == 2 else "⚠️" if val == 1 else "❌"
        score_lines.append(f"{emoji} {label}: {val}/2")

    if passed:
        score_lines.append("\n✅ PRD прошёл проверку качества")
    else:
        score_lines.append("\n*Замечания:*")
        for issue in issues:
            score_lines.append(f"• {issue}")

    await message.answer("\n".join(score_lines), parse_mode="Markdown")


@dp.message(F.text & ~F.text.startswith("/"))
async def on_message(message: Message):
    chat_id = message.chat.id
    if chat_id not in active_sessions:
        await message.answer("Начни с `/new <описание фичи>`.", parse_mode="Markdown")
        return

    config = _config(active_sessions[chat_id])
    await message.answer("✍️ Генерируем PRD...")
    await _resume_session(message, config, chat_id, message.text)


async def _resume_session(message: Message, config: dict, chat_id: int, user_input: str):
    """Возобновляет граф с ответом пользователя и завершает сессию."""
    result = await asyncio.get_event_loop().run_in_executor(
        None, lambda: graph.invoke(LGCommand(resume=user_input), config)
    )
    await _finish_session(message, config, chat_id, result)


async def _finish_session(message: Message, config: dict, chat_id: int, result: dict = None):
    """Отправляет готовый PRD с оценкой критика и закрывает сессию."""
    import json
    from pathlib import Path

    if result is None:
        result = graph.get_state(config).values

    output_path = result.get("output_path", "")
    active_sessions.pop(chat_id, None)

    if not output_path:
        await message.answer("❌ Не удалось сгенерировать PRD.")
        return

    criteria_path = Path(__file__).parent / "config" / "critique_criteria.json"
    criteria_config = json.loads(criteria_path.read_text(encoding="utf-8"))
    max_score = criteria_config["max_score"]
    threshold = criteria_config["threshold"]

    score = result.get("critique_score", 0)
    passed = result.get("critique_passed", False)
    issues = result.get("critique_issues", [])

    if passed:
        quality_text = f"📊 Оценка PRD: *{score}/{max_score}* ✅"
    else:
        quality_text = f"📊 Оценка PRD: *{score}/{max_score}* (порог: {threshold})\n⚠️ Замечания:\n"
        quality_text += "\n".join(f"• {issue}" for issue in issues)

    await message.answer(quality_text, parse_mode="Markdown")
    await message.answer_document(
        FSInputFile(output_path),
        caption="Готовый PRD в формате Markdown",
    )


async def main():
    global bot
    assert os.environ.get("TELEGRAM_BOT_TOKEN"), "Задайте TELEGRAM_BOT_TOKEN в .env файле"
    bot = Bot(token=os.environ["TELEGRAM_BOT_TOKEN"])
    print("🤖 Бот запущен. Нажми Ctrl+C для остановки.")
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
