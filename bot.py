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

from agent import build_graph, AgentState
import uuid

load_dotenv(override=True)

logging.basicConfig(level=logging.WARNING)

dp = Dispatcher()
graph = build_graph()

# chat_id → thread_id активных сессий
active_sessions: dict[int, str] = {}

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
        "Привет! Я помогу написать PRD.\n\n"
        "Напиши `/new <описание фичи>` чтобы начать.\n\n"
        "Пример:\n`/new добавить виш-лист в мобильное приложение`",
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
