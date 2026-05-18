# CLAUDE.md — PRD Agent

## Что это

Telegram-бот для продуктовых менеджеров. Генерирует PRD по описанию фичи,
оценивает качество по 8 критериям, сохраняет в базу знаний для поиска.

## Стек

- **Python 3.14**, venv в `../.venv/` (родительский каталог)
- **LangGraph** — граф состояний агента с interrupt/resume и checkpointing
- **ChromaDB** — векторное хранилище; два возможных режима: in-memory и файловый (`data/chroma_db/`)
- **OpenAI API** — gpt-4o-mini (генерация, критика), text-embedding-3-small (эмбеддинги)
- **aiogram 3** — Telegram-бот

Запуск: `source ../.venv/bin/activate && python bot.py`

## Архитектура

### Граф агента (`agent.py`)

```
START → search_context → ask_questions → generate → critique
                                                         │
                                          passed / delta < 1 → save → END
                                          not passed / delta ≥ 1 → generate (retry)
```

- `search_context` — ChromaDB поиск по `feature_description`, n=3
- `ask_questions` — LLM генерирует 3–5 вопросов, `interrupt()` ждёт ответа PM
- `generate` — LLM пишет PRD по шаблону (`config/prd_template.md`); при повторе учитывает `critique_issues`
- `critique` — вызывает `critique.core.critique()`, обновляет `critique_score/passed/issues`
- `save` — сохраняет `.md` в `data/prd_output/` и индексирует в ChromaDB

Ветвление после critique: если PRD прошёл порог (≥11/16) или прирост < 1 балла → `save`;
иначе → `generate` ещё раз.

### State (`AgentState`)

```python
feature_description: str
rag_context: list[dict]       # [{filename, text, score, rank}]
questions: list[str]
answers: str
skipped: bool
prd: str
output_path: str
critique_issues: list[str]
critique_score: int
prev_critique_score: int
critique_passed: bool
```

### Critique-пакет (`critique/`)

Публичный API: `from critique import critique_prd, critique, load_criteria`

- `critique_prd(prd_text)` — standalone вызов из бота (`/critique <номер>`)
- `critique(state)` — LangGraph-узел
- `load_criteria()` — читает `criteria.json` (порог, максимум, критерии)

Конфиг критериев — только в `criteria.json`. Промпт и JSON-схема — в `core.py`.
Ключевые слова для определения критерия по тексту — в `formatter.py` (`CRITERION_KEYWORDS`).

### Бот (`bot.py`)

Состояние бота — три in-memory словаря:
- `active_sessions: dict[int, str]` — `chat_id → thread_id` (текущая генерация)
- `search_cache: dict[int, list[dict]]` — последние результаты `/search`
- `critique_cache: dict[int, dict]` — последний результат `/critique` (для follow-up вопросов)

Порядок хэндлеров важен. `on_message` (текст без `/`) проверяет:
1. Есть ли `active_sessions[chat_id]` → resume графа
2. Есть ли `critique_cache[chat_id]` → detect_criterion → explain

`cmd_unknown` стоит последним среди slash-хэндлеров — catch-all для неизвестных команд.

## Ключевые файлы

| Файл | Роль |
|---|---|
| `agent.py` | Граф, `AgentState`, узлы, `build_graph()`, CLI |
| `bot.py` | aiogram хэндлеры, сессии, форматирование ответов |
| `critique/core.py` | LLM-логика критика, `_run_critique_llm()` |
| `critique/formatter.py` | `detect_criterion()`, `format_explanation()`, `format_all_non_perfect()` |
| `critique/criteria.json` | Порог (11), максимум (16), описание уровней по каждому критерию |
| `search.py` | `search(query, n)` → ChromaDB; `--index` индексирует `data/prd_synthetic/` |
| `save.py` | `save_prd(text, feature)` → файл + ChromaDB индексация |
| `logger.py` | `get_logger(name)` — JSON-логи в `logs/`; `get_cost_logger()` → `logs/costs.log` |
| `llm_utils.py` | `log_cost(operation, model, response)` — пишет в costs.log |
| `costs_report.py` | `python costs_report.py --period day/week/month/all` |
| `generate_synthetic.py` | Генерация синтетических PRD для RAG-базы |
| `config/prd_template.md` | Шаблон PRD (9 секций); агент пишет строго по нему |
| `config/models_pricing.json` | Цены OpenAI моделей для расчёта cost_usd |

## Переменные окружения

`.env` в корне проекта:
```
OPENAI_API_KEY=...
TELEGRAM_BOT_TOKEN=...
```

## Тесты

```bash
source ../.venv/bin/activate

# Все unit-тесты (81 тест)
pytest tests/ --ignore=tests/e2e

# E2e (требуют реальные API-ключи)
pytest tests/e2e/ -v
```

Unit-тесты не вызывают реальные API. OpenAI и ChromaDB замоканы через `unittest.mock`.
`conftest.py` — `suppress_cost_logging` автоматически глушит запись в costs.log.

Именование тестов: `test_<module>_<behaviour>_<condition>`.

## Логирование

```python
from logger import get_logger
log = get_logger("module_name")

log.info("event_name", extra={"key": "value"})  # DEBUG/INFO → только файл
log.error("event_name", extra={"key": "value"})  # ERROR → файл + терминал
```

Никогда не использовать в `extra={}` зарезервированные поля: `args`, `message`, `msg`, `name`.
Вместо `args` использовать `tool_args`.

## Статус разработки

| Шаг | Статус |
|---|---|
| 1. Шаблон и синтетика | ✅ Done |
| 2. RAG (ChromaDB) | ✅ Done |
| 3. LangGraph агент | ✅ Done |
| 4. Файловый экспорт + RAG-индексация | ✅ Done |
| 5. Telegram-бот (/new /skip /search /critique) | ✅ Done |
| 6. Персистентная память (SqliteSaver) | 🔲 Backlog |
| 7. Критик PRD (8 критериев, объяснения, follow-up) | ✅ Done |
| 8. База болей пользователей | 🔲 Backlog |
| 9. Брейншторм-помощник (/brainstorm) | 🔲 Backlog |

Детали по открытым задачам — в `BACKLOG.md`.
