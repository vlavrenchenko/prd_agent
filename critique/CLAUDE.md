# Critique Agent

Пакет для оценки качества PRD по 8 критериям с объяснениями.

## Структура

```
critique/
├── core.py         # LLM-логика: critique_prd(), critique() (LangGraph-узел)
├── formatter.py    # Форматирование для Telegram-бота
├── criteria.json   # Конфигурация критериев и порогов
└── CLAUDE.md
```

## Публичный API

```python
from critique import critique_prd, critique, load_criteria
from critique.formatter import detect_criterion, format_explanation, format_all_non_perfect
```

### `critique_prd(prd_text: str) -> dict`
Standalone-функция для оценки готового PRD из бота (`/critique <номер>`).

Возвращает:
```python
{
    "score": int,           # итоговый балл (0–16)
    "max_score": int,       # 16
    "threshold": int,       # 11
    "passed": bool,
    "issues": list[str],    # конкретные замечания для улучшения
    "scores": dict,         # балл по каждому критерию
    "explanations": dict,   # почему поставил именно эту оценку
}
```

### `critique(state: dict) -> dict`
LangGraph-узел в основном графе агента. Вызывается после `generate`, перед `save`.

### `load_criteria() -> dict`
Загружает `criteria.json` — порог, максимум, описание критериев.

## Критерии оценки

| ID               | Название               | Макс |
|------------------|------------------------|------|
| metrics          | Метрики измеримы       | 2    |
| segment          | Сегмент пользователей  | 2    |
| requirements     | Функц. требования      | 2    |
| out_of_scope     | Out of scope           | 2    |
| open_questions   | Открытые вопросы       | 2    |
| no_fluff         | Отсутствие воды        | 2    |
| jtbd             | JTBD описан            | 2    |
| business_metric  | Бизнес-метрика         | 2    |

Порог прохождения: **11/16**. Если PRD не прошёл и улучшение < 1 балла — граф сохраняет как есть.

## Зависимости

- `llm_utils.log_cost` — логирование стоимости LLM-вызовов
- `logger.get_logger("critique")` — структурированные логи
- `openai.OpenAI` — LLM-вызовы через gpt-4o-mini

## Развитие фичи

- Изменить промпт или JSON-схему → `core.py` (`_CRITIQUE_SYSTEM_PROMPT`, `_CRITIQUE_JSON_SCHEMA`)
- Добавить/изменить критерий → `criteria.json` + обновить словари в `formatter.py`
- Изменить форматирование ответа в боте → `formatter.py`
- Добавить версионирование промптов → новый файл `critique/prompts.py`
