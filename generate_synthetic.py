"""
Генерирует синтетические PRD для наполнения RAG-базы.

Запуск:
    python generate_synthetic.py --count 50
    python generate_synthetic.py --count 10 --domain fintech
    python generate_synthetic.py --estimate          # оценить стоимость без генерации
"""
import json
import time
import argparse
import datetime
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv
from logger import get_cost_logger

load_dotenv(override=True)

OUTPUT_DIR = Path(__file__).parent / "data" / "prd_synthetic"
TEMPLATE_PATH = Path(__file__).parent / "config" / "prd_template.md"
PRICING_PATH = Path(__file__).parent / "config" / "models_pricing.json"

DOMAINS = [
    "e-commerce",
    "fintech",
    "edtech",
    "healthtech",
    "b2b saas",
    "marketplace",
    "social",
    "productivity",
    "logistics",
    "hr-tech",
]

FEATURE_IDEAS = {
    "e-commerce": [
        "умный поиск с автодополнением", "программа лояльности", "one-click checkout",
        "персонализированные рекомендации", "сравнение товаров", "виш-лист",
        "уведомления о снижении цены", "отслеживание заказа в реальном времени",
    ],
    "fintech": [
        "аналитика расходов по категориям", "автоматические сбережения", "P2P переводы",
        "кредитный скоринг", "встроенный кошелёк", "мгновенные уведомления о транзакциях",
        "split bill для группы", "инвестиционный портфель",
    ],
    "edtech": [
        "адаптивный план обучения", "геймификация курсов", "групповые проекты онлайн",
        "AI-репетитор", "сертификаты прохождения", "прогресс-трекер",
        "live-сессии с экспертами", "практические задания с автопроверкой",
    ],
    "healthtech": [
        "дневник симптомов", "напоминания о приёме лекарств", "телемедицина-чат",
        "интеграция с носимыми устройствами", "персональный план питания",
        "трекер физической активности", "запись к врачу онлайн", "история анализов",
    ],
    "b2b saas": [
        "командные рабочие пространства", "ролевая модель доступа", "аудит-лог действий",
        "webhook-интеграции", "SSO авторизация", "кастомные отчёты",
        "API rate limiting", "мультитенантность",
    ],
    "marketplace": [
        "верификация продавцов", "система отзывов с фото", "escrow-платежи",
        "диспут-резолюция", "умная выдача объявлений", "подписка для продавцов",
        "push-уведомления о новых предложениях", "чат покупатель-продавец",
    ],
    "social": [
        "алгоритмическая лента", "сторис с реакциями", "закрытые группы по интересам",
        "монетизация контента", "live-стримы", "коллаборативные плейлисты",
        "система репостов с комментарием", "упоминания и теги",
    ],
    "productivity": [
        "AI-суммаризация встреч", "умный calendar scheduling", "шаблоны задач",
        "интеграция с почтой", "time tracking", "OKR-трекер",
        "автоматические напоминания", "командный дашборд",
    ],
    "logistics": [
        "маршрутизация доставки", "уведомления получателю", "подтверждение доставки с фото",
        "возвраты и рекламации", "склад-менеджмент", "интеграция с ТК-партнёрами",
        "предсказание сроков доставки", "бесконтактная доставка",
    ],
    "hr-tech": [
        "онбординг нового сотрудника", "система ревью 360", "трекер отпусков",
        "база знаний компании", "анонимный feedback", "рекрутинговая воронка",
        "планирование обучения", "пульс-опросы команды",
    ],
}

SYSTEM_PROMPT = """Ты опытный продуктовый менеджер. Ты пишешь профессиональные PRD (Product Requirements Document) на русском языке.

Структура PRD строго фиксирована. Ты заполняешь каждый раздел осмысленно, конкретно, без воды.

Требования к тексту:
- Пиши профессионально, как реальный PM в продуктовой команде
- Метрики должны быть измеримыми (не "улучшить", а "+15% конверсии")
- User stories в формате: "Как [роль], я хочу [действие], чтобы [результат]"
- Функциональные требования — нумерованный список конкретных требований к системе
- Открытые вопросы — реальные, нерешённые вопросы, а не заглушки
- Зависимости — реальные технические или командные зависимости"""


def load_model_prices() -> dict:
    if not PRICING_PATH.exists():
        return {}
    raw = json.loads(PRICING_PATH.read_text())
    prices = {}
    for model_id, model_data in raw.get("models", {}).items():
        standard = model_data.get("pricing", {}).get("standard", {})
        if "input" in standard:
            input_price = standard["input"]
            output_price = standard["output"]
        elif "short_context" in standard:
            input_price = standard["short_context"]["input"]
            output_price = standard["short_context"]["output"]
        else:
            continue
        if input_price is not None and output_price is not None:
            prices[model_id] = {"input": input_price, "output": output_price}
    return prices


MODEL_PRICES = load_model_prices()
_cost_log = get_cost_logger()


def build_user_prompt(domain: str, feature: str) -> str:
    today = datetime.date.today().isoformat()
    return f"""Напиши PRD для следующей фичи.

Домен: {domain}
Фича: {feature}
Дата: {today}

Верни ТОЛЬКО валидный JSON со следующими полями (все значения на русском языке):
{{
  "title": "название фичи",
  "status": "Draft",
  "author": "PM",
  "date": "{today}",
  "version": "1.0",
  "problem": "описание проблемы (2-4 предложения)",
  "goal": "цель и измеримый результат (2-3 предложения)",
  "users": "описание целевых пользователей (2-3 предложения)",
  "user_stories": "3-4 user story в формате маркированного списка",
  "functional_requirements": "5-8 конкретных функциональных требований нумерованным списком",
  "out_of_scope": "3-5 пунктов что НЕ входит в эту версию",
  "success_metrics": "3-4 измеримые метрики успеха",
  "open_questions": "2-4 реальных открытых вопроса",
  "dependencies": "2-3 зависимости от других команд или систем"
}}"""


def _to_markdown(value) -> str:
    if isinstance(value, list):
        return "\n".join(f"- {item}" for item in value)
    return str(value)


def render_prd(template: str, fields: dict) -> str:
    result = template
    for key, value in fields.items():
        result = result.replace("{" + key + "}", _to_markdown(value))
    return result


def generate_one(client: OpenAI, domain: str, feature: str, model: str) -> tuple[str, dict]:
    response = client.chat.completions.create(
        model=model,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(domain, feature)},
        ],
    )
    usage = {
        "input": response.usage.prompt_tokens,
        "output": response.usage.completion_tokens,
    }
    prices = MODEL_PRICES.get(model)
    cost = (usage["input"] * prices["input"] + usage["output"] * prices["output"]) / 1_000_000 if prices else None
    _cost_log.info(
        "llm_call",
        extra={
            "operation": "generate_synthetic",
            "model": model,
            "input_tokens": usage["input"],
            "output_tokens": usage["output"],
            "total_tokens": usage["input"] + usage["output"],
            "cost_usd": round(cost, 6) if cost is not None else None,
            "question_preview": f"{domain}/{feature}",
        },
    )
    return response.choices[0].message.content, usage


def format_cost(input_tokens: int, output_tokens: int, model: str) -> str:
    prices = MODEL_PRICES.get(model)
    if not prices:
        return "н/д (модель не в таблице цен)"
    cost = (input_tokens * prices["input"] + output_tokens * prices["output"]) / 1_000_000
    return f"${cost:.5f}"


def make_filename(domain: str, feature: str, idx: int) -> str:
    slug = feature.lower().replace(" ", "_").replace("-", "_")[:40]
    domain_slug = domain.replace(" ", "_").replace("-", "_")
    return f"{idx:03d}_{domain_slug}_{slug}.md"


def parse_args():
    parser = argparse.ArgumentParser(description="Генерация синтетических PRD для RAG")
    parser.add_argument("--count", type=int, default=50,
                        help="Сколько PRD сгенерировать (по умолчанию 50)")
    parser.add_argument("--domain", type=str, default=None,
                        help=f"Домен (по умолчанию все). Доступные: {', '.join(DOMAINS)}")
    parser.add_argument("--model", type=str, default="gpt-4o-mini",
                        help="Модель OpenAI (по умолчанию gpt-4o-mini)")
    parser.add_argument("--delay", type=float, default=0.5,
                        help="Пауза между запросами в секундах (по умолчанию 0.5)")
    parser.add_argument("--estimate", action="store_true",
                        help="Сгенерировать 1 PRD, показать реальную стоимость и экстраполировать на --count")
    return parser.parse_args()


def build_task_list(domain_filter: str | None, count: int) -> list[tuple[str, str]]:
    tasks = []
    if domain_filter:
        features = FEATURE_IDEAS.get(domain_filter, [])
        for f in features:
            tasks.append((domain_filter, f))
    else:
        for domain, features in FEATURE_IDEAS.items():
            for f in features:
                tasks.append((domain, f))

    result = []
    while len(result) < count:
        result.extend(tasks)
    return result[:count]


def run_estimate(client: OpenAI, tasks: list, model: str):
    """Генерирует 1 PRD, измеряет реальные токены и экстраполирует на весь список."""
    domain, feature = tasks[0]
    print(f"Тестовый запрос: {domain} / {feature}...")
    _, usage = generate_one(client, domain, feature, model)

    inp, out = usage["input"], usage["output"]
    total = len(tasks)

    print(f"\n{'='*50}")
    print(f"Модель:              {model}")
    print(f"Токены на 1 PRD:     {inp} input + {out} output = {inp + out} total")
    print(f"Стоимость 1 PRD:     {format_cost(inp, out, model)}")
    print(f"{'─'*50}")
    print(f"Экстраполяция на {total} PRD:")
    print(f"  Токены input:      {inp * total:,}")
    print(f"  Токены output:     {out * total:,}")
    print(f"  Итого токенов:     {(inp + out) * total:,}")
    print(f"  Ожидаемая стоимость: {format_cost(inp * total, out * total, model)}")
    print(f"{'='*50}")


def main():
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    template = TEMPLATE_PATH.read_text(encoding="utf-8")

    client = OpenAI()
    tasks = build_task_list(args.domain, args.count)

    if args.estimate:
        run_estimate(client, tasks, args.model)
        return

    print(f"Генерируем {len(tasks)} PRD (модель: {args.model})")
    print(f"Сохраняем в: {OUTPUT_DIR}\n")

    success = 0
    errors = 0
    total_input = total_output = 0

    for idx, (domain, feature) in enumerate(tasks, 1):
        print(f"[{idx:3d}/{len(tasks)}] {domain} / {feature}...", end=" ", flush=True)
        try:
            raw, usage = generate_one(client, domain, feature, args.model)
            total_input += usage["input"]
            total_output += usage["output"]

            fields = json.loads(raw)
            prd_text = render_prd(template, fields)

            filename = make_filename(domain, feature, idx)
            (OUTPUT_DIR / filename).write_text(prd_text, encoding="utf-8")
            print(f"✓  [{usage['input']}+{usage['output']} tok]")
            success += 1
        except Exception as e:
            print(f"✗ ошибка: {e}")
            errors += 1

        if idx < len(tasks):
            time.sleep(args.delay)

    print(f"\n{'='*50}")
    print(f"Готово:              {success} успешно, {errors} ошибок")
    print(f"Токены input:        {total_input:,}")
    print(f"Токены output:       {total_output:,}")
    print(f"Итого токенов:       {total_input + total_output:,}")
    print(f"Стоимость:           {format_cost(total_input, total_output, args.model)}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
