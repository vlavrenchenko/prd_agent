"""Форматирование результатов критики для Telegram-бота."""
import re

CRITERION_KEYWORDS: dict[str, list[str]] = {
    "metrics":         ["метрик", "измерим", "kpi", "показател"],
    "segment":         ["сегмент", "аудитор", "пользовател", "целев"],
    "requirements":    ["требовани", "функционал", "фича", "функц"],
    "out_of_scope":    ["скоуп", "scope", "не входит", "out of"],
    "open_questions":  ["вопрос", "открыт", "неопределённ", "неопределенн"],
    "no_fluff":        ["вод", "расплывч", "конкретик", "fluff", "размыт", "абстрактн", "общие слова"],
    "jtbd":            ["jtbd", "джтбд", "job to be done", "job", "джоб", "зачем пользовател"],
    "business_metric": ["бизнес", "метрик", "revenue", "retention", "конверси"],
}

CRITERION_NAMES: dict[str, str] = {
    "metrics":         "Метрики измеримы",
    "segment":         "Сегмент пользователей",
    "requirements":    "Функц. требования",
    "out_of_scope":    "Out of scope",
    "open_questions":  "Открытые вопросы",
    "no_fluff":        "Отсутствие воды",
    "jtbd":            "JTBD описан",
    "business_metric": "Бизнес-метрика",
}

CRITERION_ORDER: list[str] = list(CRITERION_KEYWORDS.keys())


def detect_criterion(text: str) -> str | None:
    """Определяет критерий по тексту вопроса. Возвращает criterion id или None."""
    text_lower = text.lower()

    # Ключевые слова проверяем первыми: "джтбд на 1?" → jtbd, не metrics
    for criterion_id, keywords in CRITERION_KEYWORDS.items():
        if any(kw in text_lower for kw in keywords):
            return criterion_id

    # По номеру критерия как запасной вариант: "про 3", "критерий 5"
    match = re.search(r'\b([1-8])\b', text_lower)
    if match:
        idx = int(match.group(1)) - 1
        if 0 <= idx < len(CRITERION_ORDER):
            return CRITERION_ORDER[idx]

    return None


def format_explanation(criterion_id: str, critique_result: dict) -> str:
    score = critique_result["scores"].get(criterion_id, 0)
    explanation = critique_result["explanations"].get(criterion_id, "Объяснение недоступно.")
    name = CRITERION_NAMES.get(criterion_id, criterion_id)
    emoji = "✅" if score == 2 else "⚠️" if score == 1 else "❌"
    return f"{emoji} *{name}: {score}/2*\n\n{explanation}"


def format_all_non_perfect(critique_result: dict) -> str:
    scores = critique_result["scores"]
    score = critique_result.get("score", sum(scores.values()))
    max_score = critique_result.get("max_score", 16)
    passed = critique_result.get("passed", score >= 11)

    non_perfect = [cid for cid in CRITERION_ORDER if scores.get(cid, 0) < 2]
    if not non_perfect:
        return "✅ Все критерии получили максимальный балл."

    status = "✅" if passed else "⚠️"
    header = "Вот что можно улучшить:" if passed else "Основные замечания:"
    lines = [f"{status} *Оценка PRD: {score}/{max_score}* — {header}\n"]

    for criterion_id in non_perfect:
        s = scores.get(criterion_id, 0)
        explanation = critique_result["explanations"].get(criterion_id, "")
        name = CRITERION_NAMES.get(criterion_id, criterion_id)
        emoji = "⚠️" if s == 1 else "❌"
        lines.append(f"{emoji} *{name}: {s}/2*\n{explanation}\n")

    lines.append("_Чтобы узнать подробнее — спроси про конкретный критерий: «почему JTBD на 1?»_")
    return "\n".join(lines)
