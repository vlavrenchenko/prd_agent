"""LLM-логика критика PRD: оценка по 8 критериям с объяснениями."""
import json
from pathlib import Path

from openai import OpenAI

from llm_utils import log_cost
from logger import get_logger

log = get_logger("critique")

_CRITERIA_PATH = Path(__file__).parent / "criteria.json"

_CRITIQUE_SYSTEM_PROMPT = (
    "Ты рецензент PRD в продуктовой команде. "
    "Оцени документ по каждому критерию и дай конкретный actionable feedback. "
    "Для каждого критерия объясни почему поставил именно эту оценку — что есть и чего не хватает. "
    "Верни результат строго в JSON."
)

_CRITIQUE_JSON_SCHEMA = (
    "{\n"
    '  "scores": {"metrics": 0, "segment": 0, "requirements": 0, "out_of_scope": 0, '
    '"open_questions": 0, "no_fluff": 0, "jtbd": 0, "business_metric": 0},\n'
    '  "explanations": {\n'
    '    "metrics": "почему поставил эту оценку",\n'
    '    "segment": "почему поставил эту оценку",\n'
    '    "requirements": "почему поставил эту оценку",\n'
    '    "out_of_scope": "почему поставил эту оценку",\n'
    '    "open_questions": "почему поставил эту оценку",\n'
    '    "no_fluff": "почему поставил эту оценку",\n'
    '    "jtbd": "почему поставил эту оценку",\n'
    '    "business_metric": "почему поставил эту оценку"\n'
    '  },\n'
    '  "issues": ["конкретное замечание для улучшения 1", "замечание 2"]\n'
    "}"
)


def load_criteria() -> dict:
    return json.loads(_CRITERIA_PATH.read_text(encoding="utf-8"))


def _clamp_scores(scores: dict, criteria: list) -> dict:
    max_per_criterion = {c["id"]: c["max"] for c in criteria}
    return {
        key: min(int(val), max_per_criterion.get(key, val))
        for key, val in scores.items()
    }


def _run_critique_llm(client: OpenAI, prd_text: str, criteria_config: dict, operation: str) -> dict:
    criteria_text = "\n".join(
        f"{i+1}. {c['name']} (0-{c['max']} баллов):\n"
        + "\n".join(f"   {score}: {desc}" for score, desc in c["levels"].items())
        for i, c in enumerate(criteria_config["criteria"])
    )

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": _CRITIQUE_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Оцени PRD по следующим критериям:\n\n{criteria_text}\n\n"
                    f"PRD для оценки:\n{prd_text}\n\n"
                    f"Верни JSON:\n{_CRITIQUE_JSON_SCHEMA}"
                ),
            },
        ],
    )
    log_cost(operation, "gpt-4o-mini", response, prd_text[:80])

    result = json.loads(response.choices[0].message.content)
    scores = _clamp_scores(result.get("scores", {}), criteria_config["criteria"])
    return {
        "scores": scores,
        "explanations": result.get("explanations", {}),
        "issues": result.get("issues", []),
        "total_score": sum(scores.values()),
    }


def critique_prd(prd_text: str) -> dict:
    """Оценивает готовый PRD по 8 критериям. Вызывается напрямую из бота.

    Возвращает:
        {
            "score": int,
            "max_score": int,
            "threshold": int,
            "passed": bool,
            "issues": list[str],
            "scores": dict,
            "explanations": dict,
        }
    """
    client = OpenAI()
    criteria_config = load_criteria()

    parsed = _run_critique_llm(client, prd_text, criteria_config, "critique_prd")
    total_score = parsed["total_score"]
    threshold = criteria_config["threshold"]
    max_score = criteria_config["max_score"]
    passed = total_score >= threshold

    log.info("critique_prd_done", extra={
        "score": total_score,
        "max_score": max_score,
        "passed": passed,
        "issues_count": len(parsed["issues"]),
    })

    return {
        "score": total_score,
        "max_score": max_score,
        "threshold": threshold,
        "passed": passed,
        "issues": parsed["issues"],
        "scores": parsed["scores"],
        "explanations": parsed["explanations"],
    }


def critique(state: dict) -> dict:
    """LangGraph-узел: оценивает качество PRD и возвращает замечания."""
    client = OpenAI()
    criteria_config = load_criteria()

    parsed = _run_critique_llm(client, state["prd"], criteria_config, "critique")
    total_score = parsed["total_score"]
    threshold = criteria_config["threshold"]
    passed = total_score >= threshold

    log.info("critique_done", extra={
        "feature": state["feature_description"],
        "score": total_score,
        "max_score": criteria_config["max_score"],
        "passed": passed,
        "issues_count": len(parsed["issues"]),
    })

    return {
        "critique_score": total_score,
        "prev_critique_score": state.get("critique_score", 0),
        "critique_issues": parsed["issues"] if not passed else [],
        "critique_passed": passed,
    }
