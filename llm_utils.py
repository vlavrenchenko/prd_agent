"""Утилиты для логирования стоимости LLM-вызовов."""
import json
from pathlib import Path

from logger import get_cost_logger

_cost_log = get_cost_logger()

_PRICING_PATH = Path(__file__).parent / "config" / "models_pricing.json"


def _load_prices() -> dict:
    if not _PRICING_PATH.exists():
        return {}
    raw = json.loads(_PRICING_PATH.read_text())
    prices = {}
    for model_id, model_data in raw.get("models", {}).items():
        standard = model_data.get("pricing", {}).get("standard", {})
        if "input" in standard:
            prices[model_id] = {"input": standard["input"], "output": standard["output"]}
        elif "short_context" in standard:
            prices[model_id] = {
                "input": standard["short_context"]["input"],
                "output": standard["short_context"]["output"],
            }
    return prices


_MODEL_PRICES = _load_prices()


def log_cost(operation: str, model: str, response, context: str = "") -> None:
    usage = response.usage
    inp, out = usage.prompt_tokens, usage.completion_tokens
    prices = _MODEL_PRICES.get(model)
    cost = (inp * prices["input"] + out * prices["output"]) / 1_000_000 if prices else None
    _cost_log.info(
        "llm_call",
        extra={
            "operation": operation,
            "model": model,
            "input_tokens": inp,
            "output_tokens": out,
            "total_tokens": inp + out,
            "cost_usd": round(cost, 6) if cost is not None else None,
            "question_preview": context[:80],
        },
    )
