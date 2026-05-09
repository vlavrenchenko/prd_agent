"""Тесты critique.formatter — без Telegram и без реального LLM."""
from critique.formatter import (
    detect_criterion,
    format_explanation,
    format_all_non_perfect,
    CRITERION_ORDER,
)


# ---------------------------------------------------------------------------
# detect_criterion
# ---------------------------------------------------------------------------

def test_detects_by_number_1():
    assert detect_criterion("расскажи про критерий 1") == "metrics"

def test_detects_by_number_8():
    assert detect_criterion("что не так с 8?") == "business_metric"

def test_detects_by_number_3():
    assert detect_criterion("объясни пункт 3") == "requirements"

def test_detects_metrics_by_keyword():
    assert detect_criterion("почему метрики получили 0?") == "metrics"

def test_detects_segment_by_keyword():
    assert detect_criterion("расскажи про сегмент") == "segment"

def test_detects_jtbd_by_keyword():
    assert detect_criterion("что с jtbd?") == "jtbd"

def test_detects_jtbd_russian_transliteration():
    assert detect_criterion("почему джтбд на 1?") == "jtbd"

def test_detects_jtbd_full_phrase():
    assert detect_criterion("job to be done не описан") == "jtbd"

def test_detects_out_of_scope_by_keyword():
    assert detect_criterion("что такое out of scope") == "out_of_scope"

def test_detects_no_fluff_by_keyword():
    assert detect_criterion("документ расплывчатый") == "no_fluff"

def test_detects_no_fluff_vody():
    # "вод" должен матчить "воды"
    assert detect_criterion("слишком много воды") == "no_fluff"

def test_returns_none_when_no_match():
    assert detect_criterion("привет как дела") is None

def test_number_out_of_range_returns_none():
    assert detect_criterion("критерий 9") is None

def test_case_insensitive_keywords():
    assert detect_criterion("KPI не указан") == "metrics"


# ---------------------------------------------------------------------------
# format_explanation
# ---------------------------------------------------------------------------

def _make_result(scores: dict, explanations: dict = None) -> dict:
    if explanations is None:
        explanations = {k: f"Объяснение: {k}" for k in scores}
    return {"scores": scores, "explanations": explanations, "score": sum(scores.values()), "passed": True}


def test_score_0_shows_red_emoji():
    result = format_explanation("metrics", _make_result({"metrics": 0}))
    assert result.startswith("❌")

def test_score_1_shows_warning_emoji():
    result = format_explanation("segment", _make_result({"segment": 1}))
    assert result.startswith("⚠️")

def test_score_2_shows_green_emoji():
    result = format_explanation("requirements", _make_result({"requirements": 2}))
    assert result.startswith("✅")

def test_contains_score_fraction():
    result = format_explanation("segment", _make_result({"segment": 1}))
    assert "1/2" in result

def test_contains_explanation_text():
    result = format_explanation("metrics", _make_result({"metrics": 0}, {"metrics": "Метрики не указаны."}))
    assert "Метрики не указаны." in result

def test_contains_criterion_name():
    result = format_explanation("jtbd", _make_result({"jtbd": 0}))
    assert "JTBD описан" in result

def test_missing_explanation_shows_fallback():
    result = format_explanation("metrics", {"scores": {"metrics": 0}, "explanations": {}})
    assert "Объяснение недоступно." in result


# ---------------------------------------------------------------------------
# format_all_non_perfect
# ---------------------------------------------------------------------------

def _make_full_result(scores: dict) -> dict:
    total = sum(scores.values())
    return {
        "scores": scores,
        "explanations": {k: f"Объяснение: {k}" for k in scores},
        "score": total,
        "max_score": 16,
        "passed": total >= 11,
    }


def _all_max() -> dict:
    return {k: 2 for k in CRITERION_ORDER}

def test_all_perfect_returns_success_message():
    result = format_all_non_perfect(_make_full_result(_all_max()))
    assert "✅" in result
    assert "максимальный балл" in result

def test_shows_only_non_perfect_criteria():
    scores = {**_all_max(), "jtbd": 0, "metrics": 1}
    result = format_all_non_perfect(_make_full_result(scores))
    assert "JTBD" in result
    assert "Метрики" in result
    assert "Объяснение: segment" not in result

def test_score_0_uses_red_emoji():
    scores = {**_all_max(), "jtbd": 0}
    result = format_all_non_perfect(_make_full_result(scores))
    assert "❌" in result

def test_score_1_uses_warning_emoji():
    scores = {**_all_max(), "metrics": 1}
    result = format_all_non_perfect(_make_full_result(scores))
    assert "⚠️" in result

def test_passed_prd_has_helpful_header():
    scores = {**_all_max(), "jtbd": 1}  # 15/16 — прошёл
    result = format_all_non_perfect(_make_full_result(scores))
    assert "можно улучшить" in result

def test_failed_prd_has_issues_header():
    scores = {k: 0 for k in CRITERION_ORDER}  # 0/16 — не прошёл
    result = format_all_non_perfect(_make_full_result(scores))
    assert "замечани" in result.lower()

def test_contains_hint_for_specific_question():
    scores = {**_all_max(), "jtbd": 1}
    result = format_all_non_perfect(_make_full_result(scores))
    assert "спроси про конкретный" in result
