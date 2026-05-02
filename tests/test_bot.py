"""Тесты утилит bot.py — без Telegram и без реального LLM."""
import pytest
from unittest.mock import patch, MagicMock


def _import_bot():
    """Импортирует bot с замоканным aiogram и переменными окружения."""
    import sys
    # Замокируем aiogram до импорта bot, чтобы не нужен был токен
    aiogram_mock = MagicMock()
    aiogram_mock.filters = MagicMock()
    aiogram_mock.filters.Command = MagicMock()
    aiogram_mock.filters.CommandStart = MagicMock()
    aiogram_mock.types = MagicMock()
    sys.modules.setdefault("aiogram", aiogram_mock)
    sys.modules.setdefault("aiogram.filters", aiogram_mock.filters)
    sys.modules.setdefault("aiogram.types", aiogram_mock.types)

    with patch.dict("os.environ", {"TELEGRAM_BOT_TOKEN": "mock"}):
        import importlib
        if "bot" in sys.modules:
            importlib.reload(sys.modules["bot"])
            return sys.modules["bot"]
        import bot as b
        return b


# ---------------------------------------------------------------------------
# _detect_criterion
# ---------------------------------------------------------------------------

class TestDetectCriterion:

    def setup_method(self):
        self.bot = _import_bot()

    def test_detects_by_number_1(self):
        assert self.bot._detect_criterion("расскажи про критерий 1") == "metrics"

    def test_detects_by_number_8(self):
        assert self.bot._detect_criterion("что не так с 8?") == "business_metric"

    def test_detects_by_number_3(self):
        assert self.bot._detect_criterion("объясни пункт 3") == "requirements"

    def test_detects_metrics_by_keyword(self):
        assert self.bot._detect_criterion("почему метрики получили 0?") == "metrics"

    def test_detects_segment_by_keyword(self):
        assert self.bot._detect_criterion("расскажи про сегмент") == "segment"

    def test_detects_jtbd_by_keyword(self):
        assert self.bot._detect_criterion("что с jtbd?") == "jtbd"

    def test_detects_out_of_scope_by_keyword(self):
        assert self.bot._detect_criterion("что такое out of scope") == "out_of_scope"

    def test_detects_no_fluff_by_keyword(self):
        assert self.bot._detect_criterion("документ расплывчатый") == "no_fluff"

    def test_returns_none_when_no_match(self):
        assert self.bot._detect_criterion("привет как дела") is None

    def test_number_takes_priority_over_keyword(self):
        # "метрики" → metrics (id 0), "1" → metrics тоже, но по номеру
        result = self.bot._detect_criterion("метрики 1")
        assert result == "metrics"

    def test_number_out_of_range_returns_none(self):
        # 9 не входит в диапазон 1-8
        assert self.bot._detect_criterion("критерий 9") is None

    def test_case_insensitive_keywords(self):
        assert self.bot._detect_criterion("KPI не указан") == "metrics"


# ---------------------------------------------------------------------------
# _format_explanation
# ---------------------------------------------------------------------------

class TestFormatExplanation:

    def setup_method(self):
        self.bot = _import_bot()
        self.critique_result = {
            "scores": {
                "metrics": 0, "segment": 1, "requirements": 2,
                "out_of_scope": 0, "open_questions": 1, "no_fluff": 2,
                "jtbd": 0, "business_metric": 1,
            },
            "explanations": {
                "metrics": "Метрики не указаны совсем.",
                "segment": "Сегмент описан размыто.",
                "requirements": "Хорошо расписаны все требования.",
                "out_of_scope": "Нет раздела out of scope.",
                "open_questions": "Есть один вопрос.",
                "no_fluff": "Документ лаконичный.",
                "jtbd": "JTBD не сформулирован.",
                "business_metric": "Метрика названа, но без цифр.",
            },
        }

    def test_score_0_shows_red_emoji(self):
        result = self.bot._format_explanation("metrics", self.critique_result)
        assert result.startswith("❌")

    def test_score_1_shows_warning_emoji(self):
        result = self.bot._format_explanation("segment", self.critique_result)
        assert result.startswith("⚠️")

    def test_score_2_shows_green_emoji(self):
        result = self.bot._format_explanation("requirements", self.critique_result)
        assert result.startswith("✅")

    def test_contains_score_fraction(self):
        result = self.bot._format_explanation("segment", self.critique_result)
        assert "1/2" in result

    def test_contains_explanation_text(self):
        result = self.bot._format_explanation("metrics", self.critique_result)
        assert "Метрики не указаны совсем." in result

    def test_contains_criterion_name(self):
        result = self.bot._format_explanation("jtbd", self.critique_result)
        assert "JTBD описан" in result

    def test_missing_explanation_shows_fallback(self):
        result = self.bot._format_explanation("metrics", {"scores": {"metrics": 0}, "explanations": {}})
        assert "Объяснение недоступно." in result


# ---------------------------------------------------------------------------
# _format_all_non_perfect
# ---------------------------------------------------------------------------

class TestFormatAllNonPerfect:

    def setup_method(self):
        self.bot = _import_bot()

    def _make_result(self, scores: dict) -> dict:
        return {
            "scores": scores,
            "explanations": {k: f"Объяснение: {k}" for k in scores},
        }

    def test_all_perfect_returns_success_message(self):
        scores = {k: 2 for k in ["metrics", "segment", "requirements", "out_of_scope",
                                   "open_questions", "no_fluff", "jtbd", "business_metric"]}
        result = self.bot._format_all_non_perfect(self._make_result(scores))
        assert "✅" in result
        assert "максимальный балл" in result

    def test_shows_only_non_perfect_criteria(self):
        scores = {k: 2 for k in ["metrics", "segment", "requirements", "out_of_scope",
                                   "open_questions", "no_fluff", "jtbd", "business_metric"]}
        scores["jtbd"] = 0
        scores["metrics"] = 1
        result = self.bot._format_all_non_perfect(self._make_result(scores))
        assert "JTBD" in result
        assert "Метрики" in result
        # Criteria with score 2 should not appear as entries
        assert "Объяснение: segment" not in result

    def test_score_0_uses_red_emoji(self):
        scores = {k: 2 for k in ["metrics", "segment", "requirements", "out_of_scope",
                                   "open_questions", "no_fluff", "jtbd", "business_metric"]}
        scores["jtbd"] = 0
        result = self.bot._format_all_non_perfect(self._make_result(scores))
        assert "❌" in result

    def test_score_1_uses_warning_emoji(self):
        scores = {k: 2 for k in ["metrics", "segment", "requirements", "out_of_scope",
                                   "open_questions", "no_fluff", "jtbd", "business_metric"]}
        scores["metrics"] = 1
        result = self.bot._format_all_non_perfect(self._make_result(scores))
        assert "⚠️" in result
