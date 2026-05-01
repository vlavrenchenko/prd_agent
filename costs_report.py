import argparse
import json
import datetime
from pathlib import Path

COSTS_LOG = Path(__file__).parent / "logs" / "costs.log"


def parse_args():
    parser = argparse.ArgumentParser(description="Отчёт о расходах на OpenAI API")
    parser.add_argument(
        "--period", choices=["hour", "day", "week", "month", "all"],
        default="day", help="Период отчёта (по умолчанию: day)"
    )
    parser.add_argument(
        "--detail", action="store_true",
        help="Показать каждый запрос отдельно"
    )
    return parser.parse_args()


def load_records(since: datetime.datetime) -> list[dict]:
    if not COSTS_LOG.exists():
        return []
    records = []
    for line in COSTS_LOG.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            d = json.loads(line)
            ts = datetime.datetime.fromisoformat(d["ts"])
            if ts >= since:
                records.append(d)
        except Exception:
            continue
    return records


def period_start(period: str) -> datetime.datetime:
    now = datetime.datetime.now()
    if period == "hour":
        return now - datetime.timedelta(hours=1)
    if period == "day":
        return now.replace(hour=0, minute=0, second=0, microsecond=0)
    if period == "week":
        return (now - datetime.timedelta(days=now.weekday())).replace(hour=0, minute=0, second=0, microsecond=0)
    if period == "month":
        return now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    return datetime.datetime.min  # all


def main():
    args = parse_args()
    since = period_start(args.period)
    records = load_records(since)

    period_labels = {
        "hour": "последний час",
        "day": "сегодня",
        "week": "эта неделя",
        "month": "этот месяц",
        "all": "всё время",
    }

    print(f"\n{'='*55}")
    print(f"💰 Расходы на OpenAI API — {period_labels[args.period]}")
    print(f"{'='*55}")

    if not records:
        print("  Нет данных за выбранный период.")
        print(f"{'='*55}\n")
        return

    def _f(val) -> float:
        try:
            return float(val) if val is not None else 0.0
        except (ValueError, TypeError):
            return 0.0

    total_cost = sum(_f(r.get("cost_usd")) for r in records)
    total_tokens = sum(int(_f(r.get("total_tokens"))) for r in records)
    total_input = sum(int(_f(r.get("input_tokens"))) for r in records)
    total_output = sum(int(_f(r.get("output_tokens"))) for r in records)

    by_model: dict[str, dict] = {}
    for r in records:
        m = r.get("model", "unknown")
        if m not in by_model:
            by_model[m] = {"cost": 0.0, "tokens": 0, "calls": 0}
        by_model[m]["cost"] += _f(r.get("cost_usd"))
        by_model[m]["tokens"] += int(_f(r.get("total_tokens")))
        by_model[m]["calls"] += 1

    print(f"  Запросов:       {len(records)}")
    print(f"  Токенов всего:  {total_tokens:,}  (in: {total_input:,} / out: {total_output:,})")
    print(f"  Стоимость:      ${total_cost:.4f}")

    if len(by_model) > 1:
        print(f"\n  По моделям:")
        for model, stats in sorted(by_model.items(), key=lambda x: -x[1]["cost"]):
            print(f"    {model}: ${stats['cost']:.4f}  ({stats['calls']} запросов, {stats['tokens']:,} токенов)")

    if args.detail:
        print(f"\n  {'Время':<20} {'Модель':<15} {'$':>8}  {'Токены':>7}  Вопрос")
        print(f"  {'-'*80}")
        for r in records:
            ts = r.get("ts", "")[:16]
            model = r.get("model", "")[:14]
            cost = _f(r.get("cost_usd"))
            tokens = int(_f(r.get("total_tokens")))
            question = r.get("question_preview", "")[:40]
            print(f"  {ts:<20} {model:<15} ${cost:>7.4f}  {tokens:>7,}  {question}")

    print(f"{'='*55}\n")


if __name__ == "__main__":
    main()
