from statistics import mean
from typing import Iterable


STATUSES = ("ok", "truncated", "extract_failed", "error")


def build_summary(name: str, records: Iterable[dict], thresholds: dict, config_snapshot: dict) -> dict:
    rows = list(records)
    total = len(rows)
    rates = {
        status: (sum(1 for record in rows if record.get("status") == status) / total if total else 0.0)
        for status in STATUSES
    }
    scored = [record["score"] for record in rows if record.get("score") is not None]
    within_thresholds = all(rates[status] <= thresholds[status] for status in ("truncated", "extract_failed", "error"))
    return {
        "name": name,
        "total": total,
        "rates": rates,
        "thresholds": thresholds,
        "within_thresholds": within_thresholds,
        "mean_score": mean(scored) if scored else None,
        "config": config_snapshot,
    }
