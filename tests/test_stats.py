from eval_pipeline.core.stats import build_summary


def test_build_summary_counts_four_rates_thresholds_and_mean_score():
    records = [
        {"status": "ok", "score": 1.0},
        {"status": "ok", "score": None},
        {"status": "truncated", "score": 0.0},
        {"status": "extract_failed", "score": None},
        {"status": "error", "score": None},
    ]

    summary = build_summary(
        name="run1",
        records=records,
        thresholds={"truncated": 0.21, "extract_failed": 0.21, "error": 0.21},
        config_snapshot={"name": "run1"},
    )

    assert summary["total"] == 5
    assert summary["rates"] == {"ok": 0.4, "truncated": 0.2, "extract_failed": 0.2, "error": 0.2}
    assert summary["within_thresholds"] is True
    assert summary["mean_score"] == 0.5
    assert summary["config"] == {"name": "run1"}


def test_build_summary_marks_threshold_violation():
    summary = build_summary(
        name="run1",
        records=[{"status": "error", "score": None}],
        thresholds={"truncated": 0.0, "extract_failed": 0.0, "error": 0.5},
        config_snapshot={},
    )

    assert summary["within_thresholds"] is False
