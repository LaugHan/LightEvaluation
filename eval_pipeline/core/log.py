import json
import sys
import time
import traceback
from datetime import UTC, datetime
from pathlib import Path

from eval_pipeline.core.io import append_jsonl


class RunLogger:
    def __init__(self, output_dir: Path, config, rank: int | None):
        self.output_dir = output_dir
        self.config = config
        self.rank = rank
        self.logs_dir = output_dir / "logs"
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        suffix = f".rank{rank}" if rank is not None else ""
        self.run_path = self.logs_dir / f"run{suffix}.jsonl"
        self.progress_path = self.logs_dir / f"progress{suffix}.jsonl"
        self.errors_path = self.logs_dir / f"errors{suffix}.jsonl"
        self.state_path = self.logs_dir / f"state{suffix}.json"
        self.fatal_path = self.logs_dir / f"fatal{suffix}.log"
        self.started = time.monotonic()
        self.last_progress = 0.0
        self.bad_records_logged = 0

    def run_event(self, event: str, level: str = "info", **fields) -> None:
        append_jsonl(self.run_path, [self._event(event, level, **fields)])

    def progress(self, completed: int, total: int, counts: dict, force: bool = False) -> None:
        now = time.monotonic()
        interval_due = now - self.last_progress >= self.config.logging.progress_interval_sec
        every_due = completed == total or completed % self.config.logging.progress_every == 0
        if not (force or interval_due or every_due):
            return
        self.last_progress = now
        elapsed = max(now - self.started, 0.000001)
        speed = completed / elapsed
        pending = max(total - completed, 0)
        eta = pending / speed if speed > 0 else None
        event = self._event(
            "progress",
            completed=completed,
            total=total,
            pending=pending,
            percent=round((completed / total * 100.0) if total else 100.0, 2),
            elapsed_sec=round(elapsed, 3),
            items_per_sec=round(speed, 3),
            eta_sec=round(eta, 3) if eta is not None else None,
            **counts,
        )
        append_jsonl(self.progress_path, [event])
        self.state("running", completed, total, counts, event)
        if self.config.logging.console:
            self._console_progress(event)

    def item_issue(self, record: dict, prompt: str) -> None:
        if record["status"] not in {"error", "extract_failed", "truncated"}:
            return
        if self.bad_records_logged >= self.config.logging.sample_bad_records:
            return
        self.bad_records_logged += 1
        append_jsonl(
            self.errors_path,
            [
                self._event(
                    "item_error" if record["status"] == "error" else "item_warning",
                    "error" if record["status"] == "error" else "warning",
                    id=record.get("id"),
                    sample_idx=record.get("sample_idx"),
                    status=record.get("status"),
                    finish_reason=record.get("finish_reason"),
                    message=record.get("error"),
                    prompt_preview=prompt[: self.config.logging.error_preview_chars],
                    raw_preview=record.get("raw", "")[: self.config.logging.error_preview_chars],
                )
            ],
        )

    def state(self, status: str, completed: int, total: int, counts: dict, progress_event: dict | None = None) -> None:
        state = {
            "status": status,
            "name": self.config.name,
            "backend": self.config.model.backend,
            "rank": self.rank,
            "updated_at": _now(),
            "completed": completed,
            "total": total,
            "percent": round((completed / total * 100.0) if total else 100.0, 2),
            "counts": counts,
        }
        if progress_event is not None:
            state["items_per_sec"] = progress_event.get("items_per_sec")
            state["eta_sec"] = progress_event.get("eta_sec")
        self.state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    def finish(self, summary: dict, completed: int, total: int, counts: dict, written: int, dropped: int) -> None:
        elapsed = max(time.monotonic() - self.started, 0.000001)
        speed = completed / elapsed
        self.run_event(
            "run_finish",
            status="succeeded",
            total=summary["total"],
            completed=completed,
            expected_total=total,
            written=written,
            dropped=dropped,
            elapsed_sec=round(elapsed, 3),
            items_per_sec=round(speed, 3),
            rates=summary["rates"],
            within_thresholds=summary["within_thresholds"],
            summary=str(self.output_dir / "summary.json"),
        )
        self.state(
            "succeeded",
            completed,
            total,
            counts,
            {"items_per_sec": round(speed, 3), "eta_sec": 0.0},
        )

    def fail(self, exc: Exception) -> None:
        self.fatal_path.write_text("".join(traceback.format_exception(exc)), encoding="utf-8")
        self.run_event(
            "run_failed",
            "error",
            error_type=type(exc).__name__,
            message=str(exc),
            traceback=str(self.fatal_path),
        )
        state = {
            "status": "failed",
            "name": self.config.name,
            "backend": self.config.model.backend,
            "rank": self.rank,
            "updated_at": _now(),
            "last_error": {
                "type": type(exc).__name__,
                "message": str(exc),
                "traceback": str(self.fatal_path),
            },
        }
        self.state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    def _event(self, event: str, level: str = "info", **fields) -> dict:
        return {"ts": _now(), "level": level, "event": event, "rank": self.rank, **fields}

    def _console_progress(self, event: dict) -> None:
        print(
            f"[{event['ts']}] {self.config.name} {event['completed']}/{event['total']} "
            f"{event['percent']}% ok={event.get('ok', 0)} truncated={event.get('truncated', 0)} "
            f"extract_failed={event.get('extract_failed', 0)} error={event.get('error', 0)} "
            f"speed={event['items_per_sec']}/s eta={event['eta_sec']}s",
            file=sys.stderr,
        )


def _now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")
