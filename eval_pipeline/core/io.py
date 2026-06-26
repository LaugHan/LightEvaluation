import json
from contextlib import suppress
from pathlib import Path
from typing import Iterable


def record_key(record: dict) -> tuple[str, int]:
    return record["id"], record["sample_idx"]


def append_jsonl(path: str | Path, records: Iterable[dict], flush_every: int = 1) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a", encoding="utf-8") as handle:
        pending = 0
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            pending += 1
            if pending >= flush_every:
                handle.flush()
                pending = 0
        if pending:
            handle.flush()


def read_records(path: str | Path) -> list[dict]:
    source = Path(path)
    if not source.exists():
        return []
    return [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line]


def scan_done(path: str | Path) -> set[tuple[str, int]]:
    source = Path(path)
    if not source.exists():
        return set()
    text = source.read_text(encoding="utf-8")
    lines = text.splitlines(keepends=True)
    done: set[tuple[str, int]] = set()
    for idx, line in enumerate(lines):
        is_tail = idx == len(lines) - 1 and not line.endswith("\n")
        if is_tail:
            parsed = False
            with suppress(json.JSONDecodeError):
                done.add(record_key(json.loads(line)))
                parsed = True
            if line and not parsed:
                source.write_text("".join(lines[:-1]), encoding="utf-8")
        else:
            done.add(record_key(json.loads(line)))
    return done


def merge_shards(shard_paths: Iterable[str | Path], output_path: str | Path) -> int:
    seen: set[tuple[str, int]] = set()
    merged: list[dict] = []
    for shard_path in shard_paths:
        for record in read_records(shard_path):
            key = record_key(record)
            if key not in seen:
                seen.add(key)
                merged.append(record)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("", encoding="utf-8")
    append_jsonl(output, merged)
    return len(merged)
