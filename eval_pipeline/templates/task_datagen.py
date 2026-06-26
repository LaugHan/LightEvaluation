import json


def load(source: str, limit: int | None) -> list[dict]:
    rows = [json.loads(line) for line in open(source, encoding="utf-8")]
    return rows[:limit] if limit is not None else rows


def build_prompt(item: dict, config) -> str:
    return item["prompt"]


def extract_answer(raw: str, item: dict) -> tuple[str | None, str]:
    text = raw.strip()
    if not text:
        return None, "failed"
    return text, "extracted"
