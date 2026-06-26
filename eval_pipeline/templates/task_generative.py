import json
import re


def load(source: str, limit: int | None) -> list[dict]:
    rows = [json.loads(line) for line in open(source, encoding="utf-8")]
    return rows[:limit] if limit is not None else rows


def build_prompt(item: dict, config) -> str:
    return item["question"]


def extract_answer(raw: str, item: dict) -> tuple[str | None, str]:
    matches = re.findall(r"-?\d+(?:\.\d+)?", raw)
    if not matches:
        return None, "failed"
    return matches[-1], "extracted"


def score(answer: str | None, item: dict) -> float | None:
    if answer is None:
        return None
    return 1.0 if str(answer).strip() == str(item.get("gold")).strip() else 0.0
