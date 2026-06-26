import json
import re


def load(source: str, limit: int | None) -> list[dict]:
    rows = [json.loads(line) for line in open(source, encoding="utf-8")]
    return rows[:limit] if limit is not None else rows


def build_prompt(item: dict, config) -> str:
    choices = "\n".join(f"{key}. {value}" for key, value in item["choices"].items())
    return f"{item['question']}\n{choices}\nAnswer with one option letter."


def extract_answer(raw: str, item: dict) -> tuple[str | None, str]:
    match = re.search(r"\b([A-Z])\b", raw.upper())
    if match is None:
        return None, "failed"
    return match.group(1), "extracted"


def score(answer: str | None, item: dict) -> float | None:
    if answer is None:
        return None
    return 1.0 if answer == item.get("gold") else 0.0
