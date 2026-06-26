import csv
import re


def load(source: str, limit: int | None) -> list[dict]:
    rows = []
    with open(source, newline="", encoding="utf-8") as handle:
        for idx, row in enumerate(csv.DictReader(handle)):
            variant = "question"
            rows.append(
                {
                    "id": f"gsm8k_rup-{idx:06d}-{variant}",
                    "question": row[variant],
                    "gold": _gold(row["answer"]),
                    "answer": row["answer"],
                    "variant": variant,
                    "row_idx": idx,
                }
            )
            if limit is not None and len(rows) >= limit:
                return rows
    return rows


def build_prompt(item: dict, config) -> str:
    return f"Return only the final numeric answer.\nProblem: {item['question']}\nAnswer:"


def extract_answer(raw: str, item: dict) -> tuple[str | None, str]:
    marker = re.search(r"####\s*([-+]?\d+(?:\.\d+)?)", raw)
    if marker:
        return _normalize(marker.group(1)), "extracted"
    numbers = re.findall(r"[-+]?\d+(?:\.\d+)?", raw)
    if numbers:
        return _normalize(numbers[-1]), "fallback"
    return None, "failed"


def score(answer: str | None, item: dict) -> float | None:
    if answer is None:
        return None
    return 1.0 if _normalize(answer) == _normalize(item["gold"]) else 0.0


def _gold(answer: str) -> str:
    marker = re.search(r"####\s*([-+]?\d+(?:\.\d+)?)", answer)
    return _normalize(marker.group(1)) if marker else ""


def _normalize(value: str) -> str:
    value = value.strip().replace(",", "")
    if "." in value:
        value = value.rstrip("0").rstrip(".")
    return value
