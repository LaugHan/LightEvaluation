import csv


def load(source: str, limit: int | None) -> list[dict]:
    rows = []
    with open(source, newline="", encoding="utf-8") as handle:
        for idx, row in enumerate(csv.DictReader(handle)):
            rows.append(
                {
                    "id": f"gsm8k_rup-{idx:06d}-datagen",
                    "question": row["question"],
                    "row_idx": idx,
                }
            )
            if limit is not None and len(rows) >= limit:
                return rows
    return rows


def build_prompt(item: dict, config) -> str:
    return f"Write one short sentence about how to solve this math problem. End with a period.\nProblem: {item['question']}\nSentence:"


def extract_answer(raw: str, item: dict) -> tuple[str | None, str]:
    text = raw.strip()
    if not text:
        return None, "failed"
    return text, "extracted"
