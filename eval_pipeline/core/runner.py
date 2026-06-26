import json
from dataclasses import replace
from pathlib import Path

from eval_pipeline.core.backends import Backend, GenResult, build_backend
from eval_pipeline.core.io import append_jsonl, read_records, scan_done
from eval_pipeline.core.log import RunLogger
from eval_pipeline.core.stats import build_summary
from eval_pipeline.core.task import load_task


def resolve_status(finish_reason: str, extract_status: str) -> str:
    if finish_reason == "error":
        return "error"
    if finish_reason == "length":
        return "truncated"
    if extract_status == "failed":
        return "extract_failed"
    return "ok"


def process_one(item: dict, sample_idx: int, prompt: str, gen: GenResult, task, config) -> dict:
    try:
        answer, extract_status = task.extract_answer(gen.text, item)
        score = task.score(answer, item) if hasattr(task, "score") else None
        status = resolve_status(gen.finish_reason, extract_status)
        return {
            "id": item["id"],
            "sample_idx": sample_idx,
            "status": status,
            "prompt": prompt,
            "raw": gen.text,
            "finish_reason": gen.finish_reason,
            "answer": answer,
            "extract_status": extract_status,
            "gold": item.get("gold"),
            "score": score,
        }
    except Exception as e:
        return {
            "id": item["id"],
            "sample_idx": sample_idx,
            "status": "error",
            "prompt": prompt,
            "raw": gen.text,
            "finish_reason": gen.finish_reason,
            "error": repr(e),
        }


def run(config, backend: Backend | None = None, rank: int | None = None, world_size: int | None = None) -> dict:
    output_dir = config.workspace / "runs" / config.name
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = RunLogger(output_dir, config, rank)
    output_path, dropped_path = _record_paths(output_dir, rank)
    task = load_task(config.workspace, config.dataset.task_file)
    source = _resolve_workspace_path(config.workspace, config.dataset.source)
    items = task.load(str(source), config.dataset.limit)
    existing_records = read_records(output_path) + read_records(dropped_path)
    done = scan_done(output_path) | scan_done(dropped_path)
    units = [(item, sample_idx) for item in items for sample_idx in range(config.dataset.n_samples)]
    if rank is not None:
        units = [unit for idx, unit in enumerate(units) if idx % world_size == rank]
    pending = [(item, sample_idx, task.build_prompt(item, config)) for item, sample_idx in units if (item["id"], sample_idx) not in done]
    logger.run_event(
        "run_start",
        name=config.name,
        backend=config.model.backend,
        model=config.model.model or config.model.path,
        workspace=str(config.workspace),
        output_dir=str(output_dir),
        world_size=world_size,
        items=len(items),
        units=len(units),
        done=len(done),
        pending=len(pending),
    )
    logger.run_event("task_loaded", task_file=config.dataset.task_file, source=config.dataset.source, n_samples=config.dataset.n_samples)
    logger.run_event("resume_scanned", done=len(done), dropped_done=len(scan_done(dropped_path)), pending=len(pending))
    active_backend = backend or build_backend(config)
    logger.run_event("backend_start", backend=config.model.backend, batch_size=config.runtime.batch_size, concurrency=config.model.concurrency)
    counts = _counts(existing_records)
    completed = len(done)
    written = 0
    dropped = 0
    logger.progress(completed, len(units), counts, force=True)
    for batch in _chunks(pending, config.runtime.batch_size):
        prompts = [prompt for _, _, prompt in batch]
        generated = active_backend.generate(prompts, config) if prompts else []
        if len(generated) != len(batch):
            raise ValueError(f"backend returned {len(generated)} results for {len(batch)} prompts")
        new_records: list[dict] = []
        dropped_records: list[dict] = []
        for (item, sample_idx, prompt), gen in zip(batch, generated):
            if gen.finish_reason == "length" and config.output.on_truncate == "retry":
                retry_config = replace(
                    config,
                    generation=replace(config.generation, max_tokens=config.generation.max_tokens * 2),
                )
                gen = active_backend.generate([prompt], retry_config)[0]
            record = process_one(item, sample_idx, prompt, gen, task, config)
            counts[record["status"]] = counts.get(record["status"], 0) + 1
            logger.item_issue(record, prompt)
            if record["status"] == "truncated" and config.output.on_truncate == "drop":
                dropped_records.append(record)
            else:
                new_records.append(record)
        append_jsonl(output_path, new_records, config.output.flush_every)
        append_jsonl(dropped_path, dropped_records, config.output.flush_every)
        written += len(new_records)
        dropped += len(dropped_records)
        completed += len(batch)
        logger.progress(completed, len(units), counts)
    summary_records = read_records(output_path) + read_records(dropped_path)
    summary = _write_summary(config, output_dir, summary_records)
    logger.finish(summary, completed, len(units), counts, written, dropped)
    return summary


def pilot(config, backend: Backend | None = None, n: int = 16) -> dict:
    task = load_task(config.workspace, config.dataset.task_file)
    source = _resolve_workspace_path(config.workspace, config.dataset.source)
    items = task.load(str(source), config.dataset.limit)[:n]
    prompts = [task.build_prompt(item, config) for item in items]
    active_backend = backend or build_backend(config)
    generated = active_backend.generate(prompts, config) if prompts else []
    records = [process_one(item, 0, prompt, gen, task, config) for item, prompt, gen in zip(items, prompts, generated)]
    return build_summary(config.name, records, _thresholds(config), config.snapshot())


def merge(config) -> int:
    from eval_pipeline.core.io import merge_shards

    output_dir = config.workspace / "runs" / config.name
    shards = sorted(output_dir.glob("output.rank*.jsonl"))
    return merge_shards(shards, output_dir / "output.jsonl")


def _write_summary(config, output_dir: Path, records: list[dict]) -> dict:
    summary = build_summary(config.name, records, _thresholds(config), config.snapshot())
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def _thresholds(config) -> dict:
    return {
        "truncated": config.thresholds.truncated,
        "extract_failed": config.thresholds.extract_failed,
        "error": config.thresholds.error,
    }


def _resolve_workspace_path(workspace: Path, path: str) -> Path:
    value = Path(path)
    return value if value.is_absolute() else workspace / value


def _record_paths(output_dir: Path, rank: int | None) -> tuple[Path, Path]:
    if rank is None:
        return output_dir / "output.jsonl", output_dir / "dropped.jsonl"
    return output_dir / f"output.rank{rank}.jsonl", output_dir / f"dropped.rank{rank}.jsonl"


def _chunks(items: list, size: int):
    for index in range(0, len(items), size):
        yield items[index : index + size]


def _counts(records: list[dict]) -> dict:
    counts = {"ok": 0, "truncated": 0, "extract_failed": 0, "error": 0}
    for record in records:
        status = record.get("status")
        if status in counts:
            counts[status] += 1
    return counts
