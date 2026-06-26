# Eval Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the lightweight `eval-pipeline` package described in `doc.md`.

**Architecture:** The package is a thin orchestration layer: YAML config and dynamic user task files feed a shared runner, which calls a backend, writes append-only JSONL records, and emits summary statistics. All runtime writes go under the workspace-local `runs/<name>/` directory.

**Tech Stack:** Python standard library plus `PyYAML` for YAML parsing, `pytest` for tests, and optional runtime integrations for `httpx` and `vllm` when those backends are used.

---

### Task 1: IO Primitives

**Files:**
- Create: `eval_pipeline/core/io.py`
- Test: `tests/test_io.py`

- [ ] Write failing tests for append-only JSONL writing, done-key scanning, bad-tail tolerance, and shard merge deduplication.
- [ ] Implement `append_jsonl`, `scan_done`, `read_records`, and `merge_shards`.
- [ ] Run `pytest tests/test_io.py -v`.

### Task 2: Config Loading

**Files:**
- Create: `eval_pipeline/core/config.py`
- Test: `tests/test_config.py`

- [ ] Write failing tests for complete config loading and missing required field failure.
- [ ] Implement dataclasses and `load_config`.
- [ ] Run `pytest tests/test_config.py -v`.

### Task 3: Task Loading

**Files:**
- Create: `eval_pipeline/core/task.py`
- Test: `tests/test_task.py`

- [ ] Write failing tests for dynamic loading from workspace-relative path, missing required function failure, and optional score.
- [ ] Implement `load_task`.
- [ ] Run `pytest tests/test_task.py -v`.

### Task 4: Backend and Runner

**Files:**
- Create: `eval_pipeline/core/backends.py`
- Create: `eval_pipeline/core/runner.py`
- Test: `tests/test_runner.py`

- [ ] Write failing tests for `process_one`, resume skipping, extract exception isolation, and summary writing.
- [ ] Implement `GenResult`, `Backend`, `APIBackend`, `VLLMBackend`, `Record`, `process_one`, `run`, `pilot`, and truncate policy handling.
- [ ] Run `pytest tests/test_runner.py -v`.

### Task 5: Stats

**Files:**
- Create: `eval_pipeline/core/stats.py`
- Test: `tests/test_stats.py`

- [ ] Write failing tests for four-rate summary, threshold comparison, and mean score ignoring `None`.
- [ ] Implement `build_summary`.
- [ ] Run `pytest tests/test_stats.py -v`.

### Task 6: CLI and Templates

**Files:**
- Create: `eval_pipeline/cli.py`
- Create: `eval_pipeline/__main__.py`
- Create: `eval_pipeline/templates/config.yaml`
- Create: `eval_pipeline/templates/task_mcq.py`
- Create: `eval_pipeline/templates/task_generative.py`
- Create: `eval_pipeline/templates/task_datagen.py`
- Test: `tests/test_cli.py`

- [ ] Write failing tests for `init` workspace generation and `show` status filtering.
- [ ] Implement `eval-pipeline init/run/pilot/show/merge`.
- [ ] Run `pytest tests/test_cli.py -v`.

### Task 7: Final Verification

- [ ] Run full `pytest -v`.
- [ ] Run `rg "try:|except " eval_pipeline` and confirm only `process_one` contains try/except.
- [ ] Smoke-test `python -m eval_pipeline.cli init /tmp/eval-pipeline-smoke`.
