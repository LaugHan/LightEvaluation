# LightEvaluation

LightEvaluation is a lightweight pipeline for running LLM inference over a dataset, extracting answers, scoring records, and writing reproducible outputs.

It is designed for two common jobs:

- **Evaluation:** run a model on benchmark items, extract answers, compute scores, and summarize error/truncation rates.
- **Data generation:** run a model over many prompts and keep the generated text without requiring a score.

The project deliberately stays small. It does not implement an inference engine, agent system, plugin registry, or multi-turn tool calling. It only orchestrates:

```text
dataset item -> prompt -> backend generation -> answer extraction -> optional score -> JSONL + summary
```

## Core Idea

There are two separate worlds:

```text
Framework package
  eval_pipeline/
    core/
    cli.py
    templates/

User workspace
  config.yaml
  tasks/my_task.py
  data/...
  runs/<run-name>/
```

The framework code is installed once and should not be edited for each dataset. A user changes behavior by editing only:

- one YAML config file
- one task file
- local input data

All outputs are written under the workspace `runs/<name>/` directory.

## What You Get

- `eval-pipeline init <dir>` creates a starter workspace.
- `eval-pipeline run config.yaml` runs inference and writes outputs.
- `eval-pipeline pilot config.yaml` runs a small pilot without writing main output.
- `eval-pipeline show config.yaml --status error --n 10` prints records for inspection.
- `eval-pipeline merge config.yaml` merges rank shard files after parallel runs.
- API backend for OpenAI-compatible chat completion servers.
- vLLM offline backend for local models.
- Resume by `(id, sample_idx)` from append-only JSONL outputs.
- Batch execution with progress logs, speed, ETA, and isolated error logs.
- Rank-based parallel execution with separate shard outputs and logs.

## Installation

Clone the repository:

```bash
git clone https://github.com/LaugHan/LightEvaluation.git
cd LightEvaluation
```

Create or activate a Python environment, then install:

```bash
pip install -e .
```

For local vLLM execution, install the optional vLLM dependencies in an environment that already supports your CUDA setup:

```bash
pip install -e ".[vllm]"
```

For development and tests:

```bash
pip install -e ".[dev]"
pytest -q
```

## Quick Start

Create a workspace:

```bash
eval-pipeline init my-experiment
cd my-experiment
```

This creates:

```text
my-experiment/
  config.yaml
  data.jsonl
  tasks/
    task_generative.py
    task_mcq.py
    task_datagen.py
```

Edit `config.yaml`, then run:

```bash
eval-pipeline run config.yaml
```

Outputs are written to:

```text
runs/<name>/
  output.jsonl
  dropped.jsonl
  summary.json
  logs/
    run.jsonl
    progress.jsonl
    state.json
    errors.jsonl      # only when warnings/errors are sampled
    fatal.log         # only when the whole run crashes
```

## API Backend

The API backend calls an OpenAI-compatible `/chat/completions` endpoint. It reads the API key from `OPENAI_API_KEY`.

Create a local `.env` from the example:

```bash
cp .env.example .env
```

Fill in:

```bash
OPENAI_API_KEY=...
OPENAI_BASE_URL=https://your-endpoint/v1
OPENAI_MODEL=your-model
```

Example config:

```yaml
name: gsm8k_api_run

model:
  backend: api
  base_url: https://your-endpoint/v1
  model: your-model
  concurrency: 8
  max_retries: 3

dataset:
  task_file: tasks/gsm8k.py
  source: data/GSM8K_RUP.csv
  limit: 20
  n_samples: 1

generation:
  max_tokens: 256
  temperature: 0.0
  stop: null
  seed: 0

output:
  on_truncate: accept
  flush_every: 1

thresholds:
  truncated: 0.2
  extract_failed: 0.2
  error: 0.2

runtime:
  batch_size: 64

logging:
  console: true
  progress_every: 100
  progress_interval_sec: 30
  error_preview_chars: 300
  sample_bad_records: 20
```

Run with environment variables loaded:

```bash
set -a
. ./.env
set +a
eval-pipeline run config.yaml
```

### API Concurrency

API parallelism has two levels:

- `runtime.batch_size`: how many pending records are handed to the backend at once.
- `model.concurrency`: how many HTTP requests are in flight inside that batch.

For example, `batch_size: 64` and `concurrency: 8` means each batch contains 64 prompts and up to 8 requests are sent concurrently. The API backend uses `asyncio.gather` plus a semaphore, so requests inside a batch are not sent one by one.

## vLLM Backend

The vLLM backend uses the offline `vllm.LLM` interface in-process. It does not start an HTTP server and does not use ports.

Example local config:

```yaml
name: local_vllm_run

model:
  backend: vllm
  path: /ssd/models/Qwen/Qwen2___5-0___5B-Instruct
  data_parallel_size: 1
  tensor_parallel_size: 1
  max_model_len: 2048
  gpu_memory_utilization: 0.2
  dtype: auto
  max_num_seqs: 8
  enforce_eager: true

dataset:
  task_file: tasks/gsm8k_datagen.py
  source: data/GSM8K_RUP.csv
  limit: 20
  n_samples: 1

generation:
  max_tokens: 64
  temperature: 0.0
  stop:
    - "."
  seed: 0

output:
  on_truncate: accept
  flush_every: 1

thresholds:
  truncated: 0.2
  extract_failed: 0.2
  error: 0.2

runtime:
  batch_size: 128

logging:
  console: true
  progress_every: 100
  progress_interval_sec: 30
  error_preview_chars: 300
  sample_bad_records: 20
```

Then run:

```bash
CUDA_VISIBLE_DEVICES=0 eval-pipeline run config.yaml
```

If vLLM fails during initialization on a shared GPU, reduce memory pressure:

- lower `gpu_memory_utilization`
- lower `max_num_seqs`
- set `enforce_eager: true`
- use a smaller `max_model_len`
- isolate the process on a less busy GPU with `CUDA_VISIBLE_DEVICES`

## Task File Contract

A task file is ordinary Python. It must define these functions:

```python
def load(source: str, limit: int | None) -> list[dict]:
    ...

def build_prompt(item: dict, config) -> str:
    ...

def extract_answer(raw: str, item: dict) -> tuple[str | None, str]:
    ...
```

For evaluation, also define:

```python
def score(answer: str | None, item: dict) -> float | None:
    ...
```

`score` is optional. If omitted, the pipeline behaves like data generation and `mean_score` is `null`.

Every item returned by `load()` must contain a stable `id`. Do not use a changing row number after filtering or shuffling unless you deliberately make it stable. Resume uses:

```text
(id, sample_idx)
```

Example:

```python
import csv
import re

def load(source: str, limit: int | None) -> list[dict]:
    rows = []
    with open(source, newline="", encoding="utf-8") as handle:
        for idx, row in enumerate(csv.DictReader(handle)):
            rows.append({
                "id": f"gsm8k-{idx:06d}",
                "question": row["question"],
                "gold": row["answer"].split("####")[-1].strip(),
            })
            if limit is not None and len(rows) >= limit:
                return rows
    return rows

def build_prompt(item: dict, config) -> str:
    return f"Solve the problem and end with #### <answer>.\n\n{item['question']}"

def extract_answer(raw: str, item: dict) -> tuple[str | None, str]:
    match = re.search(r"####\s*([-+]?\d+(?:\.\d+)?)", raw)
    if match:
        return match.group(1), "extracted"
    return None, "failed"

def score(answer: str | None, item: dict) -> float | None:
    if answer is None:
        return None
    return 1.0 if answer == item["gold"] else 0.0
```

`extract_answer` should return one of:

- `"extracted"`: normal extraction
- `"fallback"`: extraction succeeded by a weaker fallback rule
- `"failed"`: extraction failed; return `None`

## Output Files

### `output.jsonl`

Each line is a record:

```json
{
  "id": "gsm8k-000001",
  "sample_idx": 0,
  "status": "ok",
  "prompt": "...",
  "raw": "...",
  "finish_reason": "stop",
  "answer": "42",
  "extract_status": "extracted",
  "gold": "42",
  "score": 1.0
}
```

Statuses:

- `ok`: normal completion
- `truncated`: model hit `max_tokens`
- `extract_failed`: generation succeeded but answer extraction failed
- `error`: item-level processing failed

### `summary.json`

Example:

```json
{
  "name": "gsm8k_api_run",
  "total": 1000,
  "rates": {
    "ok": 0.955,
    "truncated": 0.031,
    "extract_failed": 0.012,
    "error": 0.002
  },
  "thresholds": {
    "truncated": 0.05,
    "extract_failed": 0.02,
    "error": 0.005
  },
  "within_thresholds": true,
  "mean_score": 0.78
}
```

## Logs

Logs are intentionally separated from sample outputs.

```text
logs/
  run.jsonl          # lifecycle: start, task loaded, backend start, finish/fail
  progress.jsonl     # periodic progress with speed and ETA
  errors.jsonl       # sampled bad records only
  state.json         # current state snapshot
  fatal.log          # full traceback for fatal run failures
```

For rank runs:

```text
logs/
  run.rank0.jsonl
  progress.rank0.jsonl
  state.rank0.json
  run.rank1.jsonl
  progress.rank1.jsonl
  state.rank1.json
```

To check a running job:

```bash
cat runs/<name>/logs/state.json
tail -f runs/<name>/logs/progress.jsonl
tail -f runs/<name>/logs/run.jsonl
```

Progress entries look like:

```json
{
  "event": "progress",
  "completed": 5000,
  "total": 20000,
  "pending": 15000,
  "percent": 25.0,
  "items_per_sec": 12.4,
  "eta_sec": 1209.6,
  "ok": 4980,
  "truncated": 10,
  "extract_failed": 8,
  "error": 2
}
```

## Resume

Runs are append-only. On startup, the pipeline scans existing `output*.jsonl` and `dropped*.jsonl`, collects completed `(id, sample_idx)` keys, and skips them.

If the process crashes during a write, a malformed final JSONL tail line is ignored and truncated during resume scanning.

## Truncation Policy

Set in config:

```yaml
output:
  on_truncate: accept  # accept | drop | retry
```

- `accept`: keep the truncated record in `output.jsonl`.
- `drop`: write it to `dropped.jsonl`; it still counts in summary and resume.
- `retry`: retry that one record once with `max_tokens * 2`; if still truncated, keep it as truncated.

## Parallel Runs

Use rank/world-size to split the work across multiple processes:

```bash
eval-pipeline run config.yaml --rank 0 --world-size 2
eval-pipeline run config.yaml --rank 1 --world-size 2
```

Each rank writes separate outputs:

```text
output.rank0.jsonl
output.rank1.jsonl
logs/run.rank0.jsonl
logs/run.rank1.jsonl
```

After both finish:

```bash
eval-pipeline merge config.yaml
```

This writes merged `output.jsonl` and de-duplicates by `(id, sample_idx)`.

This is real process-level parallelism: each rank is a separate process with its own backend calls, output shard, resume scan, and logs.

## Included Example Workspace

This repository includes `test_workspace/` as a working example using `GSM8K_RUP.csv`.

Useful files:

```text
test_workspace/
  data/GSM8K_RUP.csv
  tasks/gsm8k_rup.py
  tasks/gsm8k_rup_datagen.py
  config_api_20_batched_logs.yaml
  config_api_20_parallel_batched_logs.yaml
  config_vllm_20_datagen.yaml
```

API smoke:

```bash
set -a
. ./.env
set +a
eval-pipeline run test_workspace/config_api_20_batched_logs.yaml
```

Two-rank API smoke:

```bash
set -a
. ./.env
set +a
eval-pipeline run test_workspace/config_api_20_parallel_batched_logs.yaml --rank 0 --world-size 2
eval-pipeline run test_workspace/config_api_20_parallel_batched_logs.yaml --rank 1 --world-size 2
eval-pipeline merge test_workspace/config_api_20_parallel_batched_logs.yaml
```

Local vLLM data-generation smoke:

```bash
CUDA_VISIBLE_DEVICES=0 eval-pipeline run test_workspace/config_vllm_20_datagen.yaml
```

The example `runs/` outputs are intentionally ignored by git. Re-run them locally when needed.

## Development

Run tests:

```bash
pytest -q
```

Current tests cover:

- config loading and fail-loud missing fields
- workspace-contained task loading
- JSONL append/resume/merge
- batch runner behavior
- API batch concurrency
- isolated logs and fatal logging
- rank sharding and merge
- summary statistics

## Design Constraints

The project intentionally avoids:

- plugin registries
- config DSLs
- agent/tool-calling workflows
- hidden global state
- swallowing framework errors

Only item-level extraction/scoring failures are converted into `status="error"` records. Fatal framework/backend failures are logged and the command exits non-zero.

