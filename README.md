# LightEvaluation

LightEvaluation 是一个轻量级 LLM 评测与数据生成 pipeline。它负责把一批数据送进模型，收集模型输出，抽取答案，按需打分，并把结果写成可恢复、可检查、可复现的文件。

它主要适合两类任务：

- **模型评测**：对 benchmark 题目跑模型，抽取答案，计算分数，统计截断率、抽取失败率、错误率。
- **批量造数据**：对大量 prompt 跑模型，只保存生成文本，不要求打分。

这个项目刻意保持小而薄。它不实现推理引擎，不做 agent，不做工具调用，不做复杂插件系统。它只做这一层编排：

```text
数据 item -> prompt -> 后端生成 -> 答案抽取 -> 可选打分 -> JSONL + summary
```

## 核心思路

项目分成两个世界：

```text
框架代码，也就是这个包
  eval_pipeline/
    core/
    cli.py
    templates/

用户工作区
  config.yaml
  tasks/my_task.py
  data/...
  runs/<run-name>/
```

框架代码安装一次即可。换数据集、换 prompt、换抽取逻辑时，不需要改框架代码，只改用户工作区里的：

- 一个 YAML 配置文件
- 一个任务脚本
- 本地数据文件

所有运行结果都会写到工作区的 `runs/<name>/` 目录下。

## 能做什么

- `eval-pipeline init <dir>`：创建一个起步工作区。
- `eval-pipeline run config.yaml`：执行主采样/评测流程。
- `eval-pipeline pilot config.yaml`：跑一个小 pilot，用来估计截断等问题，不写主输出。
- `eval-pipeline show config.yaml --status error --n 10`：抽查某类状态的样本。
- `eval-pipeline merge config.yaml`：合并并行 rank 生成的分片输出。
- API 后端：调用 OpenAI-compatible `/chat/completions` 服务。
- vLLM 后端：使用本地 vLLM 离线接口，不起 server，不用端口。
- 支持 resume：按 `(id, sample_idx)` 跳过已经完成的样本。
- 支持 batch：一批一批采样，每批落盘。
- 支持进度日志：能看到完成数、速度、ETA、错误统计。
- 支持 rank 并行：多个进程各自跑一片数据，各自写日志和输出。

## 安装

克隆仓库：

```bash
git clone https://github.com/LaugHan/LightEvaluation.git
cd LightEvaluation
```

建议先进入自己的 Python 环境，然后安装：

```bash
pip install -e .
```

如果要用本地 vLLM，在已经配好 CUDA/vLLM 的环境里安装可选依赖：

```bash
pip install -e ".[vllm]"
```

开发和测试：

```bash
pip install -e ".[dev]"
pytest -q
```

安装后应能看到命令：

```bash
eval-pipeline --help
```

## 快速开始

创建一个工作区：

```bash
eval-pipeline init my-experiment
cd my-experiment
```

会生成：

```text
my-experiment/
  config.yaml
  data.jsonl
  tasks/
    task_generative.py
    task_mcq.py
    task_datagen.py
```

编辑 `config.yaml` 后运行：

```bash
eval-pipeline run config.yaml
```

输出会写到：

```text
runs/<name>/
  output.jsonl
  dropped.jsonl
  summary.json
  logs/
    run.jsonl
    progress.jsonl
    state.json
    errors.jsonl      # 只有出现错误、截断或抽取失败样本时才会有
    fatal.log         # 只有整个 run 崩溃时才会有
```

## API 后端

API 后端会调用 OpenAI-compatible `/chat/completions` 接口。API key 从环境变量 `OPENAI_API_KEY` 读取。

先复制示例环境文件：

```bash
cp .env.example .env
```

填写：

```bash
OPENAI_API_KEY=...
OPENAI_BASE_URL=https://your-endpoint/v1
OPENAI_MODEL=your-model
```

配置示例：

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

运行前加载 `.env`：

```bash
set -a
. ./.env
set +a
eval-pipeline run config.yaml
```

### API 并发怎么理解

API 采样有两层并发相关配置：

- `runtime.batch_size`：runner 每次交给后端多少条 prompt。
- `model.concurrency`：一个 batch 内最多同时发多少个 HTTP 请求。

例如：

```yaml
runtime:
  batch_size: 64

model:
  concurrency: 8
```

意思是每个 batch 有 64 条 prompt，其中最多 8 个请求同时在飞。API 后端使用 `asyncio.gather` 和 `Semaphore`，所以 batch 内不是一条一条串行发请求。

## vLLM 后端

vLLM 后端使用本地 `vllm.LLM` 离线接口，在当前进程内推理。它不会启动 HTTP server，也没有端口概念。

本地配置示例：

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

运行：

```bash
CUDA_VISIBLE_DEVICES=0 eval-pipeline run config.yaml
```

如果在共享 GPU 上 vLLM 初始化失败或 OOM，优先尝试：降低 `gpu_memory_utilization`、降低 `max_num_seqs`、设置 `enforce_eager: true`、使用更小的 `max_model_len`，或用 `CUDA_VISIBLE_DEVICES` 指定更空闲的 GPU。

## 任务脚本怎么写

任务脚本就是普通 Python 文件。它必须定义三个函数：

```python
def load(source: str, limit: int | None) -> list[dict]:
    ...

def build_prompt(item: dict, config) -> str:
    ...

def extract_answer(raw: str, item: dict) -> tuple[str | None, str]:
    ...
```

如果是评测任务，再定义一个可选的 `score`：

```python
def score(answer: str | None, item: dict) -> float | None:
    ...
```

如果没有 `score`，pipeline 会把它当成数据生成任务，`summary.json` 里的 `mean_score` 会是 `null`。

`load()` 返回的每个 item 必须有稳定的 `id`。resume 依赖 `(id, sample_idx)`。不要使用会因为过滤、排序、shuffle 而变化的 id。建议自己构造稳定 id，例如 `f"gsm8k-{idx:06d}"`。

一个完整例子：

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

`extract_answer()` 的第二个返回值建议用：`"extracted"`、`"fallback"` 或 `"failed"`。

## 输出文件

### `output.jsonl`

每行是一条样本记录：

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

`status` 可能是：`ok`、`truncated`、`extract_failed`、`error`。

### `summary.json`

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

重点看 `rates`、`within_thresholds` 和 `mean_score`。

## 日志怎么看

日志和样本输出分开，放在 `logs/` 目录：

```text
logs/
  run.jsonl          # 生命周期：开始、加载任务、启动后端、结束或失败
  progress.jsonl     # 周期性进度：完成数、速度、ETA、四率计数
  errors.jsonl       # 抽样记录坏样本，不会记录每个正常样本
  state.json         # 当前状态快照，适合直接 cat
  fatal.log          # 整个 run 崩溃时的完整 traceback
```

并行 rank 会分开写：

```text
logs/
  run.rank0.jsonl
  progress.rank0.jsonl
  state.rank0.json
  run.rank1.jsonl
  progress.rank1.jsonl
  state.rank1.json
```

查看当前状态：

```bash
cat runs/<name>/logs/state.json
```

持续看进度：

```bash
tail -f runs/<name>/logs/progress.jsonl
```

查看生命周期：

```bash
tail -f runs/<name>/logs/run.jsonl
```

`progress.jsonl` 类似：

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

这能帮助你判断程序有没有跑起来、当前跑到多少、速度是多少、预计还要多久，以及错误/截断/抽取失败是否异常升高。

## Resume 机制

pipeline 是 append-only 写入。启动时会扫描已有的 `output*.jsonl` 和 `dropped*.jsonl`，收集已经完成的 `(id, sample_idx)`，本次自动跳过。

如果程序在写 JSONL 时崩溃，最后一行可能是不完整 JSON。resume 扫描时会容忍并截掉这个坏尾行。

## 截断策略

```yaml
output:
  on_truncate: accept  # accept | drop | retry
```

- `accept`：截断样本仍写入 `output.jsonl`，状态是 `truncated`。
- `drop`：截断样本写入 `dropped.jsonl`，仍参与 summary 和 resume，但不进入主输出。
- `retry`：对该样本用 `max_tokens * 2` 重跑一次；如果仍截断，则保留为 `truncated`。

## 并行运行

用 `rank/world-size` 把数据切成多份，多进程并行跑：

```bash
eval-pipeline run config.yaml --rank 0 --world-size 2
eval-pipeline run config.yaml --rank 1 --world-size 2
```

每个 rank 会写自己的输出和日志：

```text
output.rank0.jsonl
output.rank1.jsonl
logs/run.rank0.jsonl
logs/run.rank1.jsonl
```

都完成后合并：

```bash
eval-pipeline merge config.yaml
```

合并会按 `(id, sample_idx)` 去重。这里的并行是真正的进程级并行：每个 rank 是单独进程，有自己的后端请求、输出分片、resume 扫描和日志。

## 仓库内置示例工作区

仓库里包含 `test_workspace/`，用 `GSM8K_RUP.csv` 做示例。

```text
test_workspace/
  data/GSM8K_RUP.csv
  tasks/gsm8k_rup.py
  tasks/gsm8k_rup_datagen.py
  config_api_20_batched_logs.yaml
  config_api_20_parallel_batched_logs.yaml
  config_vllm_20_datagen.yaml
```

API 单进程 smoke：

```bash
set -a
. ./.env
set +a
eval-pipeline run test_workspace/config_api_20_batched_logs.yaml
```

API 两 rank 并行 smoke：

```bash
set -a
. ./.env
set +a
eval-pipeline run test_workspace/config_api_20_parallel_batched_logs.yaml --rank 0 --world-size 2
eval-pipeline run test_workspace/config_api_20_parallel_batched_logs.yaml --rank 1 --world-size 2
eval-pipeline merge test_workspace/config_api_20_parallel_batched_logs.yaml
```

本地 vLLM 数据生成 smoke：

```bash
CUDA_VISIBLE_DEVICES=0 eval-pipeline run test_workspace/config_vllm_20_datagen.yaml
```

示例运行输出 `runs/` 不提交到 git。需要时在本地重新跑。

## 开发

运行测试：

```bash
pytest -q
```

当前测试覆盖：配置加载、task 加载、JSONL append/resume/merge、batch runner、API batch 内并发、日志隔离、fatal logging、rank 分片与 merge、summary 统计。

## 设计边界

本项目刻意不做插件注册系统、配置 DSL、agent/tool-calling 流程、隐藏全局状态，也不会静默吞掉框架错误。

只有单条样本的抽取/打分失败会变成 `status="error"`。框架级或后端级 fatal error 会写日志，然后命令以非 0 退出。
