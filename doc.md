# 轻量级 LLM 评估 / 数据生成 Pipeline 设计规格

> 一份**实现规格**，交给 coding agent 照此实现；同时也是最终用户的使用契约。
> 实现者请严格遵守 §4「设计原则」。

---

## 1. 这是什么

一个**薄编排层**，把"对一批输入跑 LLM 推理、提取答案、打分/落盘"固化成可复用基础设施。
两个用途：(a) 文本模型评估；(b) 为下游任务批量造数据。**不涉及 agent / 多轮 / 工具调用。**

它不自己造推理引擎：本地多卡用 vLLM 离线接口，远程模型用 asyncio 并发。本项目只拥有那层编排壳。

---

## 2. 核心设计：两个世界

整个设计的灵魂是一条隔离线：

- **框架（the package）**：安装一次、用户从不修改的代码。提供 `eval-pipeline` 命令、Task 契约、后端、Runner、统计。
- **用户工作区（the workspace）**：一个目录，装下用户碰的**所有**东西——一份 YAML、一个（或几个）任务文件、以及全部输出。

铁律：

> **框架代码不依赖任何用户文件；框架运行时不向工作区以外写一个字节。**
> 用户想改行为，只改工作区里的 YAML 和任务文件，永不碰框架。

这条线让"别人怎么用"变得显然：拿到框架 → 建一个工作区目录 → 写 YAML + 任务文件 → 跑 → 输出就在同一目录下。删掉工作区，框架毫发无伤；换台机器，把工作区拷过去即可复现。

---

## 3. 用户使用流程（最终用户视角）

用户只准备**两样东西**，放在自己的工作区目录里：

**① 一个 YAML** —— 所有"声明式、跨数据集通用"的旋钮：用什么模型、怎么起、哪个数据集、跑多少题、每题几遍、生成参数、阈值。

**② 一个任务文件 `task.py`** —— 所有"这个数据集独有"的逻辑：怎么读数据、怎么拼 prompt、怎么抽答案、怎么判分。

然后：

```bash
eval-pipeline run config.yaml
```

拿到 `runs/<name>/output.jsonl` 和 `runs/<name>/summary.json`。完事。

**关键认知**：数据集千差万别，但最终都收敛成"一段 prompt 进、一个 answer 出、一个分数"。所有差异被任务文件的几个函数吸收，**框架本身永远不动**。换数据集 = 换任务文件 + 改几个数字。

---

## 4. 设计原则（硬约束，违反即返工）

优先级高于"代码健壮性"的本能。

1. **轻量优先。** 框架核心控制在数百行量级。不要框架化的插件注册系统、抽象工厂、配置 DSL。能用函数不用类，能用类不用继承。

2. **禁止防御性编程堆砌：**
   - 不在函数入口塞参数校验 / `isinstance` / `assert` 墙。
   - 不到处 `try/except` 然后吞异常返默认值。
   - `try/except` **只允许出现在一处**：单条 item 的处理边界（§9），用于隔离坏数据。其余地方一律不加。
   - 不为"理论可能但实际不会发生"的情况写分支。

3. **fail loud, fail early。** 配置缺字段、路径不存在、后端起不来、任务文件缺函数 —— 直接抛异常退出，不 fallback。只有**数据级**错误（某条推理失败、提取失败）才记录并继续。

4. **可观测 > 正确。** 不追求消灭提取错误 / 截断，要求它们可计数、可打印、可抽查（§13）。

5. **配置集中、无隐藏状态。** 所有旋钮进 YAML/Config 对象；并发数、顺序、随机种子的处理必须可复现。

---

## 5. 目录结构

**框架（安装的包，用户不碰）：**

```
eval_pipeline/
  core/
    runner.py      # 编排：load→skip→backend→process_one→流式写→summary
    backends.py    # VLLMBackend, APIBackend, GenResult
    task.py        # Task 契约定义 + 从工作区动态加载任务文件
    io.py          # jsonl 流式读写、done 集合、坏尾行容忍、merge
    config.py      # 从 YAML 加载到 Config 对象，缺字段即报错
    stats.py       # 四率统计 + summary
  cli.py           # run / pilot / show / merge
  templates/       # 给用户拷贝的起步模板
    config.yaml
    task_mcq.py        # 多选 (loglikelihood) 示例
    task_generative.py # 生成抽取 (GSM8K 式) 示例
    task_datagen.py    # 造数据 (无 score) 示例
  tests/
    test_io.py
    test_runner.py
```

**用户工作区（一个目录，装下全部用户内容 + 输出）：**

```
my-experiment/
  config.yaml          # 用户写
  tasks/
    gsm8k.py           # 用户写（从模板拷改）
  runs/                # 框架写，全部输出在此
    gsm8k_my-model/
      output.jsonl
      output.rank0.jsonl ...   # 多卡分片，merge 后成 output.jsonl
      summary.json
```

`eval-pipeline init <dir>` 可一键生成一个含 `config.yaml` 和示例任务的空工作区。

---

## 6. YAML Schema（完整字段）

```yaml
name: gsm8k_my-model          # run 名，决定 runs/<name>/ 输出目录

model:
  backend: vllm               # "vllm" | "api"
  # --- backend=vllm 时 ---
  path: /data/models/my-model # 模型目录（本地）
  data_parallel_size: 4       # 几张卡跑几个模型副本、切数据（最快路径）
  tensor_parallel_size: 1     # 单卡装不下才 >1
  # 不需要端口！vLLM 离线接口在进程内跑，见 §8
  # --- backend=api 时 ---
  # base_url: http://host:8000/v1
  # model: gpt-4o-mini
  # concurrency: 16
  # max_retries: 5

dataset:
  task_file: tasks/gsm8k.py   # ← 用户自定义任务文件（相对工作区）
  source: /data/gsm8k/test.jsonl
  limit: 200                  # 跑多少题（null = 全部）
  n_samples: 5                # 每题跑几遍

generation:
  max_tokens: 1024
  temperature: 0.7
  stop: null                  # 可选停止序列
  seed: 0

output:
  on_truncate: drop           # "accept" | "drop" | "retry"
  flush_every: 1              # 每几条 flush 一次

thresholds:                   # run 前声明的容忍上限，结束对照
  truncated: 0.05
  extract_failed: 0.02
  error: 0.005
```

加载到一个 `Config` dataclass。缺必填字段直接报错退出，不给默认兜底。

---

## 7. Task 契约（用户在任务文件里实现）

每个数据集一个文件，实现这四个**纯函数**。框架按名字查找；缺任何一个（`score` 除外）即报错退出。

```python
# tasks/gsm8k.py

def load(source: str, limit: int | None) -> list[dict]:
    """读数据集 -> list[dict]。每个 dict 必须含 'id'；评估场景还需 'gold'。
    数据集格式的全部差异在这里吸收。"""

def build_prompt(item: dict, config) -> str:
    """item -> 发给模型的最终字符串（含 few-shot / chat template）。纯函数。"""

def extract_answer(raw: str, item: dict) -> tuple[str | None, str]:
    """原始输出 -> (answer, status)。
    status ∈ {"extracted", "fallback", "failed"}。失败返回 (None, "failed")，不抛异常。"""

def score(answer: str | None, item: dict) -> float | None:
    """(答案, item) -> 分数。造数据场景可整个省略此函数，或返回 None 表示不打分。
    无法判分返回 None，不抛异常。"""
```

约定：`item` 含 `id`（resume 的 key）；`gold` 评估时必需、造数据时可缺省。

---

## 8. 后端接口（框架内，可替换）

后端唯一职责：**一批 prompt -> 一批结果**，对任务一无所知。

```python
@dataclass
class GenResult:
    text: str
    finish_reason: str        # "stop" | "length" | "error"，length 即截断

class Backend(Protocol):
    def generate(self, prompts: list[str], config) -> list[GenResult]: ...
```

**`VLLMBackend`（推荐，你的场景）**：用 vLLM 的离线 `LLM` 类在**进程内**批量推理。
- 多卡纯数据并行：`data_parallel_size=N`（一卡一模型、切数据，最快）。
- 单卡装不下：叠 `tensor_parallel_size`。
- **没有端口概念**：离线接口不起 server，多卡由 vLLM 内部调度，框架不感知端口。batch 交给 vLLM 自动（continuous batching）。

**`APIBackend`**：打远程 OpenAI 兼容服务。`asyncio + Semaphore(concurrency)` 控并发 + 指数退避重试。重试耗尽的那条返回 `finish_reason="error"`，**不抛出**。这里才有 `base_url`。

> 两种后端共用上层 Runner 的 resume / 写入 / 统计。**不要为每种后端各写一套编排。**

---

## 9. 单条处理边界（唯一允许 try/except 的地方）

```python
def process_one(item, gen: GenResult, task, config) -> Record:
    try:
        answer, ex_status = task.extract_answer(gen.text, item)
        sc = task.score(answer, item) if hasattr(task, "score") else None
        status = resolve_status(gen.finish_reason, ex_status)
        return Record(id=item["id"], status=status, answer=answer,
                      score=sc, extract_status=ex_status, raw=gen.text,
                      finish_reason=gen.finish_reason, gold=item.get("gold"))
    except Exception as e:
        # 一条坏数据不许杀整个 run
        return Record(id=item["id"], status="error", raw=gen.text, error=repr(e))
```

`resolve_status` 优先级：`error > truncated > extract_failed > ok`。
**除此之外不要包 try/except。**

---

## 10. Record Schema（output.jsonl，每行一条）

```json
{
  "id": "stable-id",
  "sample_idx": 0,
  "status": "ok",
  "prompt": "...",
  "raw": "...",
  "finish_reason": "stop",
  "answer": "extracted or null",
  "extract_status": "extracted",
  "gold": "ground truth or null",
  "score": 1.0
}
```

`status` 枚举（item 级）：`ok` / `truncated`（finish_reason==length）/ `extract_failed`（推理成功但抽取失败）/ `error`（推理异常或重试耗尽）。

---

## 11. Resume + 流式写入 + 分片（一个契约统一三件事）

resume、防丢、错误隔离不是三套机制，由"**append-only 写带状态的 JSONL**"一并落地：

1. **启动先扫 done 集合**：读已有 `output.jsonl`，收集已完成的 `(id, sample_idx)`，本次跳过。
2. **写一条 flush 一条**（或按 `flush_every`）。崩溃最多丢正在算的一小批。
3. **容忍坏尾行**：扫 done 时若最后一行 JSON 解析失败（崩在写一半），丢弃该行，不报错。
4. **id 必须稳定**：用任务 `load` 产出的 `id`。**禁止用行号/下标当 id**（resume 会错位）。
5. **多卡分片各写各文件**：`output.rank{k}.jsonl`，各自维护 done、各自流式写；某卡挂只影响该片。结束 `merge` 按 id 合成 `output.jsonl`。

---

## 12. 截断策略 `on_truncate`

- `accept`：照常记录 `status=truncated`，分数照打（评估场景，统一即公平）。
- `drop`：不写进最终输出（造数据场景，截断样本是废数据）。
- `retry`：用 ×2 `max_tokens` 重跑该条一次，仍截断则按 accept 记。

无论哪种，截断率都进 summary。`pilot` 模式跑前 N 条只估截断率、用于定 `max_tokens`，不写主输出。

---

## 13. 错误容忍：四率 + summary

**立场（写进 README）**：评测错误的解药不是"零错误"，是"**可观测 + 预先声明的容忍阈值**"。run 前声明阈值，run 中统计，结束对照；落在阈值内即视为有效，无需纠结个案。

收尾必写 `summary.json`：

```json
{
  "name": "gsm8k_my-model",
  "total": 1000,
  "rates": { "ok": 0.955, "truncated": 0.031, "extract_failed": 0.012, "error": 0.002 },
  "thresholds": { "truncated": 0.05, "extract_failed": 0.02, "error": 0.005 },
  "within_thresholds": true,
  "mean_score": 0.78,
  "config": { "...": "完整 config 快照，保复现" }
}
```

抽查入口：`eval-pipeline show --status extract_failed --n 10`，按状态过滤打印若干 Record 供人工核对。**只打印，不做花哨 TUI。**

---

## 14. CLI 子命令

```
eval-pipeline init  <dir>          # 生成空工作区（config 模板 + 示例任务）
eval-pipeline run   config.yaml    # 主流程
eval-pipeline pilot config.yaml    # 跑前 N 条估截断率，不写主输出
eval-pipeline show  config.yaml --status <s> --n <k>   # 抽查 record
eval-pipeline merge config.yaml    # 合并多卡分片
```

---

## 15. 实现顺序与验收清单

按序实现，每步可独立验证：

1. `io.py` + `test_io.py`：流式写、扫 done、坏尾行容忍、merge。**先有测试。**
2. `config.py`：YAML 加载、缺字段报错。
3. `task.py`：从工作区路径动态加载任务文件、校验必需函数存在。
4. `backends.py`：先 `APIBackend`（好测），再 `VLLMBackend`。
5. `runner.py`：串起来，含 `process_one` 单点 try/except。
6. `stats.py` + summary。
7. `cli.py` + `templates/`。

**验收清单：**

- [ ] `init` 出的工作区，改两行就能 `run` 跑通。
- [ ] 中途 kill 后重跑，跳过已完成、补齐剩余，不重算、不丢数据。
- [ ] 任务 `extract_answer` 故意抛异常，run 不崩，该条记 `status=error`。
- [ ] `summary.json` 四率与 `within_thresholds` 判定正确。
- [ ] 多卡分片各写各文件，`merge` 后总数正确、无重复 id。
- [ ] 全局 `try/except` 只在 `process_one` 一处（grep 验证）。
- [ ] 框架运行时不向工作区以外写文件（验证隔离）。
- [ ] 框架核心无参数校验墙、无静默 fallback。

---

## 附：明确不做的事

- 不做 agent / 多轮 / 工具调用。
- 不自己实现推理引擎（多卡靠 vLLM 离线 `LLM`，远程靠 asyncio）。
- 不为 vLLM 离线模式引入端口/server 概念。
- 不做插件注册系统 / 配置 DSL / 抽象工厂。
- `score` 是简单函数；复杂指标在用户任务文件里各自处理，框架不管。
- 不追求消灭提取/截断错误，只追求可观测与阈值声明。