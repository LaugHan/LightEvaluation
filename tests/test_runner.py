import json

from eval_pipeline.core.backends import GenResult
from eval_pipeline.core.config import load_config
from eval_pipeline.core.runner import process_one, run


class StaticBackend:
    def __init__(self, results):
        self.results = results
        self.prompts = []

    def generate(self, prompts, config):
        self.prompts.extend(prompts)
        return self.results[: len(prompts)]


class BatchRecordingBackend:
    def __init__(self):
        self.batch_sizes = []

    def generate(self, prompts, config):
        self.batch_sizes.append(len(prompts))
        return [GenResult(prompt, "stop") for prompt in prompts]


class SequenceBackend:
    def __init__(self, results):
        self.results = list(results)

    def generate(self, prompts, config):
        out = self.results[: len(prompts)]
        self.results = self.results[len(prompts) :]
        return out


class CountingBackend:
    def __init__(self, result):
        self.result = result
        self.calls = 0
        self.prompts = []

    def generate(self, prompts, config):
        self.calls += 1
        self.prompts.extend(prompts)
        return [self.result for _ in prompts]


class RetryBackend:
    def __init__(self):
        self.max_tokens = []

    def generate(self, prompts, config):
        self.max_tokens.append(config.generation.max_tokens)
        if len(self.max_tokens) == 1:
            return [GenResult("too long", "length")]
        return [GenResult("fixed", "stop")]


def write_workspace(tmp_path, task_body):
    (tmp_path / "tasks").mkdir()
    (tmp_path / "tasks" / "demo.py").write_text(task_body)
    (tmp_path / "data.jsonl").write_text('{"id": "a", "gold": "A"}\n{"id": "b", "gold": "B"}\n')
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
name: demo_run
model:
  backend: api
  base_url: http://localhost:8000/v1
  model: demo
  concurrency: 2
  max_retries: 1
dataset:
  task_file: tasks/demo.py
  source: data.jsonl
  limit: null
  n_samples: 1
generation:
  max_tokens: 16
  temperature: 0.0
  stop: null
  seed: 0
output:
  on_truncate: accept
  flush_every: 1
thresholds:
  truncated: 1.0
  extract_failed: 1.0
  error: 1.0
""".lstrip()
    )
    return config_path


def test_process_one_isolates_extract_exceptions_as_item_errors():
    class Task:
        def extract_answer(self, raw, item):
            raise ValueError("bad item")

        def score(self, answer, item):
            return 1.0

    record = process_one({"id": "a"}, 0, "prompt", GenResult("raw", "stop"), Task(), None)

    assert record["id"] == "a"
    assert record["sample_idx"] == 0
    assert record["status"] == "error"
    assert "ValueError" in record["error"]


def test_process_one_isolates_score_exceptions_as_item_errors():
    class Task:
        def extract_answer(self, raw, item):
            return "answer", "extracted"

        def score(self, answer, item):
            raise RuntimeError("bad score")

    record = process_one({"id": "a"}, 0, "prompt", GenResult("raw", "stop"), Task(), None)

    assert record["status"] == "error"
    assert "RuntimeError" in record["error"]


def test_process_one_resolves_status_priority():
    class Task:
        def extract_answer(self, raw, item):
            return None, "failed"

    assert process_one({"id": "a"}, 0, "p", GenResult("", "error"), Task(), None)["status"] == "error"
    assert process_one({"id": "a"}, 0, "p", GenResult("", "length"), Task(), None)["status"] == "truncated"
    assert process_one({"id": "a"}, 0, "p", GenResult("", "stop"), Task(), None)["status"] == "extract_failed"


def test_run_writes_output_summary_and_skips_done_records(tmp_path):
    task_body = """
import json

def load(source, limit):
    rows = [json.loads(line) for line in open(source)]
    return rows[:limit] if limit is not None else rows

def build_prompt(item, config):
    return "Q:" + item["id"]

def extract_answer(raw, item):
    return raw, "extracted"

def score(answer, item):
    return 1.0 if answer else None
""".lstrip()
    config_path = write_workspace(tmp_path, task_body)
    config = load_config(config_path)
    output_dir = tmp_path / "runs" / "demo_run"
    output_dir.mkdir(parents=True)
    (output_dir / "output.jsonl").write_text(
        json.dumps({"id": "a", "sample_idx": 0, "status": "ok", "score": 1.0}) + "\n"
    )
    backend = StaticBackend([GenResult("B", "stop")])

    summary = run(config, backend=backend)

    records = [json.loads(line) for line in (output_dir / "output.jsonl").read_text().splitlines()]
    assert [record["id"] for record in records] == ["a", "b"]
    assert backend.prompts == ["Q:b"]
    assert summary["total"] == 2
    assert (output_dir / "summary.json").exists()


def test_run_drops_truncated_records_from_output_but_counts_them_in_summary(tmp_path):
    task_body = """
import json

def load(source, limit):
    return [json.loads(line) for line in open(source)]

def build_prompt(item, config):
    return item["id"]

def extract_answer(raw, item):
    return raw, "extracted"
""".lstrip()
    config_path = write_workspace(tmp_path, task_body)
    text = config_path.read_text().replace("on_truncate: accept", "on_truncate: drop")
    config_path.write_text(text)
    config = load_config(config_path)
    backend = StaticBackend([GenResult("too long", "length"), GenResult("ok", "stop")])

    summary = run(config, backend=backend)

    output = tmp_path / "runs" / "demo_run" / "output.jsonl"
    records = [json.loads(line) for line in output.read_text().splitlines()]
    assert [record["id"] for record in records] == ["b"]
    dropped = tmp_path / "runs" / "demo_run" / "dropped.jsonl"
    dropped_records = [json.loads(line) for line in dropped.read_text().splitlines()]
    assert [record["id"] for record in dropped_records] == ["a"]
    assert summary["rates"]["truncated"] == 0.5


def test_run_uses_dropped_records_as_resume_tombstones(tmp_path):
    task_body = """
import json

def load(source, limit):
    return [json.loads(line) for line in open(source)][:1]

def build_prompt(item, config):
    return item["id"]

def extract_answer(raw, item):
    return raw, "extracted"
""".lstrip()
    config_path = write_workspace(tmp_path, task_body)
    config_path.write_text(config_path.read_text().replace("on_truncate: accept", "on_truncate: drop"))
    config = load_config(config_path)
    first_backend = CountingBackend(GenResult("too long", "length"))
    run(config, backend=first_backend)
    second_backend = CountingBackend(GenResult("should not run", "stop"))

    summary = run(config, backend=second_backend)

    assert first_backend.prompts == ["a"]
    assert second_backend.prompts == []
    assert summary["rates"]["truncated"] == 1.0


def test_run_retries_truncated_record_once_with_doubled_max_tokens(tmp_path):
    task_body = """
import json

def load(source, limit):
    return [json.loads(line) for line in open(source)][:1]

def build_prompt(item, config):
    return item["id"]

def extract_answer(raw, item):
    return raw, "extracted"
""".lstrip()
    config_path = write_workspace(tmp_path, task_body)
    config_path.write_text(config_path.read_text().replace("on_truncate: accept", "on_truncate: retry"))
    config = load_config(config_path)
    backend = RetryBackend()

    summary = run(config, backend=backend)

    records = [
        json.loads(line)
        for line in (tmp_path / "runs" / "demo_run" / "output.jsonl").read_text().splitlines()
    ]
    assert records[0]["status"] == "ok"
    assert records[0]["raw"] == "fixed"
    assert backend.max_tokens == [16, 32]
    assert config.generation.max_tokens == 16
    assert summary["rates"]["truncated"] == 0.0


def test_run_retry_records_truncated_when_second_attempt_still_truncates(tmp_path):
    task_body = """
import json

def load(source, limit):
    return [json.loads(line) for line in open(source)][:1]

def build_prompt(item, config):
    return item["id"]

def extract_answer(raw, item):
    return raw, "extracted"
""".lstrip()
    config_path = write_workspace(tmp_path, task_body)
    config_path.write_text(config_path.read_text().replace("on_truncate: accept", "on_truncate: retry"))
    config = load_config(config_path)
    backend = SequenceBackend([GenResult("too long", "length"), GenResult("still too long", "length")])

    summary = run(config, backend=backend)

    records = [
        json.loads(line)
        for line in (tmp_path / "runs" / "demo_run" / "output.jsonl").read_text().splitlines()
    ]
    assert records[0]["status"] == "truncated"
    assert summary["rates"]["truncated"] == 1.0


def test_run_resume_key_includes_sample_idx_for_multiple_samples(tmp_path):
    task_body = """
import json

def load(source, limit):
    return [json.loads(line) for line in open(source)]

def build_prompt(item, config):
    return item["id"]

def extract_answer(raw, item):
    return raw, "extracted"
""".lstrip()
    config_path = write_workspace(tmp_path, task_body)
    config_path.write_text(config_path.read_text().replace("n_samples: 1", "n_samples: 2"))
    config = load_config(config_path)
    output_dir = tmp_path / "runs" / "demo_run"
    output_dir.mkdir(parents=True)
    (output_dir / "output.jsonl").write_text(
        json.dumps({"id": "a", "sample_idx": 0, "status": "ok", "score": None}) + "\n"
    )
    backend = StaticBackend([GenResult("a1", "stop"), GenResult("b0", "stop"), GenResult("b1", "stop")])

    run(config, backend=backend)

    records = [json.loads(line) for line in (output_dir / "output.jsonl").read_text().splitlines()]
    assert [(record["id"], record["sample_idx"]) for record in records] == [
        ("a", 0),
        ("a", 1),
        ("b", 0),
        ("b", 1),
    ]


def test_run_rank_writes_rank_file_and_processes_only_rank_slice(tmp_path):
    task_body = """
import json

def load(source, limit):
    return [json.loads(line) for line in open(source)]

def build_prompt(item, config):
    return item["id"]

def extract_answer(raw, item):
    return raw, "extracted"
""".lstrip()
    config_path = write_workspace(tmp_path, task_body)
    config = load_config(config_path)
    backend = StaticBackend([GenResult("B", "stop")])

    run(config, backend=backend, rank=1, world_size=2)

    output = tmp_path / "runs" / "demo_run" / "output.rank1.jsonl"
    records = [json.loads(line) for line in output.read_text().splitlines()]
    assert backend.prompts == ["b"]
    assert [(record["id"], record["sample_idx"]) for record in records] == [("b", 0)]


def test_run_writes_observable_log_file(tmp_path):
    task_body = """
import json

def load(source, limit):
    return [json.loads(line) for line in open(source)][:1]

def build_prompt(item, config):
    return item["id"]

def extract_answer(raw, item):
    return raw, "extracted"
""".lstrip()
    config_path = write_workspace(tmp_path, task_body)
    config_path.write_text(
        config_path.read_text()
        + """
runtime:
  batch_size: 1
logging:
  console: false
  progress_every: 1
  progress_interval_sec: 999
  error_preview_chars: 80
  sample_bad_records: 20
"""
    )
    config = load_config(config_path)

    run(config, backend=BatchRecordingBackend())

    log = tmp_path / "runs" / "demo_run" / "logs" / "run.jsonl"
    events = [json.loads(line) for line in log.read_text().splitlines()]
    assert events[0]["event"] == "run_start"
    assert events[-1]["event"] == "run_finish"
    assert events[0]["pending"] == 1
    assert events[-1]["total"] == 1
    assert (tmp_path / "runs" / "demo_run" / "logs" / "state.json").exists()


def test_run_batches_generation_and_writes_progress_state(tmp_path):
    task_body = """
import json

def load(source, limit):
    return [json.loads(line) for line in open(source)]

def build_prompt(item, config):
    return item["id"]

def extract_answer(raw, item):
    return raw, "extracted"
""".lstrip()
    config_path = write_workspace(tmp_path, task_body)
    config_path.write_text(
        config_path.read_text()
        + """
runtime:
  batch_size: 1
logging:
  console: false
  progress_every: 1
  progress_interval_sec: 999
  error_preview_chars: 80
  sample_bad_records: 20
"""
    )
    config = load_config(config_path)
    backend = BatchRecordingBackend()

    run(config, backend=backend)

    output_dir = tmp_path / "runs" / "demo_run"
    assert backend.batch_sizes == [1, 1]
    progress = [json.loads(line) for line in (output_dir / "logs" / "progress.jsonl").read_text().splitlines()]
    assert [event["completed"] for event in progress] == [0, 1, 2]
    state = json.loads((output_dir / "logs" / "state.json").read_text())
    assert state["status"] == "succeeded"
    assert state["completed"] == 2
    assert state["total"] == 2
    assert state["percent"] == 100.0
    assert state["items_per_sec"] >= 0


def test_run_writes_item_errors_to_isolated_error_log(tmp_path):
    task_body = """
import json

def load(source, limit):
    return [json.loads(line) for line in open(source)][:1]

def build_prompt(item, config):
    return item["id"]

def extract_answer(raw, item):
    raise ValueError("bad extract")
""".lstrip()
    config_path = write_workspace(tmp_path, task_body)
    config_path.write_text(
        config_path.read_text()
        + """
runtime:
  batch_size: 1
logging:
  console: false
  progress_every: 1
  progress_interval_sec: 999
  error_preview_chars: 40
  sample_bad_records: 20
"""
    )
    config = load_config(config_path)

    run(config, backend=StaticBackend([GenResult("raw text", "stop")]))

    errors = [
        json.loads(line)
        for line in (tmp_path / "runs" / "demo_run" / "logs" / "errors.jsonl").read_text().splitlines()
    ]
    assert errors[0]["event"] == "item_error"
    assert errors[0]["id"] == "a"
    assert "ValueError" in errors[0]["message"]
