import json
import subprocess
import sys

from eval_pipeline.cli import main


def test_init_creates_workspace_templates(tmp_path):
    workspace = tmp_path / "workspace"

    assert main(["init", str(workspace)]) == 0

    assert (workspace / "config.yaml").exists()
    assert (workspace / "tasks" / "task_generative.py").exists()
    assert (workspace / "tasks" / "task_mcq.py").exists()
    assert (workspace / "tasks" / "task_datagen.py").exists()
    assert (workspace / "data.jsonl").exists()


def test_show_filters_records_by_status(tmp_path, capsys):
    config = tmp_path / "config.yaml"
    config.write_text(
        """
name: run1
model:
  backend: api
  base_url: http://localhost:8000/v1
  model: demo
  concurrency: 1
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
    output_dir = tmp_path / "runs" / "run1"
    output_dir.mkdir(parents=True)
    (output_dir / "output.jsonl").write_text(
        json.dumps({"id": "a", "status": "ok"}) + "\n"
        + json.dumps({"id": "b", "status": "error"}) + "\n"
    )

    assert main(["show", str(config), "--status", "error", "--n", "10"]) == 0

    out = capsys.readouterr().out
    assert '"id": "b"' in out
    assert '"id": "a"' not in out


def test_show_limits_number_of_printed_records(tmp_path, capsys):
    config = tmp_path / "config.yaml"
    config.write_text(
        """
name: run1
model:
  backend: api
  base_url: http://localhost:8000/v1
  model: demo
  concurrency: 1
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
    output_dir = tmp_path / "runs" / "run1"
    output_dir.mkdir(parents=True)
    (output_dir / "output.jsonl").write_text(
        json.dumps({"id": "a", "status": "error"}) + "\n"
        + json.dumps({"id": "b", "status": "error"}) + "\n"
    )

    assert main(["show", str(config), "--status", "error", "--n", "1"]) == 0

    out = capsys.readouterr().out
    assert '"id": "a"' in out
    assert '"id": "b"' not in out


def test_cli_module_invocation_executes_main(tmp_path):
    result = subprocess.run(
        [sys.executable, "-m", "eval_pipeline.cli", "init", str(tmp_path / "workspace")],
        cwd=str(__import__("pathlib").Path(__file__).resolve().parents[1]),
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0
    assert (tmp_path / "workspace" / "config.yaml").exists()


def test_run_command_logs_fatal_errors(tmp_path):
    config = tmp_path / "config.yaml"
    config.write_text(
        """
name: bad_run
model:
  backend: api
  base_url: http://localhost:8000/v1
  model: demo
  concurrency: 1
  max_retries: 1
dataset:
  task_file: tasks/missing.py
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
runtime:
  batch_size: 1
logging:
  console: false
  progress_every: 1
  progress_interval_sec: 999
  error_preview_chars: 80
  sample_bad_records: 20
""".lstrip()
    )

    assert main(["run", str(config)]) == 1

    logs = tmp_path / "runs" / "bad_run" / "logs"
    run_events = [json.loads(line) for line in (logs / "run.jsonl").read_text().splitlines()]
    state = json.loads((logs / "state.json").read_text())
    assert run_events[-1]["event"] == "run_failed"
    assert state["status"] == "failed"
    assert (logs / "fatal.log").exists()
