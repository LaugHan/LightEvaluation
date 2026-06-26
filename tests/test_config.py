import pytest

from eval_pipeline.core.config import load_config


def write_config(path):
    path.write_text(
        """
name: run1
model:
  backend: api
  base_url: http://localhost:8000/v1
  model: demo
  concurrency: 2
  max_retries: 3
dataset:
  task_file: tasks/demo.py
  source: data.jsonl
  limit: 10
  n_samples: 2
generation:
  max_tokens: 32
  temperature: 0.0
  stop: null
  seed: 0
output:
  on_truncate: drop
  flush_every: 1
thresholds:
  truncated: 0.1
  extract_failed: 0.2
  error: 0.3
""".lstrip()
    )


def test_load_config_maps_yaml_to_dataclasses(tmp_path):
    config_path = tmp_path / "config.yaml"
    write_config(config_path)

    config = load_config(config_path)

    assert config.name == "run1"
    assert config.workspace == tmp_path
    assert config.model.backend == "api"
    assert config.dataset.task_file == "tasks/demo.py"
    assert config.generation.max_tokens == 32
    assert config.output.on_truncate == "drop"
    assert config.thresholds.error == 0.3


def test_load_config_missing_required_field_fails_loudly(tmp_path):
    config_path = tmp_path / "config.yaml"
    write_config(config_path)
    config_path.write_text(config_path.read_text().replace("name: run1\n", ""))

    with pytest.raises(KeyError):
        load_config(config_path)


def test_load_config_api_backend_requires_api_fields(tmp_path):
    config_path = tmp_path / "config.yaml"
    write_config(config_path)
    config_path.write_text(config_path.read_text().replace("  base_url: http://localhost:8000/v1\n", ""))

    with pytest.raises(KeyError):
        load_config(config_path)


def test_load_config_vllm_backend_requires_vllm_fields(tmp_path):
    config_path = tmp_path / "config.yaml"
    write_config(config_path)
    config_path.write_text(
        config_path.read_text()
        .replace("  backend: api", "  backend: vllm")
        .replace("  base_url: http://localhost:8000/v1\n", "")
        .replace("  model: demo\n", "")
        .replace("  concurrency: 2\n", "")
        .replace("  max_retries: 3\n", "")
        .replace("model:\n  backend: vllm\n", "model:\n  backend: vllm\n  data_parallel_size: 2\n  tensor_parallel_size: 1\n")
    )

    with pytest.raises(KeyError):
        load_config(config_path)
