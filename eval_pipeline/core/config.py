from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ModelConfig:
    backend: str
    path: str | None = None
    data_parallel_size: int | None = None
    tensor_parallel_size: int | None = None
    base_url: str | None = None
    model: str | None = None
    concurrency: int | None = None
    max_retries: int | None = None
    max_model_len: int | None = None
    gpu_memory_utilization: float | None = None
    dtype: str | None = None
    max_num_seqs: int | None = None
    enforce_eager: bool | None = None


@dataclass
class DatasetConfig:
    task_file: str
    source: str
    limit: int | None
    n_samples: int


@dataclass
class GenerationConfig:
    max_tokens: int
    temperature: float
    stop: str | list[str] | None
    seed: int


@dataclass
class OutputConfig:
    on_truncate: str
    flush_every: int


@dataclass
class ThresholdConfig:
    truncated: float
    extract_failed: float
    error: float


@dataclass
class RuntimeConfig:
    batch_size: int


@dataclass
class LoggingConfig:
    console: bool
    progress_every: int
    progress_interval_sec: float
    error_preview_chars: int
    sample_bad_records: int


@dataclass
class Config:
    name: str
    model: ModelConfig
    dataset: DatasetConfig
    generation: GenerationConfig
    output: OutputConfig
    thresholds: ThresholdConfig
    runtime: RuntimeConfig
    logging: LoggingConfig
    workspace: Path
    raw: dict[str, Any]

    def snapshot(self) -> dict[str, Any]:
        data = asdict(self)
        data["workspace"] = str(self.workspace)
        return data


def load_config(path: str | Path) -> Config:
    config_path = Path(path)
    raw = yaml.safe_load(config_path.read_text())
    model = raw["model"]
    dataset = raw["dataset"]
    generation = raw["generation"]
    output = raw["output"]
    thresholds = raw["thresholds"]
    runtime = raw.get("runtime", {})
    logging = raw.get("logging", {})
    backend = model["backend"]
    if backend == "api":
        model_config = ModelConfig(
            backend=backend,
            base_url=model["base_url"],
            model=model["model"],
            concurrency=model["concurrency"],
            max_retries=model["max_retries"],
        )
    elif backend == "vllm":
        model_config = ModelConfig(
            backend=backend,
            path=model["path"],
            data_parallel_size=model["data_parallel_size"],
            tensor_parallel_size=model["tensor_parallel_size"],
            max_model_len=model.get("max_model_len"),
            gpu_memory_utilization=model.get("gpu_memory_utilization"),
            dtype=model.get("dtype"),
            max_num_seqs=model.get("max_num_seqs"),
            enforce_eager=model.get("enforce_eager"),
        )
    else:
        raise ValueError(f"unknown backend: {backend}")
    return Config(
        name=raw["name"],
        model=model_config,
        dataset=DatasetConfig(
            task_file=dataset["task_file"],
            source=dataset["source"],
            limit=dataset["limit"],
            n_samples=dataset["n_samples"],
        ),
        generation=GenerationConfig(
            max_tokens=generation["max_tokens"],
            temperature=generation["temperature"],
            stop=generation["stop"],
            seed=generation["seed"],
        ),
        output=OutputConfig(
            on_truncate=output["on_truncate"],
            flush_every=output["flush_every"],
        ),
        thresholds=ThresholdConfig(
            truncated=thresholds["truncated"],
            extract_failed=thresholds["extract_failed"],
            error=thresholds["error"],
        ),
        runtime=RuntimeConfig(batch_size=runtime.get("batch_size", 64)),
        logging=LoggingConfig(
            console=logging.get("console", True),
            progress_every=logging.get("progress_every", 100),
            progress_interval_sec=logging.get("progress_interval_sec", 30),
            error_preview_chars=logging.get("error_preview_chars", 300),
            sample_bad_records=logging.get("sample_bad_records", 20),
        ),
        workspace=config_path.parent,
        raw=raw,
    )
