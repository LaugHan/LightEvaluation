import importlib.util
from pathlib import Path
from types import ModuleType


REQUIRED_FUNCTIONS = ("load", "build_prompt", "extract_answer")


def load_task(workspace: str | Path, task_file: str) -> ModuleType:
    workspace_path = Path(workspace).resolve()
    task_path = (workspace_path / task_file).resolve()
    if Path(task_file).is_absolute() or not task_path.is_relative_to(workspace_path):
        raise ValueError("task_file must stay inside the workspace")
    spec = importlib.util.spec_from_file_location("eval_pipeline_user_task", task_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    missing = [name for name in REQUIRED_FUNCTIONS if not hasattr(module, name)]
    if missing:
        raise AttributeError(f"task file missing required functions: {', '.join(missing)}")
    return module
