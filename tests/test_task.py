import pytest

from eval_pipeline.core.task import load_task


def test_load_task_imports_workspace_relative_task_file(tmp_path):
    task_dir = tmp_path / "tasks"
    task_dir.mkdir()
    (task_dir / "demo.py").write_text(
        """
def load(source, limit):
    return [{"id": "1"}]

def build_prompt(item, config):
    return "prompt"

def extract_answer(raw, item):
    return raw, "extracted"
""".lstrip()
    )

    task = load_task(tmp_path, "tasks/demo.py")

    assert task.load("source", None) == [{"id": "1"}]
    assert task.build_prompt({}, None) == "prompt"
    assert task.extract_answer("x", {}) == ("x", "extracted")
    assert not hasattr(task, "score")


def test_load_task_missing_required_function_fails_loudly(tmp_path):
    task_dir = tmp_path / "tasks"
    task_dir.mkdir()
    (task_dir / "bad.py").write_text("def load(source, limit):\n    return []\n")

    with pytest.raises(AttributeError):
        load_task(tmp_path, "tasks/bad.py")


def test_load_task_rejects_absolute_path_outside_workspace(tmp_path):
    outside = tmp_path.parent / "outside_task.py"
    outside.write_text(
        """
def load(source, limit):
    return []

def build_prompt(item, config):
    return ""

def extract_answer(raw, item):
    return None, "failed"
""".lstrip()
    )

    with pytest.raises(ValueError):
        load_task(tmp_path, str(outside))


def test_load_task_rejects_parent_escape_from_workspace(tmp_path):
    outside = tmp_path.parent / "outside_task.py"
    outside.write_text(
        """
def load(source, limit):
    return []

def build_prompt(item, config):
    return ""

def extract_answer(raw, item):
    return None, "failed"
""".lstrip()
    )

    with pytest.raises(ValueError):
        load_task(tmp_path, "../outside_task.py")
