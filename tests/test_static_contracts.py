import ast
from pathlib import Path


def test_process_one_is_the_only_try_except_boundary():
    roots = [Path("eval_pipeline")]
    try_nodes = []
    for root in roots:
        for path in root.rglob("*.py"):
            tree = ast.parse(path.read_text(), filename=str(path))
            parents = {child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}
            for node in ast.walk(tree):
                if isinstance(node, ast.Try):
                    current = node
                    function_name = None
                    while current in parents:
                        current = parents[current]
                        if isinstance(current, ast.FunctionDef):
                            function_name = current.name
                            break
                    try_nodes.append((str(path), function_name))

    assert try_nodes == [
        ("eval_pipeline/cli.py", "main"),
        ("eval_pipeline/core/runner.py", "process_one"),
    ]
