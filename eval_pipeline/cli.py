import argparse
import json
import shutil
from importlib import resources
from pathlib import Path

from eval_pipeline.core.config import load_config
from eval_pipeline.core.io import read_records
from eval_pipeline.core.log import RunLogger
from eval_pipeline.core.runner import merge, pilot, run


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="eval-pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    init_parser = subparsers.add_parser("init")
    init_parser.add_argument("dir")

    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("config")
    run_parser.add_argument("--rank", type=int)
    run_parser.add_argument("--world-size", type=int)

    pilot_parser = subparsers.add_parser("pilot")
    pilot_parser.add_argument("config")
    pilot_parser.add_argument("--n", type=int, default=16)

    show_parser = subparsers.add_parser("show")
    show_parser.add_argument("config")
    show_parser.add_argument("--status", required=True)
    show_parser.add_argument("--n", type=int, default=10)

    merge_parser = subparsers.add_parser("merge")
    merge_parser.add_argument("config")

    args = parser.parse_args(argv)
    try:
        if args.command == "init":
            init_workspace(Path(args.dir))
        elif args.command == "run":
            config = load_config(args.config)
            print(json.dumps(run(config, rank=args.rank, world_size=args.world_size), ensure_ascii=False, indent=2))
        elif args.command == "pilot":
            print(json.dumps(pilot(load_config(args.config), n=args.n), ensure_ascii=False, indent=2))
        elif args.command == "show":
            show_records(load_config(args.config), args.status, args.n)
        elif args.command == "merge":
            print(merge(load_config(args.config)))
    except Exception as exc:
        if args.command == "run":
            config = load_config(args.config)
            RunLogger(config.workspace / "runs" / config.name, config, args.rank).fail(exc)
            return 1
        raise
    return 0


def init_workspace(workspace: Path) -> None:
    workspace.mkdir(parents=True, exist_ok=True)
    (workspace / "tasks").mkdir(exist_ok=True)
    template_root = resources.files("eval_pipeline.templates")
    shutil.copyfile(template_root / "config.yaml", workspace / "config.yaml")
    for name in ("task_mcq.py", "task_generative.py", "task_datagen.py"):
        shutil.copyfile(template_root / name, workspace / "tasks" / name)
    shutil.copyfile(template_root / "data.jsonl", workspace / "data.jsonl")


def show_records(config, status: str, n: int) -> None:
    output = config.workspace / "runs" / config.name / "output.jsonl"
    shown = 0
    for record in read_records(output):
        if record.get("status") == status:
            print(json.dumps(record, ensure_ascii=False))
            shown += 1
            if shown >= n:
                return


if __name__ == "__main__":
    raise SystemExit(main())
