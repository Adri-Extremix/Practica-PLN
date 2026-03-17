from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


TASK_DEFAULT_MODELS = {
    "b": [
        "microsoft/mdeberta-v3-base",
        "xlm-roberta-base",
        "bert-base-multilingual-cased",
    ],
    "c": [
        "microsoft/mdeberta-v3-base",
        "xlm-roberta-base",
        "bert-base-multilingual-cased",
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run model sweeps for HOPE-EXP tasks.")
    parser.add_argument("--task", choices=["b", "c"], required=True)
    parser.add_argument("--models", nargs="*", default=None)
    parser.add_argument("--base-output-dir", default="outputs/model_sweeps")
    parser.add_argument("--extra-args", nargs=argparse.REMAINDER, default=[])
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def slugify_model_name(model_name: str) -> str:
    return model_name.replace("/", "__")


def build_command(task: str, model_name: str, output_dir: Path, extra_args: list[str]) -> list[str]:
    if task == "b":
        base_cmd = [
            sys.executable,
            "train_task_b_emotion.py",
            "--model-name",
            model_name,
            "--output-dir",
            str(output_dir),
            "--optimize-threshold",
        ]
    elif task == "c":
        base_cmd = [
            sys.executable,
            "train_task_c_spans.py",
            "--model-name",
            model_name,
            "--output-dir",
            str(output_dir),
        ]
    else:
        raise ValueError(f"Unsupported task: {task}")

    return base_cmd + list(extra_args)


def main() -> None:
    args = parse_args()
    models = args.models or TASK_DEFAULT_MODELS[args.task]
    task_output_dir = Path(args.base_output_dir) / f"task_{args.task}"
    task_output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Running sweep for task {args.task.upper()} with {len(models)} models")
    for model_name in models:
        output_dir = task_output_dir / slugify_model_name(model_name)
        command = build_command(args.task, model_name, output_dir, args.extra_args)
        pretty = " ".join(command)
        print(f"\n[{args.task.upper()}] {model_name}")
        print(pretty)
        if args.dry_run:
            continue
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
