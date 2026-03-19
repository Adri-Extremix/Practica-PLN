from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

from src.data_utils import apply_cleaning, format_predictions_for_submission, load_split, save_submission
from src.dataset import build_dataloader
from src.metrics import ensemble_probs
from src.model import build_tokenizer, load_model
from src.trainer import predict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict HOPE-EXP Task B emotions.")
    parser.add_argument("--input-path", default="HopeEXP_Test_unlabeled.jsonl")
    parser.add_argument("--model-dir", default="outputs/task_b_emotion")
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--output-path", default="outputs/submission_task_b.json")
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--ensemble-dirs", nargs="*", default=None)
    return parser.parse_args()


def load_experiment_config(model_dir: Path) -> dict:
    config_path = model_dir / "experiment_config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"No se encontro {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def resolve_model_path(model_dir: Path, explicit_model_path: str | None) -> Path:
    if explicit_model_path:
        return Path(explicit_model_path)
    return model_dir / "best_model_task_b.pt"


def predict_single_model(model_dir: Path, args: argparse.Namespace, texts: list[str], device: torch.device) -> tuple[np.ndarray, dict]:
    config = load_experiment_config(model_dir)
    model_name = args.model_name or config["model_name"]
    max_length = args.max_length or config["max_length"]
    batch_size = args.batch_size or config["batch_size"]
    dropout = args.dropout if args.dropout is not None else config.get("dropout", 0.1)

    tokenizer = build_tokenizer(model_name)
    dataloader = build_dataloader(
        texts=texts,
        tokenizer=tokenizer,
        labels=None,
        max_length=max_length,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    model_path = resolve_model_path(model_dir, args.model_path if model_dir == Path(args.model_dir) else None)
    model = load_model(
        path=str(model_path),
        model_name=model_name,
        dropout_prob=dropout,
        device=device,
    ).float()
    probs = predict(model, dataloader, device)
    return probs, config


def main() -> None:
    args = parse_args()
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Usando dispositivo: {device}")
    input_df = load_split(args.input_path)
    input_df = apply_cleaning(
        input_df,
        text_col="text",
        remove_urls=True,
        remove_mentions=True,
        remove_hashtag_symbol=True,
        lowercase=False,
    )
    texts = input_df["text"].tolist()
    ids = input_df["id"].tolist()

    if args.ensemble_dirs:
        model_dirs = [Path(model_dir) for model_dir in args.ensemble_dirs]
    else:
        model_dirs = [Path(args.model_dir)]

    list_of_probs = []
    configs = []
    for model_dir in model_dirs:
        probs, config = predict_single_model(model_dir, args, texts, device)
        list_of_probs.append(probs)
        configs.append(config)
        np.save(model_dir / "test_probs.npy", probs)

    if len(list_of_probs) == 1:
        final_probs = list_of_probs[0]
        config = configs[0]
    else:
        final_probs = ensemble_probs(list_of_probs)
        config = configs[0]

    threshold = args.threshold if args.threshold is not None else config.get("best_threshold", 0.5)
    submission_df = format_predictions_for_submission(ids, final_probs.tolist(), threshold=threshold)
    save_submission(submission_df, args.output_path)
    print(f"Prediccion completada con threshold={threshold:.2f}")


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
