from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split as sk_split

from src.data_utils import (
    EMOTION_LABELS,
    add_emotion_vectors,
    apply_cleaning,
    compute_class_weights,
    load_split,
)
from src.dataset import build_all_dataloaders
from src.metrics import find_best_threshold, find_best_threshold_per_class, print_metrics
from src.model import build_model, build_tokenizer, load_model
from src.span_utils import save_json, set_seed
from src.trainer import evaluate_epoch, train


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train HOPE-EXP Task B emotion classifier.")
    parser.add_argument("--train-path", default="HopeEXP_Train.jsonl")
    parser.add_argument("--test-path", default="HopeEXP_Test_unlabeled.jsonl")
    parser.add_argument("--output-dir", default="outputs/task_b_emotion")
    parser.add_argument("--model-name", default="microsoft/mdeberta-v3-base")
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--dev-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--disable-pos-weight", action="store_true")
    parser.add_argument("--max-train-examples", type=int, default=None)
    parser.add_argument("--max-dev-examples", type=int, default=None)
    parser.add_argument("--optimize-threshold", action="store_true")
    parser.add_argument("--optimize-threshold-per-class", action="store_true")
    return parser.parse_args()


def maybe_limit(df, max_examples: int | None):
    if max_examples is None:
        return df.reset_index(drop=True)
    return df.iloc[:max_examples].reset_index(drop=True)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Usando dispositivo: {device}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = output_dir / "best_model_task_b.pt"

    clean_kwargs = dict(
        remove_urls=True,
        remove_mentions=True,
        remove_hashtag_symbol=True,
        lowercase=False,
    )

    full_train_df = load_split(args.train_path)
    test_df = load_split(args.test_path)

    train_df, dev_df = sk_split(
        full_train_df,
        test_size=args.dev_size,
        random_state=args.seed,
        stratify=full_train_df["primary_label"],
    )

    train_df = maybe_limit(train_df, args.max_train_examples)
    dev_df = maybe_limit(dev_df, args.max_dev_examples)
    test_df = test_df.reset_index(drop=True)

    train_df = apply_cleaning(train_df, text_col="text", **clean_kwargs)
    dev_df = apply_cleaning(dev_df, text_col="text", **clean_kwargs)
    test_df = apply_cleaning(test_df, text_col="text", **clean_kwargs)

    train_df = add_emotion_vectors(train_df, emotions_col="emotions")
    dev_df = add_emotion_vectors(dev_df, emotions_col="emotions")

    train_texts = train_df["text"].tolist()
    train_labels = train_df["emotion_vector"].tolist()
    dev_texts = dev_df["text"].tolist()
    dev_labels = dev_df["emotion_vector"].tolist()
    test_texts = test_df["text"].tolist()
    test_ids = test_df["id"].tolist()

    tokenizer = build_tokenizer(args.model_name)
    loaders = build_all_dataloaders(
        train_texts=train_texts,
        train_labels=train_labels,
        dev_texts=dev_texts,
        dev_labels=dev_labels,
        tokenizer=tokenizer,
        test_texts=test_texts,
        max_length=args.max_length,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    pos_weight = None
    if not args.disable_pos_weight:
        pos_weight = torch.tensor(compute_class_weights(train_df, emotions_col="emotions"), dtype=torch.float32)

    model = build_model(
        model_name=args.model_name,
        num_labels=len(EMOTION_LABELS),
        dropout_prob=args.dropout,
    ).float()

    history = train(
        model=model,
        train_loader=loaders["train"],
        dev_loader=loaders["dev"],
        device=device,
        num_epochs=args.epochs,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        threshold=args.threshold,
        pos_weight=pos_weight,
        save_dir=str(output_dir),
        model_name=best_model_path.name,
        early_stopping_patience=args.patience,
        monitor_metric="f1_macro",
        verbose=True,
    )

    best_model = load_model(
        path=str(best_model_path),
        model_name=args.model_name,
        num_labels=len(EMOTION_LABELS),
        dropout_prob=args.dropout,
        device=device,
    ).float()

    dev_loss, dev_metrics, dev_probs, dev_true = evaluate_epoch(
        best_model,
        loaders["dev"],
        device,
        threshold=args.threshold,
        pos_weight=pos_weight,
    )

    best_threshold = args.threshold
    threshold_search = None
    best_threshold_per_class = None

    if args.optimize_threshold:
        best_threshold, best_threshold_score = find_best_threshold(
            dev_true,
            dev_probs,
            metric="f1_macro",
            verbose=True,
        )
        _, dev_metrics, _, _ = evaluate_epoch(
            best_model,
            loaders["dev"],
            device,
            threshold=best_threshold,
            pos_weight=pos_weight,
        )
        threshold_search = {
            "best_threshold": float(best_threshold),
            "best_f1_macro": float(best_threshold_score),
        }

    if args.optimize_threshold_per_class:
        best_threshold_per_class = find_best_threshold_per_class(dev_true, dev_probs).tolist()

    print("\nEvaluacion final en dev")
    print_metrics(dev_metrics)

    experiment_config = {
        "task": "B",
        "model_name": args.model_name,
        "max_length": args.max_length,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "warmup_ratio": args.warmup_ratio,
        "weight_decay": args.weight_decay,
        "dropout": args.dropout,
        "patience": args.patience,
        "seed": args.seed,
        "dev_size": args.dev_size,
        "best_threshold": float(best_threshold),
        "train_examples": len(train_df),
        "dev_examples": len(dev_df),
        "test_examples": len(test_df),
        "dev_loss": float(dev_loss),
        "dev_metrics": {k: float(v) for k, v in dev_metrics.items()},
        "threshold_search": threshold_search,
        "thresholds_per_class": best_threshold_per_class,
        "test_ids_preview": test_ids[:5],
    }

    save_json(history, output_dir / "training_history.json")
    save_json(experiment_config, output_dir / "experiment_config.json")
    np.save(output_dir / "dev_probs.npy", dev_probs)
    np.save(output_dir / "dev_true.npy", dev_true)


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
