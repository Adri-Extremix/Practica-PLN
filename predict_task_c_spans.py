from __future__ import annotations

import argparse
import os

import torch

from src.span_model import build_span_tokenizer, load_span_model
from src.span_utils import (
    aggregate_predictions,
    build_dataloader,
    collect_logits,
    load_primary_labels,
    load_records,
    predictions_to_submission,
    save_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict HOPE-EXP Task C spans.")
    parser.add_argument("--input-path", default="HopeEXP_Test_unlabeled.jsonl")
    parser.add_argument("--model-dir", default="outputs/task_c_spans/best_model")
    parser.add_argument("--output-path", default="outputs/submission_task_c.json")
    parser.add_argument("--primary-labels-path", default=None)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--stride", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Usando dispositivo: {device}")
    tokenizer = build_span_tokenizer(args.model_dir)
    model = load_span_model(args.model_dir, device=device)
    records = load_records(args.input_path)
    dataset, dataloader = build_dataloader(
        records,
        tokenizer,
        max_length=args.max_length,
        stride=args.stride,
        batch_size=args.batch_size,
        shuffle=False,
        include_labels=False,
    )
    logits_by_feature_id = collect_logits(model, dataloader, device)
    predictions = aggregate_predictions(dataset, logits_by_feature_id)

    primary_label_mapping = None
    if args.primary_labels_path:
        primary_label_mapping = load_primary_labels(args.primary_labels_path)

    submission = predictions_to_submission(records, predictions, primary_label_mapping=primary_label_mapping)
    save_json(submission, args.output_path)
    print(f"Saved predictions to {args.output_path}")


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
