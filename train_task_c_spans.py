from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup

from src.span_model import build_span_model, build_span_tokenizer, save_span_model
from src.span_utils import (
    aggregate_predictions,
    build_dataloader,
    collect_logits,
    compute_class_weights,
    evaluate_span_predictions,
    format_metrics,
    limit_records,
    load_records,
    save_json,
    set_seed,
    summarize_dataset,
    train_dev_split,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train HOPE-EXP Task C span extractor.")
    parser.add_argument("--train-path", default="HopeEXP_Train.jsonl")
    parser.add_argument("--dev-path", default=None)
    parser.add_argument("--output-dir", default="outputs/task_c_spans")
    parser.add_argument("--model-name", default="microsoft/mdeberta-v3-base")
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--stride", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dev-size", type=float, default=0.1)
    parser.add_argument("--max-train-examples", type=int, default=None)
    parser.add_argument("--max-dev-examples", type=int, default=None)
    return parser.parse_args()


def run_epoch(model, dataloader, optimizer, scheduler, device, class_weights, grad_clip) -> float:
    """Ejecuta una época completa de entrenamiento."""
    model.train()
    total_loss = 0.0
    # Usamos pesos en la pérdida para compensar el desbalance de clases (muchas etiquetas 'O')
    # ignore_index=-100 evita calcular pérdida en tokens de relleno o especiales
    loss_fct = torch.nn.CrossEntropyLoss(weight=class_weights.to(device=device, dtype=torch.float32), ignore_index=-100)
    for batch in dataloader:
        optimizer.zero_grad()
        inputs = {
            "input_ids": batch["input_ids"].to(device),
            "attention_mask": batch["attention_mask"].to(device),
            "labels": batch["labels"].to(device),
        }
        if "token_type_ids" in batch:
            inputs["token_type_ids"] = batch["token_type_ids"].to(device)

        outputs = model(**inputs)
        # Aseguramos precisión simple para la pérdida incluso si usamos precisiones mixtas
        logits = outputs.logits.float()
        labels = inputs["labels"]
        loss = loss_fct(logits.view(-1, model.num_labels), labels.view(-1))
        
        loss.backward()
        # Recorte de gradientes para estabilizar el entrenamiento de modelos Transformer
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()
    return total_loss / max(1, len(dataloader))


def evaluate(model, dataset, dataloader, records, device, class_weights) -> tuple[float, dict[str, float]]:
    """Evalúa el modelo tanto en términos de pérdida como de métricas de texto (ROUGE-1)."""
    model.eval()
    total_loss = 0.0
    loss_fct = torch.nn.CrossEntropyLoss(weight=class_weights.to(device=device, dtype=torch.float32), ignore_index=-100)
    with torch.no_grad():
        for batch in dataloader:
            inputs = {
                "input_ids": batch["input_ids"].to(device),
                "attention_mask": batch["attention_mask"].to(device),
                "labels": batch["labels"].to(device),
            }
            if "token_type_ids" in batch:
                inputs["token_type_ids"] = batch["token_type_ids"].to(device)
            outputs = model(**inputs)
            loss = loss_fct(outputs.logits.float().view(-1, model.num_labels), inputs["labels"].view(-1))
            total_loss += loss.item()

    # Recolectar logits y reconstruir predicciones de texto completo a partir de ventanas (chunks)
    logits_by_feature_id = collect_logits(model, dataloader, device)
    predictions = aggregate_predictions(dataset, logits_by_feature_id)
    # Cálculo de ROUGE-1 y Exact Match F1 sobre el texto reconstruido
    metrics = evaluate_span_predictions(records, predictions)
    return total_loss / max(1, len(dataloader)), metrics


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

    train_records = load_records(args.train_path)
    if args.dev_path:
        dev_records = load_records(args.dev_path)
    else:
        train_records, dev_records = train_dev_split(train_records, dev_size=args.dev_size, seed=args.seed)

    train_records = limit_records(train_records, args.max_train_examples)
    dev_records = limit_records(dev_records, args.max_dev_examples)

    tokenizer = build_span_tokenizer(args.model_name)
    train_dataset, train_loader = build_dataloader(
        train_records,
        tokenizer,
        max_length=args.max_length,
        stride=args.stride,
        batch_size=args.batch_size,
        shuffle=True,
        include_labels=True,
    )
    dev_dataset, dev_loader = build_dataloader(
        dev_records,
        tokenizer,
        max_length=args.max_length,
        stride=args.stride,
        batch_size=args.batch_size,
        shuffle=False,
        include_labels=True,
    )

    class_weights = compute_class_weights(train_dataset)
    model = build_span_model(args.model_name).to(device)

    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * args.warmup_ratio),
        num_training_steps=total_steps,
    )

    history: list[dict[str, float]] = []
    best_metric = -1.0
    best_dir = output_dir / "best_model"

    print(json.dumps({
        "train": summarize_dataset(train_records),
        "dev": summarize_dataset(dev_records),
        "device": str(device),
    }, indent=2, ensure_ascii=False))

    for epoch in range(1, args.epochs + 1):
        start = time.time()
        train_loss = run_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            class_weights=class_weights,
            grad_clip=args.grad_clip,
        )
        dev_loss, dev_metrics = evaluate(model, dev_dataset, dev_loader, dev_records, device, class_weights)
        elapsed = time.time() - start
        current = {
            "epoch": epoch,
            "train_loss": train_loss,
            "dev_loss": dev_loss,
            **dev_metrics,
        }
        history.append(current)
        print(f"Epoch {epoch}/{args.epochs} - {elapsed:.1f}s - train_loss={train_loss:.4f} - dev_loss={dev_loss:.4f} - {format_metrics(dev_metrics)}")

        if dev_metrics["rouge1_f1"] > best_metric:
            best_metric = dev_metrics["rouge1_f1"]
            save_span_model(model, tokenizer, str(best_dir))
            print(f"Saved best checkpoint to {best_dir}")

    save_json(history, output_dir / "training_history.json")
    save_json(
        {
            "model_name": args.model_name,
            "max_length": args.max_length,
            "stride": args.stride,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "warmup_ratio": args.warmup_ratio,
            "seed": args.seed,
            "best_dev_rouge1_f1": best_metric,
            "train_examples": len(train_records),
            "dev_examples": len(dev_records),
        },
        output_dir / "experiment_config.json",
    )
    print(f"Training finished. Best dev ROUGE-1 F1: {best_metric:.4f}")


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
