from __future__ import annotations

import os
import time
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.optim import AdamW
from tqdm.auto import tqdm
from transformers import get_linear_schedule_with_warmup

from src.span_model import build_span_model, build_span_tokenizer, save_span_model
from src.span_utils import (
    ID2LABEL,
    SpanCandidate,
    aggregate_predictions,
    build_dataloader,
    collect_logits,
    compute_class_weights,
    evaluate_span_predictions,
    softmax,
)


DEFAULT_CANDIDATE_MODELS: list[dict[str, Any]] = [
    {
        "name": "microsoft/mdeberta-v3-base",
        "description": "mDeBERTa v3 - strong multilingual baseline",
        "multilingual": True,
    },
    {
        "name": "xlm-roberta-base",
        "description": "XLM-RoBERTa base - robust multilingual encoder",
        "multilingual": True,
    },
    {
        "name": "xlm-roberta-large",
        "description": "XLM-RoBERTa large - higher capacity, more VRAM",
        "multilingual": True,
    },
    {
        "name": "bert-base-multilingual-cased",
        "description": "mBERT cased - classic multilingual baseline",
        "multilingual": True,
    },
]


def run_train_epoch(
    model: torch.nn.Module,
    dataloader,
    optimizer,
    scheduler,
    device: torch.device,
    class_weights: torch.Tensor,
    grad_clip: float = 1.0,
) -> float:
    model.train()
    total_loss = 0.0
    loss_fct = torch.nn.CrossEntropyLoss(
        weight=class_weights.to(device=device, dtype=torch.float32),
        ignore_index=-100,
    )

    pbar = tqdm(dataloader, desc="Train", leave=False)
    for batch in pbar:
        optimizer.zero_grad()
        inputs = {
            "input_ids": batch["input_ids"].to(device),
            "attention_mask": batch["attention_mask"].to(device),
            "labels": batch["labels"].to(device),
        }
        if "token_type_ids" in batch:
            inputs["token_type_ids"] = batch["token_type_ids"].to(device)

        outputs = model(**inputs)
        loss = loss_fct(outputs.logits.float().view(-1, model.num_labels), inputs["labels"].view(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        scheduler.step()

        loss_value = float(loss.item())
        total_loss += loss_value
        pbar.set_postfix({"loss": f"{loss_value:.4f}"})

    return total_loss / max(1, len(dataloader))


def evaluate_span_model(
    model: torch.nn.Module,
    dataset,
    dataloader,
    records: list[dict[str, Any]],
    device: torch.device,
    class_weights: torch.Tensor,
) -> tuple[float, dict[str, float], dict[str | int, list[SpanCandidate]]]:
    model.eval()
    total_loss = 0.0
    loss_fct = torch.nn.CrossEntropyLoss(
        weight=class_weights.to(device=device, dtype=torch.float32),
        ignore_index=-100,
    )

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Eval Loss", leave=False):
            inputs = {
                "input_ids": batch["input_ids"].to(device),
                "attention_mask": batch["attention_mask"].to(device),
                "labels": batch["labels"].to(device),
            }
            if "token_type_ids" in batch:
                inputs["token_type_ids"] = batch["token_type_ids"].to(device)

            outputs = model(**inputs)
            loss = loss_fct(outputs.logits.float().view(-1, model.num_labels), inputs["labels"].view(-1))
            total_loss += float(loss.item())

    logits_by_feature_id = collect_logits(model, dataloader, device)
    predictions = aggregate_predictions(dataset, logits_by_feature_id)
    metrics = evaluate_span_predictions(records, predictions)
    return total_loss / max(1, len(dataloader)), metrics, predictions


def train_span_model(
    model: torch.nn.Module,
    tokenizer,
    train_dataset,
    train_loader,
    dev_dataset,
    dev_loader,
    dev_records: list[dict[str, Any]],
    device: torch.device,
    class_weights: torch.Tensor,
    num_epochs: int,
    learning_rate: float,
    warmup_ratio: float,
    weight_decay: float,
    grad_clip: float,
    save_dir: str,
    early_stopping_patience: int,
    monitor_metric: str = "rouge1_f1",
    verbose: bool = True,
) -> tuple[dict[str, list[Any]], float, int]:
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    total_steps = len(train_loader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * warmup_ratio),
        num_training_steps=total_steps,
    )

    history: dict[str, list[Any]] = {
        "train_loss": [],
        "dev_loss": [],
        "dev_metrics": [],
    }

    best_metric = -1.0
    best_epoch = 0
    patience_counter = 0

    if verbose:
        print(f"Iniciando entrenamiento: {num_epochs} epocas, lr={learning_rate}")
        print(f"- Total steps: {total_steps} | Warmup steps: {int(total_steps * warmup_ratio)}")
        print(f"- Early stopping: patience={early_stopping_patience}, monitor='{monitor_metric}'")
        print()

    for epoch in range(1, num_epochs + 1):
        t0 = time.time()
        train_loss = run_train_epoch(model, train_loader, optimizer, scheduler, device, class_weights, grad_clip=grad_clip)
        dev_loss, dev_metrics, _ = evaluate_span_model(model, dev_dataset, dev_loader, dev_records, device, class_weights)
        elapsed = time.time() - t0
        current_metric = float(dev_metrics.get(monitor_metric, 0.0))

        history["train_loss"].append(train_loss)
        history["dev_loss"].append(dev_loss)
        history["dev_metrics"].append(dev_metrics)

        if verbose:
            print(f"Epoca {epoch}/{num_epochs}  [{elapsed:.1f}s]")
            print(f"- Train Loss: {train_loss:.4f} | Dev Loss: {dev_loss:.4f}")
            print(f"- Dev {monitor_metric}: {current_metric:.4f}")

        if current_metric > best_metric:
            best_metric = current_metric
            best_epoch = epoch
            patience_counter = 0
            save_span_model(model, tokenizer, save_dir)
            if verbose:
                print(f"Nuevo mejor modelo guardado ({monitor_metric}={best_metric:.4f})")
        else:
            patience_counter += 1
            if verbose:
                print(f"Sin mejora ({patience_counter}/{early_stopping_patience})")
            if patience_counter >= early_stopping_patience:
                print(f"\n- Early stopping activado en epoca {epoch}")
                break

        if verbose:
            print()

    if verbose:
        print(f"- Entrenamiento finalizado. Mejor {monitor_metric}: {best_metric:.4f}")

    return history, best_metric, best_epoch


def compare_span_models(
    model_names: list[str],
    train_records: list[dict[str, Any]],
    dev_records: list[dict[str, Any]],
    device: torch.device,
    max_length: int,
    stride: int,
    batch_size: int,
    num_epochs: int,
    learning_rate: float,
    warmup_ratio: float,
    weight_decay: float,
    grad_clip: float,
    save_dir: str,
    monitor_metric: str = "rouge1_f1",
    early_stopping_patience: int = 2,
    use_class_weights: bool = True,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    os.makedirs(save_dir, exist_ok=True)

    for model_name in model_names:
        print(f"\n{'=' * 60}")
        print(f"  Evaluando: {model_name}")
        print(f"{'=' * 60}")

        try:
            tokenizer = build_span_tokenizer(model_name)
            train_dataset, train_loader = build_dataloader(
                train_records,
                tokenizer,
                max_length=max_length,
                stride=stride,
                batch_size=batch_size,
                shuffle=True,
                include_labels=True,
            )
            dev_dataset, dev_loader = build_dataloader(
                dev_records,
                tokenizer,
                max_length=max_length,
                stride=stride,
                batch_size=batch_size,
                shuffle=False,
                include_labels=True,
            )

            class_weights = compute_class_weights(train_dataset) if use_class_weights else torch.ones(3)
            model = build_span_model(model_name).to(device).float()

            safe_name = model_name.replace("/", "_")
            model_dir = os.path.join(save_dir, f"{safe_name}_best")

            history, best_metric, best_epoch = train_span_model(
                model=model,
                tokenizer=tokenizer,
                train_dataset=train_dataset,
                train_loader=train_loader,
                dev_dataset=dev_dataset,
                dev_loader=dev_loader,
                dev_records=dev_records,
                device=device,
                class_weights=class_weights,
                num_epochs=num_epochs,
                learning_rate=learning_rate,
                warmup_ratio=warmup_ratio,
                weight_decay=weight_decay,
                grad_clip=grad_clip,
                save_dir=model_dir,
                early_stopping_patience=early_stopping_patience,
                monitor_metric=monitor_metric,
                verbose=True,
            )

            metric_values = [m.get(monitor_metric, 0.0) for m in history["dev_metrics"]]
            best_idx = int(np.argmax(metric_values))

            rows.append(
                {
                    "model": model_name,
                    monitor_metric: round(float(best_metric), 4),
                    "best_epoch": int(best_epoch),
                    "train_loss": round(float(history["train_loss"][best_idx]), 4),
                    "dev_loss": round(float(history["dev_loss"][best_idx]), 4),
                    "status": "ok",
                }
            )

        except Exception as exc:  # noqa: BLE001
            print(f"  Error con {model_name}: {exc}")
            rows.append(
                {
                    "model": model_name,
                    monitor_metric: 0.0,
                    "best_epoch": 0,
                    "train_loss": None,
                    "dev_loss": None,
                    "status": f"error: {exc}",
                }
            )

    results_df = pd.DataFrame(rows).sort_values(monitor_metric, ascending=False).reset_index(drop=True)
    csv_path = os.path.join(save_dir, "model_comparison.csv")
    results_df.to_csv(csv_path, index=False)

    print(f"\n{'=' * 60}")
    print("  Ranking final de modelos")
    print(f"{'=' * 60}")
    print(results_df.to_string(index=False))
    print(f"\nResultados guardados en {csv_path}")

    return results_df


def decode_feature_predictions_tuned(feature: dict[str, Any], logits: np.ndarray, min_span_score: float = 0.0) -> list[SpanCandidate]:
    probs = softmax(logits)
    pred_ids = probs.argmax(axis=-1)
    offsets = feature["offset_mapping"]
    text = feature["text"]
    row_id = feature["row_id"]

    spans: list[SpanCandidate] = []
    current_start = None
    current_end = None
    current_scores: list[float] = []

    for token_idx, (pred_id, (start, end)) in enumerate(zip(pred_ids, offsets)):
        if start == 0 and end == 0:
            continue

        label = ID2LABEL[int(pred_id)]
        token_score = float(probs[token_idx, pred_id])

        if label == "B-SPAN":
            if current_start is not None and current_end is not None and current_end > current_start:
                span_text = text[current_start:current_end].strip()
                if span_text:
                    score = float(np.mean(current_scores))
                    if score >= min_span_score:
                        spans.append(
                            SpanCandidate(row_id=row_id, start=current_start, end=current_end, text=span_text, score=score)
                        )
            current_start = start
            current_end = end
            current_scores = [token_score]
        elif label == "I-SPAN" and current_start is not None:
            current_end = end
            current_scores.append(token_score)
        else:
            if current_start is not None and current_end is not None and current_end > current_start:
                span_text = text[current_start:current_end].strip()
                if span_text:
                    score = float(np.mean(current_scores))
                    if score >= min_span_score:
                        spans.append(
                            SpanCandidate(row_id=row_id, start=current_start, end=current_end, text=span_text, score=score)
                        )
            current_start = None
            current_end = None
            current_scores = []

    if current_start is not None and current_end is not None and current_end > current_start:
        span_text = text[current_start:current_end].strip()
        if span_text:
            score = float(np.mean(current_scores))
            if score >= min_span_score:
                spans.append(SpanCandidate(row_id=row_id, start=current_start, end=current_end, text=span_text, score=score))

    return spans


def overlap_ratio_tuned(a: SpanCandidate, b: SpanCandidate) -> float:
    intersection = max(0, min(a.end, b.end) - max(a.start, b.start))
    if intersection == 0:
        return 0.0
    shortest = max(1, min(a.end - a.start, b.end - b.start))
    return intersection / shortest


def deduplicate_candidates_tuned(
    candidates: list[SpanCandidate],
    max_spans: int = 3,
    overlap_threshold: float = 0.6,
) -> list[SpanCandidate]:
    unique_by_bounds: dict[tuple[int, int], SpanCandidate] = {}
    for candidate in candidates:
        key = (candidate.start, candidate.end)
        previous = unique_by_bounds.get(key)
        if previous is None or candidate.score > previous.score:
            unique_by_bounds[key] = candidate

    ordered = sorted(unique_by_bounds.values(), key=lambda item: (item.score, item.end - item.start), reverse=True)

    selected: list[SpanCandidate] = []
    seen_texts: set[str] = set()
    for candidate in ordered:
        normalized = candidate.text.strip()
        if not normalized or normalized in seen_texts:
            continue
        if any(overlap_ratio_tuned(candidate, chosen) > overlap_threshold for chosen in selected):
            continue
        selected.append(candidate)
        seen_texts.add(normalized)
        if len(selected) == max_spans:
            break

    return sorted(selected, key=lambda item: item.start)


def aggregate_predictions_tuned(
    dataset,
    logits_by_feature_id: dict[int, np.ndarray],
    max_spans: int = 3,
    overlap_threshold: float = 0.6,
    min_span_score: float = 0.0,
) -> dict[str | int, list[SpanCandidate]]:
    grouped: dict[str | int, list[SpanCandidate]] = {}
    for feature_id, logits in logits_by_feature_id.items():
        feature = dataset.features[int(feature_id)]
        row_id = feature["row_id"]
        grouped.setdefault(row_id, []).extend(decode_feature_predictions_tuned(feature, logits, min_span_score=min_span_score))

    return {
        row_id: deduplicate_candidates_tuned(candidates, max_spans=max_spans, overlap_threshold=overlap_threshold)
        for row_id, candidates in grouped.items()
    }


def tune_postprocess(
    dataset,
    logits_by_feature_id: dict[int, np.ndarray],
    records: list[dict[str, Any]],
    max_spans_grid: list[int],
    overlap_threshold_grid: list[float],
    min_span_score_grid: list[float],
) -> tuple[pd.DataFrame, dict[str, float], dict[str, float], dict[str, float], dict[str | int, list[SpanCandidate]]]:
    baseline_predictions = aggregate_predictions(dataset, logits_by_feature_id, max_spans=3)
    baseline_metrics = evaluate_span_predictions(records, baseline_predictions)

    rows: list[dict[str, Any]] = []
    for max_spans in max_spans_grid:
        for overlap_threshold in overlap_threshold_grid:
            for min_span_score in min_span_score_grid:
                preds = aggregate_predictions_tuned(
                    dataset,
                    logits_by_feature_id,
                    max_spans=max_spans,
                    overlap_threshold=overlap_threshold,
                    min_span_score=min_span_score,
                )
                metrics = evaluate_span_predictions(records, preds)
                rows.append(
                    {
                        "max_spans": max_spans,
                        "overlap_threshold": overlap_threshold,
                        "min_span_score": min_span_score,
                        **metrics,
                    }
                )

    tuning_results = pd.DataFrame(rows).sort_values("rouge1_f1", ascending=False).reset_index(drop=True)
    best_row = tuning_results.iloc[0]

    best_config = {
        "max_spans": int(best_row["max_spans"]),
        "overlap_threshold": float(best_row["overlap_threshold"]),
        "min_span_score": float(best_row["min_span_score"]),
    }

    best_predictions = aggregate_predictions_tuned(
        dataset,
        logits_by_feature_id,
        max_spans=best_config["max_spans"],
        overlap_threshold=best_config["overlap_threshold"],
        min_span_score=best_config["min_span_score"],
    )
    best_metrics = evaluate_span_predictions(records, best_predictions)

    return tuning_results, best_config, baseline_metrics, best_metrics, best_predictions
