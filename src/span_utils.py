from __future__ import annotations

import json
import math
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

from src.shared.dataset import load_raw_records
from src.shared.schemas import EMPTY_SPAN_LABELS, HOPEFUL_LABELS


SPAN_LABELS = ("O", "B-SPAN", "I-SPAN")
LABEL2ID = {label: idx for idx, label in enumerate(SPAN_LABELS)}
ID2LABEL = {idx: label for label, idx in LABEL2ID.items()}


@dataclass(slots=True)
class SpanRange:
    start: int
    end: int
    text: str


@dataclass(slots=True)
class SpanCandidate:
    row_id: str | int
    start: int
    end: int
    text: str
    score: float


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_source_text(record: dict[str, Any]) -> str:
    title = str(record.get("title", "") or "")
    selftext = str(record.get("selftext", "") or "")
    if title and selftext:
        return f"{title}\n{selftext}"
    return title or selftext


def load_records(path: str | Path) -> list[dict[str, Any]]:
    return load_raw_records(path)


def train_dev_split(
    records: list[dict[str, Any]],
    dev_size: float = 0.1,
    seed: int = 42,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not 0.0 < dev_size < 1.0:
        raise ValueError("dev_size must be between 0 and 1")

    indices = list(range(len(records)))
    rng = random.Random(seed)
    rng.shuffle(indices)
    cut = max(1, int(len(indices) * dev_size))
    dev_indices = set(indices[:cut])
    train_records = [record for idx, record in enumerate(records) if idx not in dev_indices]
    dev_records = [record for idx, record in enumerate(records) if idx in dev_indices]
    return train_records, dev_records


def find_all_occurrences(text: str, target: str) -> list[tuple[int, int]]:
    matches: list[tuple[int, int]] = []
    start = 0
    while target and start < len(text):
        idx = text.find(target, start)
        if idx == -1:
            break
        matches.append((idx, idx + len(target)))
        start = idx + 1
    return matches


def overlaps(a_start: int, a_end: int, b_start: int, b_end: int) -> bool:
    return max(a_start, b_start) < min(a_end, b_end)


def align_gold_spans(record: dict[str, Any]) -> list[SpanRange]:
    """
    Extrae y normaliza los fragmentos anotados (gold spans) de un registro.
    
    Combina el título y el cuerpo del mensaje, y busca las posiciones exactas
    de inicio y fin de cada anotación en el texto combinado para facilitar el 
    etiquetado de tokens. Maneja posibles solapamientos y anotaciones duplicadas.
    """

    text = build_source_text(record)
    annotations = record.get("span_annotations") or []
    used_ranges: list[tuple[int, int]] = []
    aligned: list[SpanRange] = []

    for annotation in annotations:
        span_text = str(annotation.get("span", "") or "").strip()
        if not span_text:
            continue

        candidates = find_all_occurrences(text, span_text)
        chosen = None
        for start, end in candidates:
            if not any(overlaps(start, end, used_start, used_end) for used_start, used_end in used_ranges):
                chosen = (start, end)
                break

        if chosen is None and candidates:
            chosen = candidates[0]

        if chosen is None:
            continue

        used_ranges.append(chosen)
        aligned.append(SpanRange(start=chosen[0], end=chosen[1], text=span_text))

    return aligned


def compute_token_label(token_start: int, token_end: int, spans: list[SpanRange]) -> int:
    for span in spans:
        if token_end <= span.start or token_start >= span.end:
            continue
        if token_start <= span.start < token_end or token_start == span.start:
            return LABEL2ID["B-SPAN"]
        return LABEL2ID["I-SPAN"]
    return LABEL2ID["O"]


class SpanTokenClassificationDataset(Dataset):
    def __init__(self, features: list[dict[str, Any]], include_labels: bool = True):
        self.features = features
        self.include_labels = include_labels

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        feature = self.features[idx]
        item = {
            "input_ids": torch.tensor(feature["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(feature["attention_mask"], dtype=torch.long),
            "feature_id": torch.tensor(idx, dtype=torch.long),
        }
        if self.include_labels and "labels" in feature:
            item["labels"] = torch.tensor(feature["labels"], dtype=torch.long)
        if "token_type_ids" in feature:
            item["token_type_ids"] = torch.tensor(feature["token_type_ids"], dtype=torch.long)
        return item


def build_span_features(
    records: list[dict[str, Any]],
    tokenizer: AutoTokenizer,
    max_length: int = 512,
    stride: int = 128,
    include_labels: bool = True,
) -> list[dict[str, Any]]:
    if not tokenizer.is_fast:
        raise ValueError("Task C requires a fast tokenizer to recover offsets.")

    features: list[dict[str, Any]] = []
    for example_index, record in enumerate(records):
        text = build_source_text(record)
        aligned_spans = align_gold_spans(record) if include_labels else []

        tokenized = tokenizer(
            text,
            max_length=max_length,
            truncation=True,
            stride=stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
        )

        num_chunks = len(tokenized["input_ids"])
        for chunk_index in range(num_chunks):
            encoding = tokenized.encodings[chunk_index]
            offsets = tokenized["offset_mapping"][chunk_index]
            labels = []

            if include_labels:
                for token_offset, sequence_id in zip(offsets, encoding.sequence_ids):
                    token_start, token_end = token_offset
                    if sequence_id is None or (token_start == 0 and token_end == 0):
                        labels.append(-100)
                    else:
                        labels.append(compute_token_label(token_start, token_end, aligned_spans))

            feature = {
                "row_id": record.get("row_id"),
                "example_index": example_index,
                "chunk_index": chunk_index,
                "text": text,
                "input_ids": tokenized["input_ids"][chunk_index],
                "attention_mask": tokenized["attention_mask"][chunk_index],
                "offset_mapping": offsets,
            }
            if "token_type_ids" in tokenized:
                feature["token_type_ids"] = tokenized["token_type_ids"][chunk_index]
            if include_labels:
                feature["labels"] = labels
                feature["gold_spans"] = [span.text for span in aligned_spans]
            features.append(feature)

    return features


def build_dataloader(
    records: list[dict[str, Any]],
    tokenizer: AutoTokenizer,
    max_length: int = 512,
    stride: int = 128,
    batch_size: int = 8,
    shuffle: bool = False,
    include_labels: bool = True,
) -> tuple[SpanTokenClassificationDataset, DataLoader]:
    """
    Crea un DataLoader para clasificación de tokens manejando textos largos.
    
    Si un texto excede 'max_length', se divide en múltiples ventanas (chunks)
    con un solapamiento definido por 'stride' para asegurar que los fragmentos
    no se pierdan en los puntos de corte.
    
    """
    dataset = SpanTokenClassificationDataset(
        build_span_features(
            records=records,
            tokenizer=tokenizer,
            max_length=max_length,
            stride=stride,
            include_labels=include_labels,
        ),
        include_labels=include_labels,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    return dataset, loader


def compute_class_weights(dataset: SpanTokenClassificationDataset) -> torch.Tensor:
    counts = Counter()
    for feature in dataset.features:
        for label in feature.get("labels", []):
            if label >= 0:
                counts[label] += 1

    total = sum(counts.values())
    weights = []
    for label_id in range(len(SPAN_LABELS)):
        count = max(counts.get(label_id, 0), 1)
        weights.append(total / (len(SPAN_LABELS) * count))
    return torch.tensor(weights, dtype=torch.float32)


def decode_feature_predictions(feature: dict[str, Any], logits: np.ndarray) -> list[SpanCandidate]:
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
                    spans.append(
                        SpanCandidate(
                            row_id=row_id,
                            start=current_start,
                            end=current_end,
                            text=span_text,
                            score=float(np.mean(current_scores)),
                        )
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
                    spans.append(
                        SpanCandidate(
                            row_id=row_id,
                            start=current_start,
                            end=current_end,
                            text=span_text,
                            score=float(np.mean(current_scores)),
                        )
                    )
            current_start = None
            current_end = None
            current_scores = []

    if current_start is not None and current_end is not None and current_end > current_start:
        span_text = text[current_start:current_end].strip()
        if span_text:
            spans.append(
                SpanCandidate(
                    row_id=row_id,
                    start=current_start,
                    end=current_end,
                    text=span_text,
                    score=float(np.mean(current_scores)),
                )
            )

    return spans


def overlap_ratio(a: SpanCandidate, b: SpanCandidate) -> float:
    intersection = max(0, min(a.end, b.end) - max(a.start, b.start))
    if intersection == 0:
        return 0.0
    shortest = max(1, min(a.end - a.start, b.end - b.start))
    return intersection / shortest


def deduplicate_candidates(candidates: list[SpanCandidate], max_spans: int = 3) -> list[SpanCandidate]:
    unique_by_bounds: dict[tuple[int, int], SpanCandidate] = {}
    for candidate in candidates:
        key = (candidate.start, candidate.end)
        previous = unique_by_bounds.get(key)
        if previous is None or candidate.score > previous.score:
            unique_by_bounds[key] = candidate

    ordered = sorted(
        unique_by_bounds.values(),
        key=lambda item: (item.score, item.end - item.start),
        reverse=True,
    )

    selected: list[SpanCandidate] = []
    seen_texts: set[str] = set()
    for candidate in ordered:
        normalized = candidate.text.strip()
        if not normalized or normalized in seen_texts:
            continue
        if any(overlap_ratio(candidate, chosen) > 0.6 for chosen in selected):
            continue
        selected.append(candidate)
        seen_texts.add(normalized)
        if len(selected) == max_spans:
            break

    return sorted(selected, key=lambda item: item.start)


def aggregate_predictions(
    dataset: SpanTokenClassificationDataset,
    logits_by_feature_id: dict[int, np.ndarray],
    max_spans: int = 3,
) -> dict[str | int, list[SpanCandidate]]:
    grouped: dict[str | int, list[SpanCandidate]] = {}
    for feature_id, logits in logits_by_feature_id.items():
        feature = dataset.features[int(feature_id)]
        row_id = feature["row_id"]
        grouped.setdefault(row_id, []).extend(decode_feature_predictions(feature, logits))

    return {
        row_id: deduplicate_candidates(candidates, max_spans=max_spans)
        for row_id, candidates in grouped.items()
    }


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.clip(exp.sum(axis=-1, keepdims=True), 1e-12, None)


def tokenize_for_rouge(text: str) -> list[str]:
    return [token for token in text.lower().split() if token]


def rouge1_f1(prediction: str, reference: str) -> float:
    pred_tokens = tokenize_for_rouge(prediction)
    ref_tokens = tokenize_for_rouge(reference)
    if not pred_tokens and not ref_tokens:
        return 1.0
    if not pred_tokens or not ref_tokens:
        return 0.0

    pred_counts = Counter(pred_tokens)
    ref_counts = Counter(ref_tokens)
    overlap = sum(min(pred_counts[token], ref_counts[token]) for token in pred_counts)
    precision = overlap / len(pred_tokens)
    recall = overlap / len(ref_tokens)
    if precision + recall == 0.0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def evaluate_span_predictions(
    records: list[dict[str, Any]],
    predictions: dict[str | int, list[SpanCandidate]],
) -> dict[str, float]:
    rouge_scores: list[float] = []

    for record in records:
        row_id = record.get("row_id")
        gold_spans = [span.text for span in align_gold_spans(record)]
        pred_spans = [candidate.text for candidate in predictions.get(row_id, [])]

        if gold_spans or pred_spans:
            joined_pred = " ||| ".join(pred_spans)
            joined_gold = " ||| ".join(gold_spans)
            rouge_scores.append(rouge1_f1(joined_pred, joined_gold))
        else:
            rouge_scores.append(1.0)

    return {
        "rouge1_f1": float(np.mean(rouge_scores)) if rouge_scores else 0.0,
    }


def collect_logits(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> dict[int, np.ndarray]:
    model.eval()
    logits_by_feature_id: dict[int, np.ndarray] = {}

    with torch.no_grad():
        for batch in dataloader:
            feature_ids = batch["feature_id"].cpu().numpy().tolist()
            model_inputs = {
                "input_ids": batch["input_ids"].to(device),
                "attention_mask": batch["attention_mask"].to(device),
            }
            if "token_type_ids" in batch:
                model_inputs["token_type_ids"] = batch["token_type_ids"].to(device)

            outputs = model(**model_inputs)
            batch_logits = outputs.logits.detach().cpu().numpy()
            for feature_id, logits in zip(feature_ids, batch_logits):
                logits_by_feature_id[int(feature_id)] = logits

    return logits_by_feature_id


def load_primary_labels(path: str | Path) -> dict[str | int, str]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        content = handle.read().strip()

    if not content:
        return {}

    try:
        payload = json.loads(content)
        records = payload if isinstance(payload, list) else payload.get("data", [])
    except json.JSONDecodeError:
        records = [json.loads(line) for line in content.splitlines() if line.strip()]

    mapping: dict[str | int, str] = {}
    for record in records:
        row_id = record.get("row_id", record.get("id"))
        label = record.get("primary_label")
        if row_id is not None and label is not None:
            mapping[row_id] = label
    return mapping


def should_force_empty(record: dict[str, Any], primary_label_mapping: dict[str | int, str] | None = None) -> bool:
    row_id = record.get("row_id")
    label = record.get("primary_label")
    if primary_label_mapping and row_id in primary_label_mapping:
        label = primary_label_mapping[row_id]
    return label in EMPTY_SPAN_LABELS


def predictions_to_submission(
    records: list[dict[str, Any]],
    predictions: dict[str | int, list[SpanCandidate]],
    primary_label_mapping: dict[str | int, str] | None = None,
) -> list[dict[str, Any]]:
    submission = []
    for record in records:
        row_id = record.get("row_id")
        if should_force_empty(record, primary_label_mapping=primary_label_mapping):
            spans = []
        else:
            spans = [{"span": candidate.text} for candidate in predictions.get(row_id, [])]
        submission.append({"row_id": row_id, "span_annotations": spans})
    return submission


def save_json(payload: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def limit_records(records: list[dict[str, Any]], max_examples: int | None) -> list[dict[str, Any]]:
    if max_examples is None:
        return records
    return records[:max_examples]


def is_hopeful_record(record: dict[str, Any]) -> bool:
    return record.get("primary_label") in HOPEFUL_LABELS


def summarize_dataset(records: Iterable[dict[str, Any]]) -> dict[str, int]:
    total = 0
    hopeful = 0
    with_spans = 0
    for record in records:
        total += 1
        if is_hopeful_record(record):
            hopeful += 1
        if record.get("span_annotations"):
            with_spans += 1
    return {
        "total_records": total,
        "hopeful_records": hopeful,
        "records_with_spans": with_spans,
    }


def format_metrics(metrics: dict[str, float]) -> str:
    return " | ".join(
        f"{name}={value:.4f}" for name, value in metrics.items() if isinstance(value, (int, float)) and not math.isnan(value)
    )
