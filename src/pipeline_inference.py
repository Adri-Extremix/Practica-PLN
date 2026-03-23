from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
from typing import Any, Callable

import torch

from src.data_utils import (
    PRIMARY_LABELS,
    format_predictions_for_submission,
    format_predictions_for_submission_hope,
    load_split,
)
from src.dataset import build_dataloader as build_cls_dataloader
from src.model import build_tokenizer, load_model
from src.shared.schemas import EMPTY_SPAN_LABELS
from src.span_experiments import aggregate_predictions_tuned
from src.span_model import build_span_tokenizer, load_span_model
from src.span_utils import (
    collect_logits,
    load_primary_labels,
    load_records,
    predictions_to_submission,
)
from src.trainer import predict


def load_json_file(path: Path) -> Any:
    content = path.read_text(encoding="utf-8").strip()
    if not content:
        return []
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        return [json.loads(line) for line in content.splitlines() if line.strip()]


def save_json_file(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_optional_config(path: Path | None) -> dict[str, Any]:
    if not path or not path.exists():
        return {}
    payload = load_json_file(path)
    return payload if isinstance(payload, dict) else {}


def pick_existing_path(candidates: list[Path]) -> Path | None:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def normalize_row_id(value: Any) -> str:
    return str(value)


def records_to_id_map(records: list[dict[str, Any]], id_keys: tuple[str, ...] = ("row_id", "id")) -> dict[str, dict[str, Any]]:
    mapping: dict[str, dict[str, Any]] = {}
    for record in records:
        row_id = None
        for key in id_keys:
            if key in record and record[key] is not None:
                row_id = normalize_row_id(record[key])
                break
        if row_id is not None:
            mapping[row_id] = record
    return mapping


def validate_coverage(task_name: str, expected_ids: list[str], predicted: dict[str, Any]) -> None:
    missing = [row_id for row_id in expected_ids if row_id not in predicted]
    extras = [row_id for row_id in predicted if row_id not in set(expected_ids)]
    if missing:
        raise RuntimeError(f"{task_name}: faltan {len(missing)} ids en predicciones.")
    if extras:
        print(f"[WARN] {task_name}: {len(extras)} ids extra serán ignorados.")


def load_task_predictions_fallback(path: Path) -> list[dict[str, Any]]:
    payload = load_json_file(path)
    if not isinstance(payload, list):
        raise RuntimeError(f"Fallback inválido en {path}: se esperaba una lista JSON.")
    return payload


def run_task_a(
    test_path: Path,
    output_dir: Path,
    model_path: Path,
    config_path: Path | None,
    fallback_json: Path | None,
    device: torch.device,
) -> tuple[list[dict[str, Any]], str]:
    test_df = load_split(str(test_path), combine_title=True)
    expected_ids = [normalize_row_id(x) for x in test_df["id"].tolist()]

    if model_path.exists():
        cfg = load_optional_config(config_path)
        model_name = cfg.get("model_name", "microsoft/mdeberta-v3-base")
        dropout = float(cfg.get("dropout", 0.1))
        max_length = int(cfg.get("max_length", 512))
        batch_size = int(cfg.get("batch_size", 16))

        tokenizer = build_tokenizer(model_name)
        loader = build_cls_dataloader(
            texts=test_df["text"].tolist(),
            tokenizer=tokenizer,
            labels=None,
            max_length=max_length,
            batch_size=batch_size,
            shuffle=False,
        )
        model = load_model(
            path=str(model_path),
            model_name=model_name,
            num_labels=len(PRIMARY_LABELS),
            dropout_prob=dropout,
            device=device,
        ).float()
        probs = predict(model, loader, device)
        submission_df = format_predictions_for_submission_hope(
            ids=test_df["id"].tolist(),
            predictions=probs,
            label_set=PRIMARY_LABELS,
        )
        raw_preds = submission_df.to_dict(orient="records")
        source = f"model:{model_path}"
    elif fallback_json and fallback_json.exists():
        raw_preds = load_task_predictions_fallback(fallback_json)
        source = f"fallback_json:{fallback_json}"
    else:
        raise RuntimeError("Task A: no hay modelo ni fallback JSON disponible.")

    mapped = records_to_id_map(raw_preds, id_keys=("row_id", "id"))
    validate_coverage("Task A", expected_ids, mapped)

    result = [{"row_id": row_id, "primary_label": mapped[row_id]["primary_label"]} for row_id in expected_ids]
    save_json_file(result, output_dir / "task_a_predictions.json")
    return result, source


def run_task_b(
    test_path: Path,
    output_dir: Path,
    model_path: Path,
    config_path: Path | None,
    fallback_json: Path | None,
    device: torch.device,
) -> tuple[list[dict[str, Any]], str]:
    test_df = load_split(str(test_path), combine_title=True)
    expected_ids = [normalize_row_id(x) for x in test_df["id"].tolist()]

    if model_path.exists():
        cfg = load_optional_config(config_path)
        model_name = cfg.get("model_name", "microsoft/mdeberta-v3-base")
        dropout = float(cfg.get("dropout", 0.1))
        max_length = int(cfg.get("max_length", 512))
        batch_size = int(cfg.get("batch_size", 16))
        threshold = float(cfg.get("optimal_threshold", 0.5))

        tokenizer = build_tokenizer(model_name)
        loader = build_cls_dataloader(
            texts=test_df["text"].tolist(),
            tokenizer=tokenizer,
            labels=None,
            max_length=max_length,
            batch_size=batch_size,
            shuffle=False,
        )
        model = load_model(
            path=str(model_path),
            model_name=model_name,
            num_labels=7,
            dropout_prob=dropout,
            device=device,
        ).float()
        probs = predict(model, loader, device)
        submission_df = format_predictions_for_submission(
            ids=test_df["id"].tolist(),
            predictions=probs,
            threshold=threshold,
        )
        raw_preds = submission_df.to_dict(orient="records")
        source = f"model:{model_path}"
    elif fallback_json and fallback_json.exists():
        raw_preds = load_task_predictions_fallback(fallback_json)
        source = f"fallback_json:{fallback_json}"
    else:
        raise RuntimeError("Task B: no hay modelo ni fallback JSON disponible.")

    mapped = records_to_id_map(raw_preds, id_keys=("row_id", "id"))
    validate_coverage("Task B", expected_ids, mapped)

    result = [
        {
            "row_id": row_id,
            "trigger_emotions": mapped[row_id].get("trigger_emotions", []),
        }
        for row_id in expected_ids
    ]
    save_json_file(result, output_dir / "task_b_predictions.json")
    return result, source


def run_task_c(
    test_path: Path,
    output_dir: Path,
    model_dir: Path,
    config_path: Path | None,
    fallback_json: Path | None,
    device: torch.device,
    task_a_predictions: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], str]:
    records = load_records(test_path)
    expected_ids = [normalize_row_id(record["row_id"]) for record in records]

    if model_dir.exists() and model_dir.is_dir():
        cfg = load_optional_config(config_path)
        model_name = cfg.get("model_name", "microsoft/mdeberta-v3-base")
        max_length = int(cfg.get("max_length", 128))
        stride = int(cfg.get("stride", 32))
        batch_size = int(cfg.get("batch_size", 8))
        tuned = cfg.get("optimal_postprocess", {}) if isinstance(cfg.get("optimal_postprocess", {}), dict) else {}
        max_spans = int(tuned.get("max_spans", 3))
        overlap_threshold = float(tuned.get("overlap_threshold", 0.6))
        min_span_score = float(tuned.get("min_span_score", 0.0))

        tokenizer = build_span_tokenizer(model_name)
        from src.span_utils import build_dataloader as build_span_dataloader

        test_dataset, test_loader = build_span_dataloader(
            records,
            tokenizer,
            max_length=max_length,
            stride=stride,
            batch_size=batch_size,
            shuffle=False,
            include_labels=False,
        )
        model = load_span_model(str(model_dir), device=device).float()
        test_logits = collect_logits(model, test_loader, device)
        preds = aggregate_predictions_tuned(
            test_dataset,
            test_logits,
            max_spans=max_spans,
            overlap_threshold=overlap_threshold,
            min_span_score=min_span_score,
        )

        a_map = {normalize_row_id(row["row_id"]): row["primary_label"] for row in task_a_predictions}
        a_native: dict[str | int, str] = {}
        for record in records:
            rid = record["row_id"]
            label = a_map.get(normalize_row_id(rid))
            if label is not None:
                a_native[rid] = label

        raw_preds = predictions_to_submission(records, preds, primary_label_mapping=a_native)
        source = f"model:{model_dir}"
    elif fallback_json and fallback_json.exists():
        raw_preds = load_task_predictions_fallback(fallback_json)
        source = f"fallback_json:{fallback_json}"
    else:
        raise RuntimeError("Task C: no hay modelo ni fallback JSON disponible.")

    mapped = records_to_id_map(raw_preds, id_keys=("row_id", "id"))
    validate_coverage("Task C", expected_ids, mapped)

    a_map = {normalize_row_id(row["row_id"]): row["primary_label"] for row in task_a_predictions}
    result = []
    for row_id in expected_ids:
        label = a_map.get(row_id)
        spans = mapped[row_id].get("span_annotations", [])
        if label in EMPTY_SPAN_LABELS:
            spans = []
        result.append({"row_id": row_id, "span_annotations": spans[:3]})

    save_json_file(result, output_dir / "task_c_predictions.json")
    return result, source


def load_adapter(adapter_spec: str) -> Callable[..., list[dict[str, Any]]]:
    if ":" not in adapter_spec:
        raise ValueError("El adapter debe tener formato module:function")
    module_name, func_name = adapter_spec.split(":", 1)
    module = importlib.import_module(module_name)
    fn = getattr(module, func_name, None)
    if fn is None or not callable(fn):
        raise ValueError(f"No se encontró callable '{func_name}' en módulo '{module_name}'.")
    return fn


def run_task_d(
    output_dir: Path,
    test_ids: list[str],
    spans_from_c: list[dict[str, Any]],
    model_path_or_dir: Path | None,
    config_path: Path | None,
    fallback_json: Path | None,
    adapter_spec: str | None,
) -> tuple[list[dict[str, Any]], str]:
    raw_preds: list[dict[str, Any]] | None = None
    source = ""

    has_model = bool(model_path_or_dir and model_path_or_dir.exists())
    if has_model and adapter_spec:
        adapter = load_adapter(adapter_spec)
        raw_preds = adapter(
            model_path=str(model_path_or_dir),
            config_path=str(config_path) if config_path else None,
            spans=spans_from_c,
            row_ids=test_ids,
        )
        source = f"model+adapter:{model_path_or_dir}"
    elif has_model and not adapter_spec:
        print("[WARN] Task D: modelo detectado pero sin adapter de inferencia; se intentará fallback JSON.")

    if raw_preds is None:
        if fallback_json and fallback_json.exists():
            raw_preds = load_task_predictions_fallback(fallback_json)
            source = f"fallback_json:{fallback_json}"
        else:
            raise RuntimeError(
                "Task D: no hay inferencia disponible. Añade --task-d-adapter module:function o un fallback JSON de D."
            )

    mapped = records_to_id_map(raw_preds, id_keys=("row_id", "id"))
    validate_coverage("Task D", test_ids, mapped)

    result = []
    for row_id in test_ids:
        annotations = mapped[row_id].get("span_annotations", [])
        result.append({"row_id": row_id, "span_annotations": annotations})
    save_json_file(result, output_dir / "task_d_predictions.json")
    return result, source


def merge_c_and_d(
    c_predictions: list[dict[str, Any]],
    d_predictions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    d_map = {normalize_row_id(row["row_id"]): row.get("span_annotations", []) for row in d_predictions}
    merged: list[dict[str, Any]] = []

    for row in c_predictions:
        row_id = normalize_row_id(row["row_id"])
        c_spans = row.get("span_annotations", [])
        d_spans = d_map.get(row_id, [])

        text_to_labels: dict[str, list[dict[str, str]]] = {}
        for ann in d_spans:
            span_text = str(ann.get("span", ""))
            text_to_labels.setdefault(span_text, []).append(
                {
                    "outcome_stance": ann.get("outcome_stance", "Desired"),
                    "actor": ann.get("actor", "Unclear"),
                }
            )

        merged_annotations = []
        for idx, c_ann in enumerate(c_spans[:3]):
            span_text = str(c_ann.get("span", ""))
            labels = text_to_labels.get(span_text, [])
            if labels:
                label_info = labels.pop(0)
            elif idx < len(d_spans):
                label_info = {
                    "outcome_stance": d_spans[idx].get("outcome_stance", "Desired"),
                    "actor": d_spans[idx].get("actor", "Unclear"),
                }
            else:
                label_info = {"outcome_stance": "Desired", "actor": "Unclear"}

            merged_annotations.append(
                {
                    "span": span_text,
                    "outcome_stance": label_info["outcome_stance"],
                    "actor": label_info["actor"],
                }
            )

        merged.append({"row_id": row_id, "span_annotations": merged_annotations})

    return merged


def build_final_submission(
    row_ids: list[str],
    a_preds: list[dict[str, Any]],
    b_preds: list[dict[str, Any]],
    c_with_d: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    a_map = {normalize_row_id(row["row_id"]): row["primary_label"] for row in a_preds}
    b_map = {normalize_row_id(row["row_id"]): row.get("trigger_emotions", []) for row in b_preds}
    c_map = {normalize_row_id(row["row_id"]): row.get("span_annotations", []) for row in c_with_d}

    final = []
    for row_id in row_ids:
        final.append(
            {
                "row_id": row_id,
                "primary_label": a_map[row_id],
                "trigger_emotions": b_map[row_id],
                "span_annotations": c_map[row_id],
            }
        )
    return final


def main() -> None:
    parser = argparse.ArgumentParser(description="Pipeline de inferencia HOPE-EXP A->B->C->D con fallback tolerante.")
    parser.add_argument("--test-path", type=Path, default=Path("./HopeEXP_Test_unlabeled.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("./outputs/pipeline/submission_final.json"))
    parser.add_argument("--work-dir", type=Path, default=Path("./outputs/pipeline"))

    parser.add_argument("--task-a-model", type=Path, default=Path("./outputs/best_model_task_a.pt"))
    parser.add_argument("--task-a-config", type=Path, default=Path("./outputs/experiment_config.json"))
    parser.add_argument("--task-a-fallback", type=Path, default=Path("./outputs/submission_task_a.json"))

    parser.add_argument("--task-b-model", type=Path, default=Path("./outputs/best_model_task_b.pt"))
    parser.add_argument("--task-b-config", type=Path, default=Path("./outputs/task_b/experiment_config.json"))
    parser.add_argument("--task-b-fallback", type=Path, default=Path("./outputs/task_b/submission_task_b.json"))

    parser.add_argument("--task-c-model-dir", type=Path, default=Path("./outputs/task_c_spans_notebook/best_model"))
    parser.add_argument("--task-c-config", type=Path, default=Path("./outputs/task_c_spans_notebook/experiment_config.json"))
    parser.add_argument(
        "--task-c-fallback",
        type=Path,
        default=Path("./outputs/task_c_spans_notebook/submission_task_c_notebook.json"),
    )

    parser.add_argument("--task-d-model", type=Path, default=Path("./outputs/task_d/best_model"))
    parser.add_argument("--task-d-config", type=Path, default=Path("./outputs/task_d/experiment_config.json"))
    parser.add_argument("--task-d-fallback", type=Path, default=Path("./outputs/task_d/submission_task_d.json"))
    parser.add_argument(
        "--task-d-adapter",
        type=str,
        default=None,
        help="Callable para inferencia D con formato module:function",
    )

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.work_dir.mkdir(parents=True, exist_ok=True)

    status: dict[str, Any] = {
        "device": str(device),
        "tasks": {},
        "output": str(args.output),
    }

    test_records = load_records(args.test_path)
    row_ids = [normalize_row_id(record["row_id"]) for record in test_records]

    try:
        print("[Task A] Verificando artefactos e inferencia...")
        task_a, source_a = run_task_a(
            test_path=args.test_path,
            output_dir=args.work_dir,
            model_path=args.task_a_model,
            config_path=args.task_a_config,
            fallback_json=args.task_a_fallback,
            device=device,
        )
        status["tasks"]["A"] = {"status": "ok", "source": source_a}

        print("[Task B] Verificando artefactos e inferencia...")
        task_b, source_b = run_task_b(
            test_path=args.test_path,
            output_dir=args.work_dir,
            model_path=args.task_b_model,
            config_path=args.task_b_config,
            fallback_json=args.task_b_fallback,
            device=device,
        )
        status["tasks"]["B"] = {"status": "ok", "source": source_b}

        print("[Task C] Verificando artefactos e inferencia...")
        task_c, source_c = run_task_c(
            test_path=args.test_path,
            output_dir=args.work_dir,
            model_dir=args.task_c_model_dir,
            config_path=args.task_c_config,
            fallback_json=args.task_c_fallback,
            device=device,
            task_a_predictions=task_a,
        )
        status["tasks"]["C"] = {"status": "ok", "source": source_c}

        print("[Task D] Verificando artefactos e inferencia...")
        task_d, source_d = run_task_d(
            output_dir=args.work_dir,
            test_ids=row_ids,
            spans_from_c=task_c,
            model_path_or_dir=args.task_d_model,
            config_path=args.task_d_config,
            fallback_json=args.task_d_fallback,
            adapter_spec=args.task_d_adapter,
        )
        status["tasks"]["D"] = {"status": "ok", "source": source_d}

        c_with_d = merge_c_and_d(task_c, task_d)
        final_submission = build_final_submission(row_ids, task_a, task_b, c_with_d)
        save_json_file(final_submission, args.output)
        status["final_status"] = "ok"
        status["final_records"] = len(final_submission)
        print(f"Submission final guardada en: {args.output}")

    except Exception as exc:  # noqa: BLE001
        status["final_status"] = "error"
        status["error"] = str(exc)
        print(f"[ERROR] {exc}")

    save_json_file(status, args.work_dir / "pipeline_report.json")
    print(f"Reporte guardado en: {args.work_dir / 'pipeline_report.json'}")


if __name__ == "__main__":
    main()
