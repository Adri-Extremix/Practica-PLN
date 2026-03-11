"""
metrics.py
==========
Cálculo de métricas para clasificación multietiqueta de emociones.
"""

import numpy as np
import torch
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    classification_report,
    hamming_loss,
)
from typing import List, Dict, Optional, Tuple

from src.data_utils import EMOTION_LABELS


# ──────────────────────────────────────────────
# Binarización de predicciones
# ──────────────────────────────────────────────

def binarize(probs: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """
    Convierte probabilidades a predicciones binarias.

    Args:
        probs: Array [n_samples, n_labels] con probabilidades.
        threshold: Umbral de decisión.

    Returns:
        Array binario [n_samples, n_labels].
    """
    return (probs >= threshold).astype(int)


# ──────────────────────────────────────────────
# Métricas principales
# ──────────────────────────────────────────────

def compute_metrics(
    y_true: np.ndarray,
    y_pred_probs: np.ndarray,
    threshold: float = 0.5,
    label_names: List[str] = EMOTION_LABELS,
) -> Dict[str, float]:
    """
    Calcula métricas completas de clasificación multietiqueta.

    Args:
        y_true:       Array binario ground-truth [n_samples, n_labels].
        y_pred_probs: Array de probabilidades [n_samples, n_labels].
        threshold:    Umbral de binarización.
        label_names:  Nombres de las etiquetas para el report.

    Returns:
        Dict con f1_macro, f1_micro, f1_weighted, precision_macro,
        recall_macro, hamming_loss, y f1 por clase.
    """

    y_pred = binarize(y_pred_probs, threshold)

    metrics = {
        "f1_macro":      f1_score(y_true, y_pred, average="macro",    zero_division=0),
        "f1_micro":      f1_score(y_true, y_pred, average="micro",    zero_division=0),
        "f1_weighted":   f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro":    recall_score(y_true, y_pred, average="macro",    zero_division=0),
        "hamming_loss":    hamming_loss(y_true, y_pred),
        "threshold":       threshold,
    }

    # F1 por clase
    f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
    for name, val in zip(label_names, f1_per_class):
        metrics[f"f1_{name}"] = val

    return metrics


def print_metrics(metrics: Dict[str, float], label_names: List[str] = EMOTION_LABELS) -> None:
    """Imprime un resumen legible de las métricas."""

    print("=" * 50)
    print(f"  F1 Macro:    {metrics['f1_macro']:.4f}")
    print(f"  F1 Micro:    {metrics['f1_micro']:.4f}")
    print(f"  F1 Weighted: {metrics['f1_weighted']:.4f}")
    print(f"  Precision:   {metrics['precision_macro']:.4f}")
    print(f"  Recall:      {metrics['recall_macro']:.4f}")
    print(f"  Hamming Loss:{metrics['hamming_loss']:.4f}")
    print(f"  Threshold:   {metrics['threshold']:.2f}")
    print("-" * 50)
    print("  F1 por emoción:")
    for label in label_names:
        key = f"f1_{label}"
        if key in metrics:
            print(f"    {label:12s}: {metrics[key]:.4f}")
    print("=" * 50)


def classification_report_str(
    y_true: np.ndarray,
    y_pred_probs: np.ndarray,
    threshold: float = 0.5,
    label_names: List[str] = EMOTION_LABELS,
) -> str:
    """Devuelve el classification_report de sklearn como string."""

    y_pred = binarize(y_pred_probs, threshold)
    return classification_report(
        y_true, y_pred,
        target_names=label_names,
        zero_division=0,
    )


# ──────────────────────────────────────────────
# Búsqueda del umbral óptimo
# ──────────────────────────────────────────────

def find_best_threshold(
    y_true: np.ndarray,
    y_pred_probs: np.ndarray,
    thresholds: Optional[List[float]] = None,
    metric: str = "f1_macro",
    verbose: bool = True,
) -> Tuple[float, float]:
    """
    Busca el umbral que maximiza la métrica dada en validación.

    Args:
        y_true:       Array binario ground-truth [n_samples, n_labels].
        y_pred_probs: Array de probabilidades [n_samples, n_labels].
        thresholds:   Lista de umbrales a probar. Por defecto 0.1 a 0.9.
        metric:       Métrica a maximizar ('f1_macro', 'f1_micro', etc.).
        verbose:      Si True, imprime resultados por umbral.

    Returns:
        Tupla (best_threshold, best_metric_value).
    """

    if thresholds is None:
        thresholds = np.arange(0.1, 0.91, 0.05).tolist()

    best_threshold = 0.5
    best_value     = -1.0

    if verbose:
        print(f"{'Threshold':>12} | {metric:>12}")
        print("-" * 30)

    for t in thresholds:
        m = compute_metrics(y_true, y_pred_probs, threshold=t)
        val = m[metric]
        if verbose:
            print(f"{t:12.2f} | {val:12.4f}")
        if val > best_value:
            best_value     = val
            best_threshold = t

    if verbose:
        print(f"\n✅ Mejor umbral: {best_threshold:.2f}  ({metric} = {best_value:.4f})")

    return best_threshold, best_value


def find_best_threshold_per_class(
    y_true: np.ndarray,
    y_pred_probs: np.ndarray,
    thresholds: Optional[List[float]] = None,
    label_names: List[str] = EMOTION_LABELS,
) -> np.ndarray:
    """
    Busca el umbral óptimo por cada clase de forma independiente.
    Maximiza F1 por clase.

    Returns:
        Array de umbrales [n_labels].
    """

    if thresholds is None:
        thresholds = np.arange(0.1, 0.91, 0.05).tolist()

    best_thresholds = []
    print(f"{'Emoción':15} | {'Umbral':>8} | {'F1':>8}")
    print("-" * 38)

    for i, label in enumerate(label_names):
        best_t   = 0.5
        best_f1  = -1.0
        for t in thresholds:
            pred_col = (y_pred_probs[:, i] >= t).astype(int)
            f1 = f1_score(y_true[:, i], pred_col, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_t  = t
        best_thresholds.append(best_t)
        print(f"{label:15} | {best_t:8.2f} | {best_f1:8.4f}")

    return np.array(best_thresholds)


# ──────────────────────────────────────────────
# Agregación de logits de múltiples modelos
# ──────────────────────────────────────────────

def ensemble_probs(list_of_probs: List[np.ndarray]) -> np.ndarray:
    """
    Promedia las probabilidades de varios modelos (ensemble).

    Args:
        list_of_probs: Lista de arrays [n_samples, n_labels].

    Returns:
        Array promediado [n_samples, n_labels].
    """

    return np.mean(np.stack(list_of_probs, axis=0), axis=0)
