"""
trainer.py
==========
Bucle de entrenamiento y evaluación para HopeEXP Task B.
Incluye early stopping, scheduler con warmup y logging.
"""

import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from typing import Optional, Dict, List, Tuple

from src.metrics import compute_metrics, print_metrics
from src.model import save_model


# ──────────────────────────────────────────────
# Epoch: train
# ──────────────────────────────────────────────

def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: torch.device,
    pos_weight: Optional[torch.Tensor] = None,
    max_grad_norm: float = 1.0,
) -> float:
    """
    Ejecuta una época de entrenamiento.

    Returns:
        Loss media de la época.
    """

    model.train()
    total_loss = 0.0

    for batch in dataloader:
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels         = batch["labels"].to(device)
        token_type_ids = batch.get("token_type_ids")
        if token_type_ids is not None:
            token_type_ids = token_type_ids.to(device)

        optimizer.zero_grad()

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            labels=labels,
            pos_weight=pos_weight.to(device) if pos_weight is not None else None,
        )

        loss = outputs["loss"]
        loss.backward()

        # Gradient clipping
        nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


# ──────────────────────────────────────────────
# Epoch: evaluate
# ──────────────────────────────────────────────

def evaluate_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    threshold: float = 0.5,
    pos_weight: Optional[torch.Tensor] = None,
) -> Tuple[float, Dict[str, float], np.ndarray, np.ndarray]:
    """
    Evalúa el modelo en un split.

    Returns:
        Tupla (loss_media, metrics_dict, all_probs, all_labels)
    """

    model.eval()
    total_loss = 0.0
    all_probs  = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch.get("token_type_ids")
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(device)

            labels = batch.get("labels")
            labels_device = labels.to(device) if labels is not None else None

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                labels=labels_device,
                pos_weight=pos_weight.to(device) if pos_weight is not None else None,
            )

            if "loss" in outputs:
                total_loss += outputs["loss"].item()

            probs = torch.sigmoid(outputs["logits"]).cpu().numpy()
            all_probs.append(probs)

            if labels is not None:
                all_labels.append(labels.numpy())

    all_probs  = np.vstack(all_probs)
    all_labels = np.vstack(all_labels) if all_labels else None

    avg_loss = total_loss / len(dataloader) if total_loss > 0 else 0.0
    metrics  = {}

    if all_labels is not None:
        metrics = compute_metrics(all_labels, all_probs, threshold=threshold)

    return avg_loss, metrics, all_probs, all_labels


# ──────────────────────────────────────────────
# Bucle de entrenamiento completo
# ──────────────────────────────────────────────

def train(
    model: nn.Module,
    train_loader: DataLoader,
    dev_loader: DataLoader,
    device: torch.device,
    num_epochs: int = 5,
    learning_rate: float = 2e-5,
    warmup_ratio: float = 0.1,
    weight_decay: float = 0.01,
    threshold: float = 0.5,
    pos_weight: Optional[torch.Tensor] = None,
    save_dir: str = "outputs",
    model_name: str = "best_model.pt",
    early_stopping_patience: int = 3,
    monitor_metric: str = "f1_macro",
    max_grad_norm: float = 1.0,
    verbose: bool = True,
) -> Dict[str, List]:
    """
    Entrena el modelo con early stopping y guardado del mejor checkpoint.

    Args:
        model: Modelo a entrenar.
        train_loader: DataLoader de entrenamiento.
        dev_loader: DataLoader de validación.
        device: Dispositivo (cuda/cpu).
        num_epochs: Número máximo de épocas.
        learning_rate: Learning rate inicial.
        warmup_ratio: Fracción de steps para warmup lineal.
        weight_decay: Regularización L2.
        threshold: Umbral de decisión para evaluación.
        pos_weight: Pesos positivos para BCEWithLogitsLoss.
        save_dir: Directorio donde guardar el mejor modelo.
        model_name: Nombre del fichero del mejor modelo.
        early_stopping_patience: Épocas sin mejora antes de parar.
        monitor_metric: Métrica a monitorizar para early stopping.
        max_grad_norm: Norma máxima de gradientes (clipping).
        verbose: Si True, imprime métricas por época.

    Returns:
        Historial de entrenamiento con listas de losses y métricas.
    """

    os.makedirs(save_dir, exist_ok=True)
    model = model.to(device)

    # Optimizer: no aplicar weight decay a bias y LayerNorm
    no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
    optimizer_params = [
        {
            "params": [
                p for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": weight_decay,
        },
        {
            "params": [
                p for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_params, lr=learning_rate)

    # Scheduler con warmup lineal
    total_steps  = len(train_loader) * num_epochs
    warmup_steps = int(total_steps * warmup_ratio)
    scheduler    = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    # Historial
    history = {
        "train_loss": [],
        "dev_loss":   [],
        "dev_metrics": [],
    }

    best_metric        = -1.0
    patience_counter   = 0
    best_model_path    = os.path.join(save_dir, model_name)

    print(f"Iniciando entrenamiento: {num_epochs} épocas, lr={learning_rate}")
    print(f"- Total steps: {total_steps} | Warmup steps: {warmup_steps}")
    print(f"- Early stopping: patience={early_stopping_patience}, monitor='{monitor_metric}'")
    print()

    for epoch in range(1, num_epochs + 1):
        t0 = time.time()

        # ── Train ──
        train_loss = train_epoch(
            model, train_loader, optimizer, scheduler, device,
            pos_weight=pos_weight, max_grad_norm=max_grad_norm,
        )

        # ── Evaluate ──
        dev_loss, dev_metrics, _, _ = evaluate_epoch(
            model, dev_loader, device, threshold=threshold, pos_weight=pos_weight,
        )

        elapsed = time.time() - t0
        current_metric = dev_metrics.get(monitor_metric, 0.0)

        history["train_loss"].append(train_loss)
        history["dev_loss"].append(dev_loss)
        history["dev_metrics"].append(dev_metrics)

        if verbose:
            print(f"Época {epoch}/{num_epochs}  [{elapsed:.1f}s]")
            print(f"- Train Loss: {train_loss:.4f}  |  Dev Loss: {dev_loss:.4f}")
            print(f"- Dev {monitor_metric}: {current_metric:.4f}")

        # ── Early stopping y guardado ──
        if current_metric > best_metric:
            best_metric = current_metric
            patience_counter = 0
            save_model(model, best_model_path)
            if verbose:
                print(f"Nuevo mejor modelo guardado ({monitor_metric}={best_metric:.4f})")
        else:
            patience_counter += 1
            if verbose:
                print(f"Sin mejora ({patience_counter}/{early_stopping_patience})")
            if patience_counter >= early_stopping_patience:
                print(f"\n- Early stopping activado en época {epoch}")
                break

        print()

    print(f"- Entrenamiento finalizado. Mejor {monitor_metric}: {best_metric:.4f}")
    return history


# ──────────────────────────────────────────────
# Predicción sobre test
# ──────────────────────────────────────────────

def predict(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> np.ndarray:
    """
    Genera probabilidades para un dataloader sin etiquetas (test).

    Returns:
        Array [n_samples, n_labels] con probabilidades.
    """

    model.eval()
    model.to(device)
    all_probs = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch.get("token_type_ids")
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )
            probs = torch.sigmoid(outputs["logits"]).cpu().numpy()
            all_probs.append(probs)

    return np.vstack(all_probs)


# ──────────────────────────────────────────────
# Comparación de modelos candidatos
# ──────────────────────────────────────────────

def compare_models(
    model_names: List[str],
    train_loader: DataLoader,
    dev_loader: DataLoader,
    device: torch.device,
    num_labels: int = 7,
    num_epochs: int = 3,
    learning_rate: float = 2e-5,
    warmup_ratio: float = 0.1,
    weight_decay: float = 0.01,
    threshold: float = 0.5,
    pos_weight: Optional[torch.Tensor] = None,
    save_dir: str = "outputs/model_comparison",
    monitor_metric: str = "f1_macro",
    early_stopping_patience: int = 2,
) -> "pd.DataFrame":
    """
    Entrena cada arquitectura de la lista y devuelve un ranking comparativo.

    Se recomienda usar esta función en lugar de HPO sobre un único modelo:
    explorar distintas arquitecturas suele ofrecer mayor ganancia con el
    mismo presupuesto de cómputo.

    Args:
        model_names:            Lista de nombres HuggingFace a comparar.
                                Usa ``[m["name"] for m in CANDIDATE_MODELS]``
                                de model.py para obtener la lista completa.
        train_loader:           DataLoader de entrenamiento (ya construido).
        dev_loader:             DataLoader de validación.
        device:                 Dispositivo (cuda/cpu).
        num_labels:             Número de etiquetas de emoción.
        num_epochs:             Épocas máximas por modelo.
        monitor_metric:         Métrica para el ranking final ('f1_macro', …).
        early_stopping_patience: Paciencia por modelo.
        ...resto:               Igual que train().

    Returns:
        DataFrame con columnas ['model', monitor_metric, 'best_epoch',
        'train_loss', 'dev_loss', 'status'], ordenado de mejor a peor.

    Example::
        from src.model import CANDIDATE_MODELS
        names = [m["name"] for m in CANDIDATE_MODELS if m["multilingual"]]
        results = compare_models(names, train_loader, dev_loader, device)
    """
    import pandas as pd  # ya importado en el módulo; se deja explícito por claridad
    from src.model import build_model, build_tokenizer
    from src.dataset import build_dataloader

    # Extraer textos, etiquetas y parámetros del loader original
    # para poder retokenizar con cada arquitectura
    train_texts  = train_loader.dataset.texts
    train_labels = train_loader.dataset.labels
    dev_texts    = dev_loader.dataset.texts
    dev_labels   = dev_loader.dataset.labels
    batch_size   = train_loader.batch_size
    max_length   = train_loader.dataset.max_length

    os.makedirs(save_dir, exist_ok=True)
    rows = []

    for model_name in model_names:
        print(f"\n{'='*60}")
        print(f"  Evaluando: {model_name}")
        print(f"{'='*60}")

        try:
            # Tokenizer y dataloaders propios de esta arquitectura
            # (cada modelo tiene un vocabulario distinto: reutilizar
            # los IDs de otro tokenizer causa índices fuera de rango en GPU)
            tokenizer = build_tokenizer(model_name)
            arch_train_loader = build_dataloader(
                train_texts, tokenizer, train_labels,
                max_length=max_length, batch_size=batch_size,
                shuffle=True,
            )
            arch_dev_loader = build_dataloader(
                dev_texts, tokenizer, dev_labels,
                max_length=max_length, batch_size=batch_size,
                shuffle=False,
            )

            model = build_model(model_name=model_name, num_labels=num_labels)
            model = model.float()
            safe_name = model_name.replace("/", "_")

            history = train(
                model=model,
                train_loader=arch_train_loader,
                dev_loader=arch_dev_loader,
                device=device,
                num_epochs=num_epochs,
                learning_rate=learning_rate,
                warmup_ratio=warmup_ratio,
                weight_decay=weight_decay,
                threshold=threshold,
                pos_weight=pos_weight,
                save_dir=save_dir,
                model_name=f"{safe_name}_best.pt",
                early_stopping_patience=early_stopping_patience,
                monitor_metric=monitor_metric,
                verbose=True,
            )

            metric_values   = [m.get(monitor_metric, 0.0) for m in history["dev_metrics"]]
            best_epoch      = int(np.argmax(metric_values)) + 1
            best_metric_val = max(metric_values)

            rows.append({
                "model":        model_name,
                monitor_metric: round(best_metric_val, 4),
                "best_epoch":   best_epoch,
                "train_loss":   round(history["train_loss"][best_epoch - 1], 4),
                "dev_loss":     round(history["dev_loss"][best_epoch - 1], 4),
                "status":       "ok",
            })

        except Exception as exc:
            print(f"  ✗ Error con {model_name}: {exc}")
            rows.append({
                "model":        model_name,
                monitor_metric: 0.0,
                "best_epoch":   0,
                "train_loss":   None,
                "dev_loss":     None,
                "status":       f"error: {exc}",
            })

    results_df = (
        pd.DataFrame(rows)
        .sort_values(monitor_metric, ascending=False)
        .reset_index(drop=True)
    )

    print(f"\n{'='*60}")
    print("  Ranking final de modelos")
    print(f"{'='*60}")
    print(results_df.to_string(index=False))

    csv_path = os.path.join(save_dir, "model_comparison.csv")
    results_df.to_csv(csv_path, index=False)
    print(f"\nResultados guardados en {csv_path}")

    return results_df

# ──────────────────────────────────────────────
# Construcción del optimizer/scheduler de forma standalone
# ──────────────────────────────────────────────

def build_optimizer_and_scheduler(
    model: nn.Module,
    num_training_steps: int,
    learning_rate: float = 2e-5,
    warmup_ratio: float = 0.1,
    weight_decay: float = 0.01,
):
    """
    Devuelve (optimizer, scheduler) listos para usar.
    Útil cuando se quiere más control fuera del bucle train().
    """

    no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
    params = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(params, lr=learning_rate)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(num_training_steps * warmup_ratio),
        num_training_steps=num_training_steps,
    )
    return optimizer, scheduler
