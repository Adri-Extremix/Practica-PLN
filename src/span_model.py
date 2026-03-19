from __future__ import annotations

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

from src.span_utils import ID2LABEL, LABEL2ID


def build_span_model(
    model_name: str = "microsoft/mdeberta-v3-base",
    num_labels: int = 3,
) -> AutoModelForTokenClassification:
    """
    Carga y configura un modelo de clasificación de tokens (NER) desde Hugging Face.

    Args:
        model_name: Nombre o ruta del modelo pre-entrenado.
        num_labels: Número de etiquetas a predecir (típicamente 3: O, B-SPAN, I-SPAN).

    Returns:
        Modelo configurado con el mapeo de IDs a etiquetas del proyecto.
    """
    model = AutoModelForTokenClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        dtype=torch.float32,
    )
    return model.float()


def build_span_tokenizer(model_name: str = "microsoft/mdeberta-v3-base"):
    """Carga el tokenizador rápido asociado al modelo."""
    return AutoTokenizer.from_pretrained(model_name, use_fast=True)


def save_span_model(model: AutoModelForTokenClassification, tokenizer, output_dir: str) -> None:
    """Guarda tanto el modelo como el tokenizador en el directorio especificado."""
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


def load_span_model(model_dir: str, device: torch.device | None = None) -> AutoModelForTokenClassification:
    """
    Carga un modelo guardado localmente para inferencia o continuación de entrenamiento.
    
    Asegura que el modelo esté en el dispositivo correcto y en modo de precisión simple.
    """
    model = AutoModelForTokenClassification.from_pretrained(model_dir, dtype=torch.float32).float()
    if device is not None:
        model = model.to(device)
    return model
