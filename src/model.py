"""
model.py
========
Definición del modelo para clasificación multietiqueta de emociones.
Basado en microsoft/mdeberta-v3-base con cabeza clasificadora propia.
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig, AutoTokenizer
from typing import Optional


# ──────────────────────────────────────────────
# Modelo principal
# ──────────────────────────────────────────────

class MultiLabelEmotionClassifier(nn.Module):
    """
    Clasificador multietiqueta basado en DeBERTa (o cualquier encoder).

    Arquitectura:
        [CLS] embedding → Dropout → Linear → logits (sin sigmoid)

    La sigmoid se aplica fuera durante la inferencia.
    Durante el entrenamiento se usa BCEWithLogitsLoss (numéricamente estable).

    Args:
        model_name: Nombre del modelo HuggingFace (p.ej. 'microsoft/mdeberta-v3-base').
        num_labels: Número de etiquetas de emoción.
        dropout_prob: Probabilidad de dropout antes de la cabeza.
        hidden_size: Si None, se infiere del config del modelo base.
    """

    def __init__(
        self,
        model_name: str = "microsoft/mdeberta-v3-base",
        num_labels: int = 7,
        dropout_prob: float = 0.1,
        hidden_size: Optional[int] = None,
    ):
        super().__init__()
        self.num_labels = num_labels

        # Encoder base
        self.config  = AutoConfig.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name, config=self.config)

        # Tamaño del embedding
        enc_hidden = hidden_size or self.config.hidden_size

        # Cabeza clasificadora
        self.dropout    = nn.Dropout(dropout_prob)
        self.classifier = nn.Linear(enc_hidden, num_labels)

        # Inicialización de la cabeza
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        pos_weight: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            input_ids:      [batch, seq_len]
            attention_mask: [batch, seq_len]
            token_type_ids: [batch, seq_len] (opcional)
            labels:         [batch, num_labels] (float) — si se pasa, calcula la loss
            pos_weight:     [num_labels] — pesos positivos para BCEWithLogitsLoss

        Returns:
            dict con 'logits' y opcionalmente 'loss'
        """

        kwargs = dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        if token_type_ids is not None:
            kwargs["token_type_ids"] = token_type_ids

        outputs = self.encoder(**kwargs)

        # Usar el token [CLS] (posición 0 de last_hidden_state)
        cls_output = outputs.last_hidden_state[:, 0, :]
        cls_output = self.dropout(cls_output)
        logits     = self.classifier(cls_output)       # [batch, num_labels]

        result = {"logits": logits}

        if labels is not None:
            loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            result["loss"] = loss_fn(logits, labels)

        return result

    def get_probabilities(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Devuelve probabilidades [0,1] aplicando sigmoid a los logits."""
        with torch.no_grad():
            out = self.forward(input_ids, attention_mask, token_type_ids)
        return torch.sigmoid(out["logits"])


# ──────────────────────────────────────────────
# Helpers de construcción y guardado
# ──────────────────────────────────────────────

def build_model(
    model_name: str = "microsoft/mdeberta-v3-base",
    num_labels: int = 7,
    dropout_prob: float = 0.1,
) -> MultiLabelEmotionClassifier:
    """Instancia y devuelve el modelo."""
    return MultiLabelEmotionClassifier(
        model_name=model_name,
        num_labels=num_labels,
        dropout_prob=dropout_prob,
    )


def build_tokenizer(model_name: str = "microsoft/mdeberta-v3-base") -> AutoTokenizer:
    """Devuelve el tokenizer asociado al modelo."""
    return AutoTokenizer.from_pretrained(model_name)


def save_model(model: nn.Module, path: str) -> None:
    """Guarda los pesos del modelo."""
    torch.save(model.state_dict(), path)
    print(f"Modelo guardado en {path}")


def load_model(
    path: str,
    model_name: str = "microsoft/mdeberta-v3-base",
    num_labels: int = 7,
    dropout_prob: float = 0.1,
    device: Optional[torch.device] = None,
) -> MultiLabelEmotionClassifier:
    """Carga un modelo desde un fichero de pesos."""

    model = build_model(model_name, num_labels, dropout_prob)
    state = torch.load(path, map_location=device or torch.device("cpu"))
    model.load_state_dict(state)
    if device:
        model = model.to(device)
    print(f"Modelo cargado desde {path}")
    return model


def count_parameters(model: nn.Module) -> int:
    """Cuenta el número de parámetros entrenables."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
