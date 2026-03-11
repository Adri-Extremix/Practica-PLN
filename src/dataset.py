"""
dataset.py
==========
Clase Dataset de PyTorch para HopeEXP Task B.
Reutilizable para otras tareas con mínimos cambios.
"""

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from typing import List, Optional, Dict, Any


class HopeEXPDataset(Dataset):
    """
    Dataset PyTorch para clasificación multietiqueta de emociones.

    Args:
        texts: Lista de textos de entrada.
        labels: Lista de vectores binarios [n_samples, n_labels].
                Si es None (test sin etiquetas), devuelve tensores vacíos.
        tokenizer: Tokenizer de HuggingFace.
        max_length: Longitud máxima de tokenización.
    """

    def __init__(self, texts: List[str], tokenizer: AutoTokenizer, labels: Optional[List[List[int]]] = None, max_length: int = 128):
        self.texts     = texts
        self.labels    = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        encoding = self.tokenizer(self.texts[idx], max_length=self.max_length, padding="max_length", truncation=True, return_tensors="pt")

        item = {
            "input_ids":      encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
        }

        # token_type_ids no siempre está disponible (p.ej. RoBERTa no lo usa)
        if "token_type_ids" in encoding:
            item["token_type_ids"] = encoding["token_type_ids"].squeeze(0)

        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.float32)

        return item


# ──────────────────────────────────────────────
# Funciones de construcción de DataLoaders
# ──────────────────────────────────────────────

def build_dataloader(
    texts: List[str],
    tokenizer: AutoTokenizer,
    labels: Optional[List[List[int]]] = None,
    max_length: int = 128,
    batch_size: int = 16,
    shuffle: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    """
    Construye un DataLoader a partir de textos y etiquetas.

    Args:
        texts: Lista de textos.
        tokenizer: Tokenizer de HuggingFace.
        labels: Vectores binarios (None para test sin etiquetas).
        max_length: Longitud máxima de tokenización.
        batch_size: Tamaño de batch.
        shuffle: Si True, mezcla los datos (True para train, False para dev/test).
        num_workers: Workers para carga paralela (0 en Windows/Colab).
    """

    dataset = HopeEXPDataset(
        texts=texts,
        tokenizer=tokenizer,
        labels=labels,
        max_length=max_length,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def build_all_dataloaders(
    train_texts: List[str],
    train_labels: List[List[int]],
    dev_texts: List[str],
    dev_labels: List[List[int]],
    tokenizer: AutoTokenizer,
    test_texts: Optional[List[str]] = None,
    max_length: int = 128,
    batch_size: int = 16,
    num_workers: int = 0,
) -> Dict[str, DataLoader]:
    """
    Construye los tres DataLoaders (train, dev, test) de una vez.
    Devuelve un dict con claves 'train', 'dev', 'test'.
    """

    loaders = {
        "train": build_dataloader(
            train_texts, tokenizer, train_labels,
            max_length, batch_size, shuffle=True, num_workers=num_workers
        ),
        "dev": build_dataloader(
            dev_texts, tokenizer, dev_labels,
            max_length, batch_size, shuffle=False, num_workers=num_workers
        ),
    }
    if test_texts is not None:
        loaders["test"] = build_dataloader(
            test_texts, tokenizer, labels=None,
            max_length=max_length, batch_size=batch_size,
            shuffle=False, num_workers=num_workers,
        )
    return loaders
