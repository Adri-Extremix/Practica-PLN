"""
data_utils.py
=============
Utilidades de carga, limpieza y preprocesamiento de datos para HopeEXP.
Reutilizable en otras tareas del sistema (A, C, D).
"""

import re
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple, Optional

# ──────────────────────────────────────────────
# Constantes globales
# ──────────────────────────────────────────────

EMOTION_LABELS = ["sadness", "joy", "love", "anger", "fear", "surprise", "Nuetral/unclear"]

EMOTION2IDX = {label: idx for idx, label in enumerate(EMOTION_LABELS)}
IDX2EMOTION = {idx: label for label, idx in EMOTION2IDX.items()}


def normalize_emotion(emotion: str) -> str:
    """Devuelve la etiqueta en minúsculas sin espacios para comparación interna.
    No se modifica el valor original del dataset (p.ej. 'Nuetral/unclear' se conserva)."""
    return emotion.lower().strip()

PRIMARY_LABELS = [
	"General Hope",
    "Realistic Hope",
    "Unrealistic Hope",
    "Sarcastic Hope",
    "Hopelessness",
    "Not Hope",
]

# ──────────────────────────────────────────────
# Carga de datos
# ──────────────────────────────────────────────

def load_json(path: str) -> List[Dict]:
    """
    Carga un fichero HopeEXP en formato JSONL (una línea = un JSON)
    o JSON estándar (array o dict con clave 'data').
    """
    with open(path, "r", encoding="utf-8") as f:
        content = f.read().strip()

    # Intentar primero como JSON estándar
    try:
        data = json.loads(content)
        if isinstance(data, list):
            return data
        if isinstance(data, dict) and "data" in data:
            return data["data"]
    except json.JSONDecodeError:
        pass

    # Si falla, tratar como JSONL (una línea por registro)
    records = []
    for line in content.splitlines():
        line = line.strip()
        if line:
            records.append(json.loads(line))
    return records


def load_split(path: str, combine_title: bool = True) -> pd.DataFrame:
    """
    Carga un split (train/dev/test) y lo convierte a DataFrame.

    Formato real de HopeEXP:
        row_id, lang, title, selftext, primary_label,
        span_annotations, trigger_emotions

    Args:
        path: Ruta al fichero JSONL.
        combine_title: Si True, crea la columna 'text' como
                       "título. cuerpo" para dar más contexto al modelo.
                       Si False, usa solo 'selftext'.
    """
    records = load_json(path)
    df = pd.DataFrame(records)
    df.columns = [c.lower().strip() for c in df.columns]

    # Crear columna de texto unificada
    if combine_title and "title" in df.columns and "selftext" in df.columns:
        has_primary = "primary_label" in df.columns
        def _build_text(r):
            selftext = str(r["selftext"]).strip() if pd.notna(r["selftext"]) else ""
            # Ignorar selftext si está vacío o es igual a la primary_label (artefacto del dataset)
            ignore = not selftext or (has_primary and selftext == str(r["primary_label"]))
            return str(r["title"]) if ignore else f"{r['title']}. {selftext}"
        df["text"] = df.apply(_build_text, axis=1)
    elif "selftext" in df.columns:
        df["text"] = df["selftext"]

    # Mantener el nombre original 'trigger_emotions' del dataset

    # Renombrar row_id → id para compatibilidad interna
    if "row_id" in df.columns and "id" not in df.columns:
        df["id"] = df["row_id"]

    return df


def load_all_splits(
    train_path: str,
    dev_path: str,
    test_path: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame]]:
    """Carga train, dev y opcionalmente test."""
    train_df = load_split(train_path)
    dev_df   = load_split(dev_path)
    test_df  = load_split(test_path) if test_path else None
    return train_df, dev_df, test_df


# ──────────────────────────────────────────────
# Limpieza de texto
# ──────────────────────────────────────────────

def clean_text(
    text: str,
    remove_urls: bool = True,
    remove_mentions: bool = True,
    remove_hashtag_symbol: bool = True,
    lowercase: bool = False,
) -> str:
    """
    Limpieza básica de texto de redes sociales.

    Args:
        text: Texto original.
        remove_urls: Si True, elimina URLs.
        remove_mentions: Si True, sustituye @usuario por [USER].
        remove_hashtag_symbol: Si True, elimina el símbolo # pero conserva la palabra.
        lowercase: Si True, convierte a minúsculas.
    """
    if not isinstance(text, str):
        return ""

    # URLs
    if remove_urls:
        text = re.sub(r"http\S+|www\.\S+", "[URL]", text)

    # Menciones
    if remove_mentions:
        text = re.sub(r"@\w+", "[USER]", text)

    # Hashtags: quitar solo el símbolo
    if remove_hashtag_symbol:
        text = re.sub(r"#(\w+)", r"\1", text)

    # Espacios múltiples
    text = re.sub(r"\s+", " ", text).strip()

    if lowercase:
        text = text.lower()

    return text


def apply_cleaning(df: pd.DataFrame, text_col: str = "text", **kwargs) -> pd.DataFrame:
    """Aplica clean_text a una columna del DataFrame."""
    df = df.copy()
    df[text_col] = df[text_col].apply(lambda t: clean_text(t, **kwargs))
    return df

# ──────────────────────────────────────────────
# Codificación de esperanza (Task A)
# ──────────────────────────────────────────────

def encode_hope(
    label: str,
    label_set: List[str] = PRIMARY_LABELS,
) -> int:
    """
    Convierte una esperanza a vector binario.
    Aplica normalización automática.

    Ejemplo:
        encode_hope("General Hope")  →  [1, 0, 0, 0, 0, 0]
    """
    vector = [0] * len(label_set)
    if label in label_set:
        vector[label_set.index(label)] = 1
    return vector


def decode_hope(
    binary_vector: List[int],
    label_set: List[str] = PRIMARY_LABELS,
) -> str:
    """
    Convierte un vector binario (o de probabilidades) a una etiqueta
    Si se pasan probabilidades, toma el máximo.
    """
    return label_set[max(enumerate(binary_vector), key=lambda x: x[1])[0]]


def add_hope(df: pd.DataFrame, hope_col: str = "primary_labels") -> pd.DataFrame:
    """
    Añade al DataFrame una columna 'hope' con el entero.
    Asume que la columna `hope_col` contiene un string.
    """
    df = df.copy()
    df["hope"] = df[hope_col].apply(encode_hope)
    return df


# ──────────────────────────────────────────────
# Codificación de etiquetas (Task B)
# ──────────────────────────────────────────────

def encode_emotions(
    emotion_list: List[str],
    label_set: List[str] = EMOTION_LABELS,
) -> List[int]:
    """
    Convierte una lista de emociones a vector binario multietiqueta.
    Aplica normalización automática (p.ej. "Nuetral/unclear" → "neutral/unclear").

    Ejemplo:
        encode_emotions(["sadness", "fear"])  →  [1, 0, 0, 0, 1, 0, 0]
    """
    label_set_lower = [l.lower().strip() for l in label_set]
    vector = [0] * len(label_set_lower)
    for emotion in emotion_list:
        e = normalize_emotion(emotion)   # normalizar antes de buscar
        if e in label_set_lower:
            vector[label_set_lower.index(e)] = 1
    return np.array(vector, dtype=np.int8)


def decode_emotions(
    binary_vector: List[int],
    label_set: List[str] = EMOTION_LABELS,
    threshold: float = 0.5,
) -> List[str]:
    """
    Convierte un vector binario (o de probabilidades) a lista de etiquetas.
    Si se pasan probabilidades, aplica el threshold.
    """
    return [
        label_set[i]
        for i, val in enumerate(binary_vector)
        if val >= threshold
    ]


def add_emotion_vectors(df: pd.DataFrame, emotions_col: str = "trigger_emotions") -> pd.DataFrame:
    """
    Añade al DataFrame una columna 'emotion_vector' con el vector binario multietiqueta.
    Asume que la columna `emotions_col` contiene listas de strings.
    """
    df = df.copy()
    df["emotion_vector"] = df[emotions_col].apply(
        lambda x: encode_emotions(x if isinstance(x, list) else [])
    )
    return df


# ──────────────────────────────────────────────
# Estadísticas del dataset
# ──────────────────────────────────────────────

def hope_distribution(df: pd.DataFrame, hope_col: str = "primary_label") -> pd.Series:
    """Calcula la frecuencia de cada esperanza en el dataset."""
    counts = {label: 0 for label in PRIMARY_LABELS}
    for hope in df[hope_col]:
        if hope in counts:
            counts[hope] += 1
    return pd.Series(counts).sort_values(ascending=False)

def emotion_distribution(df: pd.DataFrame, emotions_col: str = "trigger_emotions") -> pd.Series:
    """Calcula la frecuencia de cada emoción en el dataset."""
    counts = {label: 0 for label in EMOTION_LABELS}
    for emotions in df[emotions_col]:
        if isinstance(emotions, list):
            for e in emotions:
                key = normalize_emotion(e)
                if key in counts:
                    counts[key] += 1
    return pd.Series(counts).sort_values(ascending=False)


def cooccurrence_matrix(df: pd.DataFrame, emotions_col: str = "trigger_emotions") -> pd.DataFrame:
    """Calcula la matriz de co-ocurrencia de emociones."""
    n = len(EMOTION_LABELS)
    matrix = np.zeros((n, n), dtype=int)
    for emotions in df[emotions_col]:
        if not isinstance(emotions, list):
            continue
        indices = [
            EMOTION2IDX[e.lower().strip()]
            for e in emotions
            if e.lower().strip() in EMOTION2IDX
        ]
        for i in indices:
            for j in indices:
                matrix[i][j] += 1
    return pd.DataFrame(matrix, index=EMOTION_LABELS, columns=EMOTION_LABELS)


def compute_class_weights(df: pd.DataFrame, emotions_col: str = "trigger_emotions", labels: list[str] = EMOTION_LABELS) -> np.ndarray:
    """
    Calcula pesos por clase para manejar el desbalanceo.
    Útil para pasar a BCEWithLogitsLoss como pos_weight.
    Fórmula: (N - n_pos) / n_pos  por clase.
    """
    n = len(df)
    weights = []
    for label in labels:
        n_pos = sum(
            1 for emotions in df[emotions_col]
            if isinstance(emotions, list) and label in emotions
        )
        n_pos = max(n_pos, 1)  # evitar división por cero
        weights.append((n - n_pos) / n_pos)
    return np.array(weights, dtype=np.float32)


# ──────────────────────────────────────────────
# Formateo de predicciones para CodaBench
# ──────────────────────────────────────────────

def format_predictions_for_submission_hope(
    ids: List[str],
    predictions: List[List[int]],
    label_set: List[str] = PRIMARY_LABELS,
) -> pd.DataFrame:
    """
    Genera un DataFrame con el formato de submission para CodaBench.

    Args:
        ids: Lista de identificadores de ejemplo.
        predictions: Lista de vectores binarios o probabilidades.
        label_set: Etiquetas de emoción.
    """
    rows = []
    for id_, pred in zip(ids, predictions):
        hope = decode_hope(pred, label_set)
        rows.append({"id": id_, "primary_label": hope})
    return pd.DataFrame(rows)

def format_predictions_for_submission(
    ids: List[str],
    predictions: List[List[int]],
    label_set: List[str] = EMOTION_LABELS,
    threshold: float = 0.5,
) -> pd.DataFrame:
    """
    Genera un DataFrame con el formato de submission para CodaBench.

    Args:
        ids: Lista de identificadores de ejemplo.
        predictions: Lista de vectores binarios o probabilidades.
        label_set: Etiquetas de emoción.
        threshold: Umbral para binarización si se pasan probabilidades.
    """
    rows = []
    for id_, pred in zip(ids, predictions):
        emotions = decode_emotions(pred, label_set, threshold)
        rows.append({"id": id_, "trigger_emotions": emotions})
    return pd.DataFrame(rows)


def save_submission(df: pd.DataFrame, output_path: str) -> None:
    """Guarda el DataFrame de submission en JSON."""
    records = df.to_dict(orient="records")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    print(f"Submission guardada en {output_path}")
