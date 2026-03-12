from __future__ import annotations

import re
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd
import spacy
import matplotlib.pyplot as plt
import seaborn as sns

from src.shared.schemas import HopeExpPost

# Diccionario para cargar modelos de spaCy según el idioma
SPACY_MODELS = {
    'en': 'en_core_web_sm',
    'es': 'es_core_news_sm'
}
_loaded_models = {}

def get_spacy_model(lang: str):
    model_name = SPACY_MODELS.get(lang, SPACY_MODELS['en'])
    if model_name not in _loaded_models:
        try:
            _loaded_models[model_name] = spacy.load(model_name)
        except OSError:
            # Si no está instalado, intentamos con el de inglés por defecto
            _loaded_models[model_name] = spacy.load(SPACY_MODELS['en'])
    return _loaded_models[model_name]

def get_ngrams(text: str, n: int) -> list[str]:
    text = re.sub(r'[^\w\s]', '', str(text).lower())
    tokens = text.split()
    if len(tokens) < n:
        return []
    return [" ".join(tokens[i:i+n]) for i in range(len(tokens)-n+1)]

def generate_eda_plots(df: pd.DataFrame, output_dir: str | Path) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    sns.set_theme(style="whitegrid")
    
    # 1. Distribución de métricas de longitud
    metrics = ['num_words', 'num_chars', 'num_sentences', 'max_word_len', 'avg_sentence_len']
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    for i, metric in enumerate(metrics):
        sns.histplot(x=df[metric], kde=True, ax=axes[i], color='skyblue')
        axes[i].set_title(f'Distribución de {metric}')
    
    # Eliminar el último eje sobrante si hay
    if len(metrics) < len(axes):
        fig.delaxes(axes[-1])
        
    plt.tight_layout()
    plt.savefig(output_path / "text_length_distributions.png")
    plt.close()

    # 2. Análisis por Etiquetas e Idiomas
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    
    # Distribución de Idiomas
    sns.countplot(data=df, x='lang', ax=axes[0], hue='lang', palette='viridis', legend=False)
    axes[0].set_title('Posts por Idioma')
    
    # Relación entre Etiqueta y Longitud (Boxplot)
    df['label'] = df['metadata'].apply(lambda x: x.get('primary_label') if isinstance(x, dict) else 'Unknown')
    sns.boxplot(data=df, y='label', x='num_words', ax=axes[1], hue='label', palette='magma', legend=False)
    axes[1].set_title('Longitud de Palabras por Etiqueta')
    
    plt.tight_layout()
    plt.savefig(output_path / "categories_and_labels.png")
    plt.close()

    # 3. N-Grams por Idioma
    for lang in df['lang'].unique():
        lang_text = " ".join(df[df['lang'] == lang]['text'])
        if not lang_text.strip(): continue
        
        fig, axes = plt.subplots(1, 2, figsize=(18, 8))
        for i, n in enumerate([2, 3]):
            ngrams = Counter(get_ngrams(lang_text, n)).most_common(15)
            if not ngrams: continue
            
            words, counts = zip(*ngrams)
            sns.barplot(x=list(counts), y=list(words), ax=axes[i], hue=list(words), palette='rocket', legend=False)
            axes[i].set_title(f'Top 15 {"Bigramas" if n==2 else "Trigramas"} ({lang})')
        
        plt.tight_layout()
        plt.savefig(output_path / f"ngrams_{lang}.png")
        plt.close()

def summarize_posts(posts: list[HopeExpPost], plot_dir: str | Path | None = None) -> dict[str, Any]: 
    if not posts:
        return {"error": "No posts provided"}

    df = pd.DataFrame([p.to_dict() for p in posts])