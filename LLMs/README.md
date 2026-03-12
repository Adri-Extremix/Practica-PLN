# LLMs

Primera aproximacion para HOPE-EXP con modelos servidos mediante una API compatible con OpenAI.

## Objetivo

- Reutilizar la misma estructura con Ollama local o con cualquier proveedor OpenAI-compatible.
- Devolver un JSON unificado con `primary_label`, `trigger_emotions` y `span_annotations`.
- Compartir lectura de dataset y EDA con la parte de `Fine-Tuning/`.

## Configuracion

Copia `.env.example` a `.env` y ajusta los valores si hace falta.

Valores por defecto pensados para Ollama:

```env
OPENAI_BASE_URL=http://localhost:11434/v1
OPENAI_API_KEY=ollama
OPENAI_MODEL=qwen2.5:7b-instruct
OPENAI_TEMPERATURE=0.2
OPENAI_TIMEOUT=120
```

## Uso

Inferencia:

```bash
python3 LLMs/run_inference.py --input data/dev.jsonl --output outputs/dev_predictions.jsonl --verbose
```

EDA comun:

```bash
python3 shared/run_eda.py --input data/dev.jsonl --output outputs/dev_eda.json
```

## Notas

- `shared/dataset.py` soporta `.jsonl`, `.json`, `.csv` y `.tsv`.
- Si el dataset final cambia de nombres de columnas, solo habra que adaptar `shared/dataset.py`.
- La validacion posterior intenta corregir spans para que sean substrings exactos del texto.
