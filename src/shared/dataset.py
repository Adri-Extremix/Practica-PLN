from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, cast

import pandas as pd
from src.shared.schemas import HopeExpPost


def load_raw_records(path: str | Path) -> list[dict[str, Any]]:
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset not found: {file_path}")
    
    # Usamos pandas para una carga más eficiente de JSONL
    df = pd.read_json(file_path, lines=True)
    return cast(list[dict[str, Any]], df.to_dict(orient="records"))


def normalize_post(record: dict[str, Any]) -> HopeExpPost:
    """Mapeo directo basándose en el formato conocido del dataset."""
    return HopeExpPost(
        row_id=record["row_id"],
        lang=record["lang"],
        title=record["title"],
        selftext=record["selftext"],
        metadata={
            "primary_label": record.get("primary_label"),
            "span_annotations": record.get("span_annotations", []),
            "trigger_emotions": record.get("trigger_emotions", []),
        },
    )


def load_posts(path: str | Path) -> list[HopeExpPost]:
    return [normalize_post(record) for record in load_raw_records(path)]


def save_jsonl(records: Iterable[dict[str, Any]], path: str | Path) -> None:
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

def convert_posts_to_df(posts: list[HopeExpPost]) -> pd.DataFrame:
    data = []
    for post in posts:
        data.append({
            "row_id": post.row_id,
            "lang": post.lang,
            "title": post.title,
            "selftext": post.selftext,
            "source_text": post.source_text,
            "primary_label": post.metadata.get("primary_label"),
            "trigger_emotions": post.metadata.get("trigger_emotions", []),
            "span_annotations": post.metadata.get("span_annotations", []),
        })
    return pd.DataFrame(data)