from __future__ import annotations

import json
from typing import Any

from shared.schemas import (
    ACTORS,
    EMOTIONS,
    EMPTY_SPAN_LABELS,
    OUTCOME_STANCES,
    PRIMARY_LABELS,
    HopeExpPost,
    HopeExpPrediction,
    SpanAnnotation,
)


PRIMARY_LABEL_ALIASES = {label.lower(): label for label in PRIMARY_LABELS}
EMOTION_ALIASES = {emotion.lower(): emotion for emotion in EMOTIONS}
EMOTION_ALIASES["nuetral/unclear"] = "Neutral/unclear"
STANCE_ALIASES = {stance.lower(): stance for stance in OUTCOME_STANCES}
ACTOR_ALIASES = {actor.lower(): actor for actor in ACTORS}
ACTOR_ALIASES["world"] = "World/System"
ACTOR_ALIASES["system"] = "World/System"


def _extract_first_json_object(text: str) -> str:
    start = text.find("{")
    if start == -1:
        raise ValueError(f"No JSON object found in model output: {text!r}")

    depth = 0
    in_string = False
    escape = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]

    raise ValueError("Unclosed JSON object in model output")


def _normalize_primary_label(value: Any) -> str:
    label = PRIMARY_LABEL_ALIASES.get(str(value).strip().lower())
    return label or "Not Hope"


def _normalize_emotions(value: Any) -> list[str]:
    if not isinstance(value, list):
        return ["Neutral/unclear"]
    normalized: list[str] = []
    for item in value:
        canonical = EMOTION_ALIASES.get(str(item).strip().lower())
        if canonical and canonical not in normalized:
            normalized.append(canonical)
    return normalized or ["Neutral/unclear"]


def _normalize_stance(value: Any) -> str:
    return STANCE_ALIASES.get(str(value).strip().lower(), "Desired")


def _normalize_actor(value: Any) -> str:
    return ACTOR_ALIASES.get(str(value).strip().lower(), "Unclear")


def _align_span(source_text: str, candidate: str) -> str | None:
    if not candidate:
        return None
    if candidate in source_text:
        return candidate

    compact_candidate = " ".join(candidate.split())
    if compact_candidate in source_text:
        return compact_candidate

    lowered_source = source_text.lower()
    lowered_candidate = compact_candidate.lower()
    position = lowered_source.find(lowered_candidate)
    if position == -1:
        return None
    return source_text[position : position + len(lowered_candidate)]


def parse_prediction(raw_text: str, post: HopeExpPost) -> HopeExpPrediction:
    json_blob = _extract_first_json_object(raw_text)
    payload = json.loads(json_blob)

    primary_label = _normalize_primary_label(payload.get("primary_label"))
    trigger_emotions = _normalize_emotions(payload.get("trigger_emotions"))

    if primary_label in EMPTY_SPAN_LABELS:
        return HopeExpPrediction(
            primary_label=primary_label,
            trigger_emotions=trigger_emotions,
            span_annotations=[],
        )

    span_annotations: list[SpanAnnotation] = []
    raw_spans = payload.get("span_annotations")
    if isinstance(raw_spans, list):
        for item in raw_spans[:3]:
            if not isinstance(item, dict):
                continue
            aligned_span = _align_span(post.source_text, str(item.get("span", "")).strip())
            if not aligned_span:
                continue
            span_annotations.append(
                SpanAnnotation(
                    span=aligned_span,
                    outcome_stance=_normalize_stance(item.get("outcome_stance")),
                    actor=_normalize_actor(item.get("actor")),
                )
            )

    return HopeExpPrediction(
        primary_label=primary_label,
        trigger_emotions=trigger_emotions,
        span_annotations=span_annotations,
    )
