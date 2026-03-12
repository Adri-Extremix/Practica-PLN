from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


PRIMARY_LABELS = (
    "General Hope",
    "Realistic Hope",
    "Unrealistic Hope",
    "Sarcastic Hope",
    "Hopelessness",
    "Not Hope",
)

EMOTIONS = (
    "sadness",
    "joy",
    "love",
    "anger",
    "fear",
    "surprise",
    "Neutral/unclear",
)

OUTCOME_STANCES = (
    "Desired",
    "Avoided",
)

ACTORS = (
    "Self",
    "Other",
    "World/System",
    "Unclear",
)

HOPEFUL_LABELS = (
    "General Hope",
    "Realistic Hope",
    "Unrealistic Hope",
    "Sarcastic Hope",
)

EMPTY_SPAN_LABELS = (
    "Hopelessness",
    "Not Hope",
)


@dataclass(slots=True)
class SpanAnnotation:
    span: str
    outcome_stance: str
    actor: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(slots=True)
class HopeExpPrediction:
    primary_label: str
    trigger_emotions: list[str] = field(default_factory=list)
    span_annotations: list[SpanAnnotation] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["span_annotations"] = [item.to_dict() for item in self.span_annotations]
        return payload


@dataclass(slots=True)
class HopeExpPost:
    row_id: str | int
    lang: str
    title: str
    selftext: str
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def source_text(self) -> str:
        if self.title and self.selftext:
            return f"{self.title}\n{self.selftext}"
        return self.title or self.selftext

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["source_text"] = self.source_text
        return payload