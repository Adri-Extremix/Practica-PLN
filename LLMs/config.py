from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def load_dotenv_file(path: str | Path = ".env") -> None:
    env_path = Path(path)
    if not env_path.exists():
        return

    for line in env_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        key = key.strip()
        value = value.strip().strip("\"").strip("'")
        os.environ.setdefault(key, value)


@dataclass(slots=True)
class LLMConfig:
    base_url: str
    api_key: str
    model: str
    temperature: float = 0.2
    timeout_seconds: int = 120

    @classmethod
    def from_env(cls, env_path: str | Path = ".env") -> "LLMConfig":
        load_dotenv_file(env_path)
        base_url = os.getenv("OPENAI_BASE_URL") or os.getenv("OLLAMA_BASE_URL") or "http://localhost:11434/v1"
        api_key = os.getenv("OPENAI_API_KEY") or os.getenv("OLLAMA_API_KEY") or "ollama"
        model = os.getenv("OPENAI_MODEL") or os.getenv("OLLAMA_MODEL") or "qwen2.5:7b-instruct"
        temperature = float(os.getenv("OPENAI_TEMPERATURE", "0.2"))
        timeout_seconds = int(os.getenv("OPENAI_TIMEOUT", "120"))
        return cls(
            base_url=base_url.rstrip("/"),
            api_key=api_key,
            model=model,
            temperature=temperature,
            timeout_seconds=timeout_seconds,
        )
