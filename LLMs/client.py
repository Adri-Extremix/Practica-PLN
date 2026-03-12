from __future__ import annotations

import json
from urllib import error, request

from LLMs.config import LLMConfig


class OpenAICompatibleClient:
    def __init__(self, config: LLMConfig) -> None:
        self.config = config

    def chat_completion(self, system_prompt: str, user_prompt: str) -> str:
        url = f"{self.config.base_url}/chat/completions"
        payload = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": self.config.temperature,
            "stream": False,
        }

        data = json.dumps(payload).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.config.api_key}",
        }
        http_request = request.Request(url, data=data, headers=headers, method="POST")

        try:
            with request.urlopen(http_request, timeout=self.config.timeout_seconds) as response:
                body = response.read().decode("utf-8")
        except error.HTTPError as exc:
            error_body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"LLM request failed with HTTP {exc.code}: {error_body}") from exc
        except error.URLError as exc:
            raise RuntimeError(f"Could not connect to LLM endpoint {url}: {exc}") from exc

        payload = json.loads(body)
        choices = payload.get("choices") or []
        if not choices:
            raise RuntimeError(f"Malformed LLM response: {payload}")
        message = choices[0].get("message") or {}
        content = message.get("content")
        if not isinstance(content, str):
            raise RuntimeError(f"Missing text content in LLM response: {payload}")
        return content
