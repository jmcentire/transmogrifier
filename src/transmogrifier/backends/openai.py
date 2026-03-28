"""OpenAI backend for Level 3 translation and calibration."""

from __future__ import annotations

import os


class OpenAIBackend:
    def __init__(self, api_key: str | None = None, model: str | None = None):
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self._model = model or os.environ.get("TRANSMOG_MODEL", "gpt-4o-mini")
        self._client = None

    def _ensure_client(self):
        if self._client is None:
            import openai
            self._client = openai.OpenAI(api_key=self._api_key)

    def complete(self, system: str, messages: list[dict], max_tokens: int = 1024) -> str:
        self._ensure_client()
        msgs = []
        if system:
            msgs.append({"role": "system", "content": system})
        msgs.extend(messages)
        response = self._client.chat.completions.create(
            model=self._model, messages=msgs, max_tokens=max_tokens, temperature=0
        )
        return response.choices[0].message.content
