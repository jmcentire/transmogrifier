"""Anthropic backend for Level 3 translation and calibration."""

from __future__ import annotations

import os


class AnthropicBackend:
    def __init__(self, api_key: str | None = None, model: str | None = None):
        self._api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        self._model = model or os.environ.get("TRANSMOG_MODEL", "claude-haiku-4-5-20251001")
        self._client = None

    def _ensure_client(self):
        if self._client is None:
            import anthropic
            self._client = anthropic.Anthropic(api_key=self._api_key)

    def complete(self, system: str, messages: list[dict], max_tokens: int = 1024) -> str:
        self._ensure_client()
        kwargs = {"model": self._model, "max_tokens": max_tokens, "messages": messages}
        if system:
            kwargs["system"] = system
        response = self._client.messages.create(**kwargs)
        return response.content[0].text
