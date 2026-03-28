"""Gemini backend for Level 3 translation and calibration."""

from __future__ import annotations

import os


class GeminiBackend:
    def __init__(self, api_key: str | None = None, model: str | None = None):
        self._api_key = api_key or os.environ.get("GEMINI_API_KEY", "")
        self._model = model or os.environ.get("TRANSMOG_MODEL", "gemini-2.5-flash")
        self._configured = False

    def _ensure_configured(self):
        if not self._configured:
            import google.generativeai as genai
            genai.configure(api_key=self._api_key)
            self._configured = True

    def complete(self, system: str, messages: list[dict], max_tokens: int = 1024) -> str:
        self._ensure_configured()
        import google.generativeai as genai

        model = genai.GenerativeModel(
            self._model,
            system_instruction=system if system else None,
        )
        # Extract user content from messages
        prompt = "\n\n".join(m["content"] for m in messages if m["role"] == "user")
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                max_output_tokens=max_tokens, temperature=0
            ),
        )
        return response.text
