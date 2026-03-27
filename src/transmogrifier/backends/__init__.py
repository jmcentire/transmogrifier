"""Backend protocol and factory for Level 3 LLM-based translation."""

from __future__ import annotations

from typing import Protocol


class Backend(Protocol):
    def complete(self, system: str, messages: list[dict], max_tokens: int = 1024) -> str: ...


def create_backend(backend: str | None = None, **kwargs) -> Backend:
    """Factory: create backend from env vars or explicit args."""
    import os
    backend = backend or os.environ.get("TRANSMOG_BACKEND", "anthropic")
    if backend == "anthropic":
        from .anthropic import AnthropicBackend
        return AnthropicBackend(**kwargs)
    elif backend == "openai":
        from .openai import OpenAIBackend
        return OpenAIBackend(**kwargs)
    elif backend == "gemini":
        from .gemini import GeminiBackend
        return GeminiBackend(**kwargs)
    raise ValueError(f"Unknown backend: {backend}")
