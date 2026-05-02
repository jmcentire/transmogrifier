# === Transmogrifier Backend Protocol and Factory (src_transmogrifier_backends___init__) v1 ===
#  Dependencies: os, typing.Protocol, src.transmogrifier.backends.anthropic.AnthropicBackend, src.transmogrifier.backends.openai.OpenAIBackend, src.transmogrifier.backends.gemini.GeminiBackend
# Backend protocol and factory for Level 3 LLM-based translation. Defines the Backend protocol interface for LLM completion and provides a factory function to instantiate backend implementations (Anthropic, OpenAI, Gemini) based on environment variables or explicit parameters.

# Module invariants:
#   - Default backend is 'anthropic' when TRANSMOG_BACKEND is not set
#   - Only three backend types are supported: 'anthropic', 'openai', 'gemini'
#   - All backends must implement the Backend protocol (complete method)

class Backend:
    """Protocol class defining the interface for LLM backend implementations. Any conforming backend must implement the complete() method."""
    pass

def create_backend(
    backend: str | None = None,
    kwargs: dict = {},
) -> Backend:
    """
    Factory function that creates a backend instance from environment variables or explicit arguments. Reads TRANSMOG_BACKEND env var (defaults to 'anthropic') and dynamically imports the corresponding backend implementation.

    Postconditions:
      - Returns a Backend instance that implements the complete() method
      - The returned backend is one of: AnthropicBackend, OpenAIBackend, or GeminiBackend

    Errors:
      - unknown_backend (ValueError): backend parameter (or TRANSMOG_BACKEND env var) is not one of 'anthropic', 'openai', or 'gemini'
          message: Unknown backend: {backend}

    Side effects: Reads environment variable TRANSMOG_BACKEND, Dynamically imports backend modules based on selection
    Idempotent: no
    """
    ...

def complete(
    system: str,
    messages: list[dict],
    max_tokens: int = 1024,
) -> str:
    """
    Protocol method for LLM completion. Takes a system prompt, message list, and max token count, returns a string completion. This is a protocol method signature, not a concrete implementation.

    Postconditions:
      - Returns a string completion from the LLM

    Side effects: none
    Idempotent: no
    """
    ...

# ── REQUIRED EXPORTS ──────────────────────────────────
# Your implementation module MUST export ALL of these names
# with EXACTLY these spellings. Tests import them by name.
# __all__ = ['Backend', 'create_backend', 'complete']
