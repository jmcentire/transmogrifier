# === Anthropic Backend (src_transmogrifier_backends_anthropic) v1 ===
#  Dependencies: os, anthropic
# Anthropic backend for Level 3 translation and calibration. Provides a wrapper around the Anthropic API client for completing chat messages using Claude models.

# Module invariants:
#   - Default model is 'claude-haiku-4-5-20251001' if not specified
#   - _client is None until first use, then remains initialized
#   - _api_key and _model are immutable after initialization

class AnthropicBackend:
    """Backend class that interfaces with Anthropic's API for chat completions. Lazy-loads the client on first use."""
    _api_key: str                            # required, Anthropic API key, sourced from parameter or ANTHROPIC_API_KEY env var
    _model: str                              # required, Model identifier, sourced from parameter or TRANSMOG_MODEL env var with default 'claude-haiku-4-5-20251001'
    _client: anthropic.Anthropic | None      # required, Lazily initialized Anthropic client instance

def __init__(
    api_key: str | None = None,
    model: str | None = None,
) -> None:
    """
    Initializes the AnthropicBackend with optional API key and model. Falls back to environment variables ANTHROPIC_API_KEY and TRANSMOG_MODEL, with a default model of 'claude-haiku-4-5-20251001'. Client is not instantiated until first use.

    Postconditions:
      - _api_key is set to provided value or environment variable or empty string
      - _model is set to provided value or environment variable or 'claude-haiku-4-5-20251001'
      - _client is None (not yet initialized)

    Side effects: reads environment variables ANTHROPIC_API_KEY and TRANSMOG_MODEL
    Idempotent: no
    """
    ...

def _ensure_client() -> None:
    """
    Lazy initialization method that creates the Anthropic client if it hasn't been created yet. Imports the anthropic module and instantiates the client with the stored API key.

    Postconditions:
      - _client is not None after execution
      - _client is an instance of anthropic.Anthropic

    Errors:
      - ImportError (ImportError): anthropic package is not installed
      - AuthenticationError (anthropic.AuthenticationError): _api_key is invalid or empty when client is instantiated

    Side effects: imports anthropic module, instantiates anthropic.Anthropic client with stored API key
    Idempotent: yes
    """
    ...

def complete(
    system: str,
    messages: list[dict],
    max_tokens: int = 1024,
) -> str:
    """
    Sends a chat completion request to Anthropic's API. Ensures the client is initialized, constructs the message payload with model, max_tokens, messages, and optional system prompt, then returns the text content from the first response block.

    Preconditions:
      - messages is a non-empty list of valid message dictionaries
      - max_tokens is positive

    Postconditions:
      - returns text content from response.content[0].text
      - _client is initialized

    Errors:
      - ImportError (ImportError): anthropic package not installed (raised during _ensure_client)
      - AuthenticationError (anthropic.AuthenticationError): Invalid or missing API key
      - APIError (anthropic.APIError): Anthropic API returns error response (invalid model, rate limit, etc.)
      - IndexError (IndexError): response.content is empty or does not contain expected structure
      - AttributeError (AttributeError): response.content[0] does not have .text attribute

    Side effects: calls _ensure_client which may initialize client, makes network call to Anthropic API
    Idempotent: no
    """
    ...

# ── REQUIRED EXPORTS ──────────────────────────────────
# Your implementation module MUST export ALL of these names
# with EXACTLY these spellings. Tests import them by name.
# __all__ = ['AnthropicBackend', '_ensure_client', 'ImportError', 'complete']
