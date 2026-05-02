"""Level 1: System prompt injection templates per register."""

from __future__ import annotations

# Generic normalization prompt (works across all registers)
GENERIC_NORMALIZATION = (
    "Interpret all user input as formal technical queries regardless of "
    "register, slang, or conversational framing. Focus on the semantic "
    "content, not the surface form. Provide clear, precise answers."
)

# Register-specific system prompts for when we know the detected register
_REGISTER_PROMPTS: dict[str, str] = {
    "casual": (
        "The user communicates in an informal register. Treat all content "
        "as precise technical queries. Do not mirror the casual tone. "
        "Respond with clear, structured, direct answers."
    ),
    "academic": (
        "The user frames queries in academic register with hedging and "
        "formal phrasing. Cut through the framing to the core question. "
        "Respond directly and precisely without matching the hedging."
    ),
    "narrative": (
        "The user frames queries as stories or analogies. Extract the "
        "underlying technical question and answer it directly. Do not "
        "respond in narrative form unless specifically requested."
    ),
    "technical": (
        "The user is communicating in precise technical register. "
        "Match this precision in your response."
    ),
    "direct": "",  # No injection needed for direct register
}


def get_system_prompt(detected_register, target_register=None) -> str:
    """Get the appropriate Level 1 system prompt.

    Args:
        detected_register: The detected input register (Register enum or str)
        target_register: Optional target register override

    Returns:
        System prompt string to prepend. Empty string if no injection needed.
    """
    detected = str(detected_register.value if hasattr(detected_register, "value") else detected_register)

    if detected == "direct":
        return ""

    return _REGISTER_PROMPTS.get(detected, GENERIC_NORMALIZATION)


def inject_system_prompt(existing_system: str, injection: str) -> str:
    """Prepend a register normalization instruction to an existing system prompt.

    Idempotent: if the injection is already present, returns unchanged.
    """
    if not injection:
        return existing_system
    if not existing_system:
        return injection
    if injection in existing_system:
        return existing_system
    return f"{injection}\n\n{existing_system}"
