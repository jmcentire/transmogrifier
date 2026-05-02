# === System Prompts (src_transmogrifier_system_prompts) v1 ===
# Provides register-specific system prompt injection templates for normalizing user input across different communication registers (casual, academic, narrative, technical, direct). Implements Level 1 prompt engineering to interpret varied linguistic registers as precise technical queries.

# Module invariants:
#   - GENERIC_NORMALIZATION is a constant fallback prompt for unknown registers
#   - _REGISTER_PROMPTS maps exactly 5 register keys: 'casual', 'academic', 'narrative', 'technical', 'direct'
#   - _REGISTER_PROMPTS['direct'] is always empty string
#   - get_system_prompt always returns a string (empty or populated)
#   - inject_system_prompt is idempotent: multiple applications produce same result as single application

str = primitive  # String type for prompts and register identifiers

def get_system_prompt(
    detected_register: str | Register,
    target_register: str | None = None,
) -> str:
    """
    Retrieves the appropriate Level 1 system prompt injection based on detected register. Returns empty string for 'direct' register. Falls back to GENERIC_NORMALIZATION for unknown registers. Accepts either Register enum instances (with .value attribute) or string values.

    Postconditions:
      - Returns empty string if detected_register == 'direct'
      - Returns register-specific prompt from _REGISTER_PROMPTS if key exists
      - Returns GENERIC_NORMALIZATION for unknown register values
      - Always returns a string (never None)

    Side effects: none
    Idempotent: yes
    """
    ...

def inject_system_prompt(
    existing_system: str,
    injection: str,
) -> str:
    """
    Prepends a register normalization instruction to an existing system prompt. Idempotent operation: if injection is already present in existing_system, returns unchanged. Handles empty/falsy inputs gracefully.

    Postconditions:
      - Returns existing_system unchanged if injection is empty/falsy
      - Returns injection if existing_system is empty/falsy
      - Returns existing_system unchanged if injection substring already present
      - Returns '{injection}\n\n{existing_system}' otherwise
      - Idempotent: calling with same arguments produces same result

    Side effects: none
    Idempotent: yes
    """
    ...

# ── REQUIRED EXPORTS ──────────────────────────────────
# Your implementation module MUST export ALL of these names
# with EXACTLY these spellings. Tests import them by name.
# __all__ = ['get_system_prompt', 'inject_system_prompt']
