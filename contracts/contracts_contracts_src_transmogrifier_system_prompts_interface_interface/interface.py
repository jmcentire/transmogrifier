# === System Prompt Injection Templates Interface (contracts_contracts_src_transmogrifier_system_prompts_interface_interface) v1 ===
#  Dependencies: enum
# Provides register-specific system prompt templates for normalizing user input across different communication registers (casual, academic, narrative, technical, direct). Implements Level 1 system prompt injection to ensure consistent technical interpretation regardless of input framing.

# Module invariants:
#   - GENERIC_NORMALIZATION is a non-empty constant string
#   - _REGISTER_PROMPTS contains keys: 'casual', 'academic', 'narrative', 'technical', 'direct'
#   - _REGISTER_PROMPTS['direct'] is always empty string
#   - All _REGISTER_PROMPTS values are strings
#   - get_system_prompt always returns a string (never None)
#   - inject_system_prompt is idempotent for any given (existing_system, injection) pair

class Register(Enum):
    """Register enum type representing communication registers with .value attribute"""
    casual = "casual"
    academic = "academic"
    narrative = "narrative"
    technical = "technical"
    direct = "direct"

def get_system_prompt(
    detected_register: str | Register,
    target_register: str | None = None,
) -> str:
    """
    Retrieves the appropriate Level 1 system prompt for a detected register. Returns register-specific normalization instructions from _REGISTER_PROMPTS dictionary, or GENERIC_NORMALIZATION as fallback. Returns empty string for 'direct' register.

    Postconditions:
      - Returns empty string if detected_register is 'direct'
      - Returns register-specific prompt if detected_register in _REGISTER_PROMPTS
      - Returns GENERIC_NORMALIZATION for unknown registers
      - Return value is always a string (never None)

    Side effects: none
    Idempotent: yes
    """
    ...

def inject_system_prompt(
    existing_system: str,
    injection: str,
) -> str:
    """
    Prepends a register normalization instruction to an existing system prompt. Idempotent: if the injection is already present in existing_system, returns unchanged. Handles empty string cases gracefully.

    Postconditions:
      - Returns existing_system unchanged if injection is empty/falsy
      - Returns injection if existing_system is empty/falsy
      - Returns existing_system unchanged if injection already present (substring match)
      - Returns '{injection}\n\n{existing_system}' otherwise
      - Function is idempotent: inject_system_prompt(inject_system_prompt(a, b), b) == inject_system_prompt(a, b)

    Side effects: none
    Idempotent: yes
    """
    ...

# ── REQUIRED EXPORTS ──────────────────────────────────
# Your implementation module MUST export ALL of these names
# with EXACTLY these spellings. Tests import them by name.
# __all__ = ['Register', 'get_system_prompt', 'inject_system_prompt']
