# === Register Rewrite Rule Engine (contracts_src_transmogrifier_rules_interface) v1 ===
#  Dependencies: re, dataclasses
# Rule-based register rewriting engine that transforms text between linguistic registers (casual, academic, narrative, technical → direct) using deterministic regex patterns. Zero API calls, sub-millisecond performance.

# Module invariants:
#   - All rule transformations use case-insensitive regex matching (re.IGNORECASE)
#   - Supported register mappings: (casual→direct), (academic→direct), (narrative→direct), (technical→direct)
#   - Empty results after rule application fall back to original text
#   - Rules are applied sequentially in list order
#   - Both string replacements and callable replacements use re.sub with same flags

class RewriteRule:
    """Dataclass representing a single regex-based rewrite transformation"""
    pattern: str                             # required, Regex pattern to match in source text
    replacement: str | Callable              # required, Replacement string or callable for matched pattern

class RuleEngine:
    """Stateless engine that applies register rewrite rules deterministically"""
    pass

def rewrite(
    text: str,
    source: str | enum,
    target: str | enum,
) -> str:
    """
    Rewrite text from source register to target register using regex-based rules. Returns original text if source equals target or no rules exist. Falls back through 'direct' register if no direct path exists.

    Preconditions:
      - text is a string
      - source and target are either strings or objects with .value attribute

    Postconditions:
      - Returns original text if source == target
      - Returns original text if result after all rules is empty string
      - Returns stripped result otherwise
      - If no direct (src, tgt) rules exist and tgt != 'direct', attempts (src, 'direct') first

    Side effects: none
    Idempotent: yes
    """
    ...

# ── REQUIRED EXPORTS ──────────────────────────────────
# Your implementation module MUST export ALL of these names
# with EXACTLY these spellings. Tests import them by name.
# __all__ = ['RewriteRule', 'RuleEngine', 'rewrite']
