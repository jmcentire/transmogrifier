"""Level 2: Rule-based register rewriting. Zero API calls, <1ms."""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class RewriteRule:
    pattern: str
    replacement: str


# --- Casual → Direct ---
_CASUAL_TO_DIRECT = [
    # Strip informal prefixes
    RewriteRule(r"^(yo |hey |so |um |uh |ok so |alright so )", ""),
    RewriteRule(r"^so like,?\s*", ""),
    RewriteRule(r"^(tell me about|gimme the lowdown on|what's the deal with)\s+", ""),
    RewriteRule(r"^(can you break down|explain .+ to me)\s*[?]?\s*", ""),
    RewriteRule(r"^(what even is|what do you know about)\s+", "What is "),
    RewriteRule(r"^(how does) (.+) (work)\s*[?]?\s*$", r"Explain \2"),
    # Strip informal suffixes
    RewriteRule(r"\.\.\.\s*(what's up with that|right\??|ya know\??|you know\??)?\s*$", ""),
    RewriteRule(r"\s*lol\s*$", ""),
    # Strip filler
    RewriteRule(r"\b(like,? |basically,? |you know,? |i mean,? )", ""),
    # Restore initial capitalization
    RewriteRule(r"^([a-z])", lambda m: m.group(1).upper()),
]

# --- Academic → Direct ---
_ACADEMIC_TO_DIRECT = [
    RewriteRule(r"^in the context of (established |the )?(\w+ )*knowledge,?\s*", ""),
    RewriteRule(r"^it is (well )?established that\s+", ""),
    RewriteRule(r"^as documented in the literature,?\s*", ""),
    RewriteRule(r"\.\s*provide a scholarly response\.?\s*$", ""),
    RewriteRule(r"\b(furthermore|moreover|nevertheless|notwithstanding),?\s*", ""),
    RewriteRule(r"\b(pertaining to|with respect to)\s+", "about "),
    RewriteRule(r"\b(it (?:should be|is) noted that)\s+", ""),
    RewriteRule(r"^([a-z])", lambda m: m.group(1).upper()),
]

# --- Narrative → Direct ---
_NARRATIVE_TO_DIRECT = [
    RewriteRule(r"^explain this as if telling a story:\s*", ""),
    RewriteRule(r"^imagine (that |this: ?)?", ""),
    RewriteRule(r"^picture this:\s*", ""),
    RewriteRule(r"^once upon a time,?\s*", ""),
    RewriteRule(r"^think of it (as|like)\s+", ""),
    RewriteRule(r"^let me tell you about\s+", ""),
    RewriteRule(r"^([a-z])", lambda m: m.group(1).upper()),
]

# --- Technical → Direct (minimal changes) ---
_TECHNICAL_TO_DIRECT = [
    RewriteRule(r"^provide a precise technical answer:\s*", ""),
    RewriteRule(r"^identify the following:\s*", ""),
    RewriteRule(r"^specify:\s*", ""),
    RewriteRule(r"^([a-z])", lambda m: m.group(1).upper()),
]

_RULES: dict[tuple[str, str], list[RewriteRule]] = {
    ("casual", "direct"): _CASUAL_TO_DIRECT,
    ("academic", "direct"): _ACADEMIC_TO_DIRECT,
    ("narrative", "direct"): _NARRATIVE_TO_DIRECT,
    ("technical", "direct"): _TECHNICAL_TO_DIRECT,
}


class RuleEngine:
    """Apply register rewrite rules. Deterministic, <1ms."""

    def rewrite(self, text: str, source, target) -> str:
        """Rewrite text from source register to target register.

        Args:
            source: Register enum or string
            target: Register enum or string

        Returns:
            Rewritten text. If no rules match, returns original.
        """
        src = str(source.value if hasattr(source, "value") else source)
        tgt = str(target.value if hasattr(target, "value") else target)

        if src == tgt:
            return text

        rules = _RULES.get((src, tgt))
        if not rules:
            # No direct path — try routing through direct
            if tgt != "direct" and (src, "direct") in _RULES:
                text = self.rewrite(text, src, "direct")
            return text

        result = text
        for rule in rules:
            if callable(rule.replacement):
                result = re.sub(rule.pattern, rule.replacement, result, flags=re.IGNORECASE)
            else:
                result = re.sub(rule.pattern, rule.replacement, result, flags=re.IGNORECASE)

        return result.strip() or text
