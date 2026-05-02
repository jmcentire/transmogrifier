"""Register detection via surface-form heuristics. Zero API calls, <1ms."""

from __future__ import annotations

import re
from dataclasses import dataclass

# Import Register lazily to avoid circular imports
CASUAL_MARKERS = [
    (r"\b(yo|hey|lol|lmao|nah|yep|yup|gonna|wanna|gotta|kinda|sorta)\b", 2.0),
    (r"\b(like,|basically,|you know,|i mean,|right\?)\b", 1.5),
    (r"^(so |so like |um |uh )", 2.0),
    (r"\.\.\.", 1.0),
    (r"what'?s (the deal|up) with", 2.5),
    (r"(tell me about|gimme|lemme|dunno)", 2.0),
    (r"[!?]{2,}", 1.0),
]

TECHNICAL_MARKERS = [
    (r"^(provide|identify|specify|implement|define|describe|list|explain)\b", 2.0),
    (r"\b(precise|technical|specific|detailed|comprehensive)\b", 1.5),
    (r"\b(e\.g\.|i\.e\.|cf\.|viz\.)\b", 1.5),
    (r"\b(algorithm|function|parameter|interface|protocol|architecture)\b", 1.0),
    (r"^[A-Z][a-z]+ (the|a|an) ", 1.0),
]

ACADEMIC_MARKERS = [
    (r"\b(in the context of|it is (well )?established|as documented)\b", 2.5),
    (r"\b(literature|scholarly|theoretical|empirical|methodolog)\b", 2.0),
    (r"\b(furthermore|moreover|nevertheless|notwithstanding|whilst)\b", 1.5),
    (r"\b(pertaining to|with respect to|in terms of)\b", 1.5),
    (r"\b(arguably|ostensibly|putatively|prima facie)\b", 2.0),
    (r"(is characterized by|can be defined as|has been shown to)\b", 1.5),
]

NARRATIVE_MARKERS = [
    (r"^(imagine|picture this|once upon|think of it as|let me tell you)", 3.0),
    (r"\b(story|tale|journey|adventure|chapter)\b", 1.5),
    (r"\b(as if|like a|just as|much like)\b", 1.0),
    (r"^explain this as if", 3.0),
    (r"\b(hero|villain|protagonist|narrator)\b", 1.5),
]

DIRECT_MARKERS = [
    # Direct is the default — short, unframed queries
    # Detected by absence of other markers + brevity
]


@dataclass
class _FeatureScores:
    casual: float = 0.0
    technical: float = 0.0
    academic: float = 0.0
    narrative: float = 0.0
    direct: float = 0.0


class RegisterDetector:
    """Classify input text into one of 5 registers via heuristics."""

    def detect(self, text: str) -> tuple:
        """Detect register and confidence.

        Returns:
            (Register, confidence) where confidence is 0-1.
            Register is returned as a string to avoid circular imports;
            callers should use Register(result[0]).
        """
        from .core import Register

        if not text or not text.strip():
            return Register.direct, 1.0

        scores = self._score(text)
        scored = [
            (Register.casual, scores.casual),
            (Register.technical, scores.technical),
            (Register.academic, scores.academic),
            (Register.narrative, scores.narrative),
            (Register.direct, scores.direct),
        ]
        scored.sort(key=lambda x: x[1], reverse=True)

        best_reg, best_score = scored[0]
        second_score = scored[1][1] if len(scored) > 1 else 0.0

        if best_score == 0:
            return Register.direct, 0.8

        total = sum(s for _, s in scored) or 1.0
        confidence = min((best_score - second_score) / total + 0.5, 1.0)

        return best_reg, round(confidence, 3)

    def _score(self, text: str) -> _FeatureScores:
        lower = text.lower()
        scores = _FeatureScores()

        for pattern, weight in CASUAL_MARKERS:
            if re.search(pattern, lower, re.IGNORECASE):
                scores.casual += weight

        for pattern, weight in TECHNICAL_MARKERS:
            if re.search(pattern, lower, re.IGNORECASE):
                scores.technical += weight

        for pattern, weight in ACADEMIC_MARKERS:
            if re.search(pattern, lower, re.IGNORECASE):
                scores.academic += weight

        for pattern, weight in NARRATIVE_MARKERS:
            if re.search(pattern, lower, re.IGNORECASE):
                scores.narrative += weight

        # Direct heuristic: short text with no strong markers
        words = text.split()
        if len(words) <= 12:
            scores.direct += 1.5
        if len(words) <= 6:
            scores.direct += 1.5

        # Penalize direct if other scores are high
        max_other = max(scores.casual, scores.technical, scores.academic, scores.narrative)
        if max_other > 2.0:
            scores.direct *= 0.3

        return scores
