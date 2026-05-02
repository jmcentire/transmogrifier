# === Register Detector Interface (contracts_src_transmogrifier_detector_interface) v1 ===
#  Dependencies: re, dataclasses, src.transmogrifier.core
# Heuristic-based text register classifier that analyzes surface-form patterns to categorize input text into one of five linguistic registers (casual, technical, academic, narrative, direct) with confidence scoring. Implements zero-API stateless classification via regex pattern matching with sub-millisecond performance.

# Module invariants:
#   - CASUAL_MARKERS contains exactly 7 (pattern, weight) tuples with weights in range [1.0, 2.5]
#   - TECHNICAL_MARKERS contains exactly 5 (pattern, weight) tuples with weights in range [1.0, 2.0]
#   - ACADEMIC_MARKERS contains exactly 6 (pattern, weight) tuples with weights in range [1.5, 2.5]
#   - NARRATIVE_MARKERS contains exactly 5 (pattern, weight) tuples with weights in range [1.0, 3.0]
#   - DIRECT_MARKERS is empty (direct register detected by absence of markers plus brevity)
#   - All pattern matching is case-insensitive via re.IGNORECASE or .lower()
#   - RegisterDetector is stateless — all calls are independent
#   - Confidence output is always in range [0.0, 1.0] and rounded to 3 decimal places
#   - Empty or whitespace-only text always returns (Register.direct, 1.0)
#   - Word count threshold for direct boost: <= 12 words adds +1.5, <= 6 words adds additional +1.5
#   - Direct penalty threshold: max(other_scores) > 2.0 triggers direct_score *= 0.3

class _FeatureScores:
    """Internal dataclass holding weighted scores for each register category. Used as intermediate representation before final register selection."""
    casual: float                            # required, Cumulative weight from casual language pattern matches
    technical: float                         # required, Cumulative weight from technical language pattern matches
    academic: float                          # required, Cumulative weight from academic language pattern matches
    narrative: float                         # required, Cumulative weight from narrative language pattern matches
    direct: float                            # required, Score for direct register based on brevity and absence of other markers

class RegisterDetector:
    """Stateless classifier that categorizes input text into one of 5 linguistic registers using regex-based pattern matching. No instance state; all classification logic uses module-level constants."""
    pass

def detect(
    self: RegisterDetector,
    text: str,
) -> tuple[Register, float]:
    """
    Detects the linguistic register of input text using pattern-based heuristics. Returns the most likely register and a confidence score. Empty or whitespace-only text defaults to 'direct' register with 1.0 confidence.

    Preconditions:
      - text is a string (may be empty)

    Postconditions:
      - Returns tuple of (Register, confidence)
      - confidence is in range [0.0, 1.0]
      - confidence is rounded to 3 decimal places
      - Empty or whitespace-only text returns (Register.direct, 1.0)
      - If all scores are 0, returns (Register.direct, 0.8)
      - confidence = min((best_score - second_score) / total + 0.5, 1.0)

    Side effects: Imports .core.Register module (lazy import)
    Idempotent: yes
    """
    ...

def _score(
    self: RegisterDetector,
    text: str,
) -> _FeatureScores:
    """
    Internal scoring function that computes weighted scores for each register category based on regex pattern matches. Applies heuristics for brevity to boost 'direct' register score, and penalizes 'direct' when other strong markers are present.

    Preconditions:
      - text is a non-empty string (caller handles empty case)

    Postconditions:
      - Returns _FeatureScores with all five register scores >= 0.0
      - casual score = sum of weights for matched CASUAL_MARKERS patterns
      - technical score = sum of weights for matched TECHNICAL_MARKERS patterns
      - academic score = sum of weights for matched ACADEMIC_MARKERS patterns
      - narrative score = sum of weights for matched NARRATIVE_MARKERS patterns
      - direct score starts at 0, +1.5 if words <= 12, +1.5 if words <= 6
      - direct score *= 0.3 if max(casual, technical, academic, narrative) > 2.0

    Side effects: none
    Idempotent: yes
    """
    ...

# ── REQUIRED EXPORTS ──────────────────────────────────
# Your implementation module MUST export ALL of these names
# with EXACTLY these spellings. Tests import them by name.
# __all__ = ['_FeatureScores', 'RegisterDetector', 'detect', '_score']
