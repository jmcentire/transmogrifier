# === Register Detector (src_transmogrifier_detector) v1 ===
#  Dependencies: re, dataclasses, src.transmogrifier.core
# Classifies text into one of five linguistic registers (casual, technical, academic, narrative, direct) using surface-form regex heuristics. Designed for zero API calls and sub-millisecond performance.

# Module invariants:
#   - CASUAL_MARKERS contains 7 (pattern, weight) tuples with weights 1.0-2.5
#   - TECHNICAL_MARKERS contains 5 (pattern, weight) tuples with weights 1.0-2.0
#   - ACADEMIC_MARKERS contains 6 (pattern, weight) tuples with weights 1.5-2.5
#   - NARRATIVE_MARKERS contains 5 (pattern, weight) tuples with weights 1.0-3.0
#   - DIRECT_MARKERS is empty list (direct detected by absence + brevity)
#   - All marker patterns are compiled case-insensitively via re.IGNORECASE
#   - Confidence calculation: min((best_score - second_score) / total + 0.5, 1.0)
#   - Default confidence for all-zero scores is 0.8
#   - Word count thresholds: ≤6 words (strong direct), ≤12 words (moderate direct)
#   - Direct penalty threshold: max_other > 2.0 → multiply direct by 0.3

class _FeatureScores:
    """Internal dataclass holding weighted scores for each register category during detection"""
    casual: float                            # required, Accumulated weight for casual register markers
    technical: float                         # required, Accumulated weight for technical register markers
    academic: float                          # required, Accumulated weight for academic register markers
    narrative: float                         # required, Accumulated weight for narrative register markers
    direct: float                            # required, Accumulated weight for direct register (short, unframed queries)

class RegisterDetector:
    """Main classifier class that uses pattern matching to determine linguistic register"""
    pass

class tuple[Register, float]:
    """Return type for detect method: (Register enum value, confidence score 0-1)"""
    register: Register                       # required, The detected register enum value from src.transmogrifier.core
    confidence: float                        # required, Confidence score between 0 and 1, rounded to 3 decimal places

def detect(
    self: RegisterDetector,
    text: str,
) -> tuple[Register, float]:
    """
    Detects the linguistic register of input text and returns confidence score. Uses weighted pattern matching against predefined marker lists. Empty/whitespace-only text defaults to direct register with high confidence.

    Preconditions:
      - text is a string (may be empty or whitespace-only)

    Postconditions:
      - Returns tuple of (Register, confidence)
      - confidence is between 0 and 1 inclusive
      - confidence is rounded to 3 decimal places
      - Empty or whitespace-only text returns (Register.direct, 1.0)
      - If all scores are 0, returns (Register.direct, 0.8)
      - Register is the highest-scoring category from _score results

    Side effects: Imports Register from .core module on first call (lazy import)
    Idempotent: yes
    """
    ...

def _score(
    self: RegisterDetector,
    text: str,
) -> _FeatureScores:
    """
    Computes weighted scores for each register category by matching regex patterns against input text. Applies length-based heuristics for direct register and penalizes direct score when other categories are strongly present.

    Preconditions:
      - text is a non-empty string

    Postconditions:
      - Returns _FeatureScores with non-negative float values
      - casual score is sum of weights from matching CASUAL_MARKERS patterns
      - technical score is sum of weights from matching TECHNICAL_MARKERS patterns
      - academic score is sum of weights from matching ACADEMIC_MARKERS patterns
      - narrative score is sum of weights from matching NARRATIVE_MARKERS patterns
      - direct score starts at 0, adds 1.5 if ≤12 words, adds another 1.5 if ≤6 words
      - direct score is multiplied by 0.3 if max(other scores) > 2.0

    Side effects: none
    Idempotent: yes
    """
    ...

# ── REQUIRED EXPORTS ──────────────────────────────────
# Your implementation module MUST export ALL of these names
# with EXACTLY these spellings. Tests import them by name.
# __all__ = ['_FeatureScores', 'RegisterDetector', 'tuple[Register, float]', 'detect', '_score']
