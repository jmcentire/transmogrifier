# === Task Classifier (contracts_src_transmogrifier_task_classifier_interface) v1 ===
#  Dependencies: enum, re
# Heuristic-based task type classifier that categorizes text prompts into task types (factual, reasoning, code, analysis, creative, instruction, unknown) using weighted regex pattern matching. Designed for sub-millisecond performance with no API calls.

# Module invariants:
#   - _FACTUAL_MARKERS contains weighted regex patterns for factual questions
#   - _REASONING_MARKERS contains weighted regex patterns for reasoning tasks
#   - _CODE_MARKERS contains weighted regex patterns for code-related tasks
#   - _ANALYSIS_MARKERS contains weighted regex patterns for analytical tasks
#   - _CREATIVE_MARKERS contains weighted regex patterns for creative tasks
#   - _INSTRUCTION_MARKERS contains weighted regex patterns for instructional content
#   - _ALL_MARKERS is immutable list of (TaskType, markers) tuples excluding 'unknown' type
#   - Pattern weights are positive floats reflecting importance of matches
#   - All regex patterns are case-insensitive when applied

class TaskType(Enum):
    """Enumeration of task types for prompt classification"""
    factual = "factual"
    reasoning = "reasoning"
    code = "code"
    analysis = "analysis"
    creative = "creative"
    instruction = "instruction"
    unknown = "unknown"

class TaskClassifier:
    """Classifier that categorizes prompts into task types via heuristics"""
    pass

def classify(
    self: TaskClassifier,
    text: str,
) -> tuple[TaskType, float]:
    """
    Classifies input text into a task type using weighted regex pattern matching across predefined marker sets. Returns the task type with highest score and a confidence value (0-1) based on score separation.

    Preconditions:
      - Input text may be empty or whitespace-only (handled gracefully)

    Postconditions:
      - Returns tuple of (TaskType, confidence) where confidence is in range [0.0, 1.0]
      - Empty or whitespace-only input returns (TaskType.unknown, 0.0)
      - If no patterns match (score == 0), returns (TaskType.unknown, 0.5)
      - Confidence is rounded to 3 decimal places
      - Confidence is computed as min((best_score - second_score) / total + 0.5, 1.0)

    Side effects: none
    Idempotent: yes
    """
    ...

# ── REQUIRED EXPORTS ──────────────────────────────────
# Your implementation module MUST export ALL of these names
# with EXACTLY these spellings. Tests import them by name.
# __all__ = ['TaskType', 'TaskClassifier', 'classify']
