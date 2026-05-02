# === Task Classifier (src_transmogrifier_task_classifier) v1 ===
#  Dependencies: enum, re
# Heuristic-based task type classifier using regex pattern matching. Categorizes text prompts into task types (factual, reasoning, code, analysis, creative, instruction, unknown) with confidence scores. Runs in <1ms with no API calls.

# Module invariants:
#   - Weighted marker patterns are immutable constants
#   - TaskType.unknown is never included in scoring calculations
#   - Confidence calculation: min((best_score - second_score) / total + 0.5, 1.0)
#   - Pattern matching is case-insensitive
#   - All regex patterns are pre-defined and do not change at runtime
#   - Total of 6 classification categories (excluding unknown): factual, reasoning, code, analysis, creative, instruction

class TaskType(Enum):
    """Enumeration of task types for classification"""
    factual = "factual"
    reasoning = "reasoning"
    code = "code"
    analysis = "analysis"
    creative = "creative"
    instruction = "instruction"
    unknown = "unknown"

class TaskClassifier:
    """Classifier class for categorizing prompts into task types via heuristics"""
    pass

def classify(
    self: TaskClassifier,
    text: str,
) -> tuple[TaskType, float]:
    """
    Classifies input text into a task type using weighted regex pattern matching against predefined markers. Returns the best matching task type and a confidence score between 0 and 1.

    Postconditions:
      - Returns tuple with TaskType and confidence score
      - Confidence score is in range [0.0, 1.0]
      - Confidence is rounded to 3 decimal places
      - Empty or whitespace-only input returns (TaskType.unknown, 0.0)
      - If no patterns match, returns (TaskType.unknown, 0.5)

    Side effects: none
    Idempotent: yes
    """
    ...

# ── REQUIRED EXPORTS ──────────────────────────────────
# Your implementation module MUST export ALL of these names
# with EXACTLY these spellings. Tests import them by name.
# __all__ = ['TaskType', 'TaskClassifier', 'classify']
