"""Task type classifier. Heuristic-based, <1ms, no API calls."""

from __future__ import annotations

import enum
import re


class TaskType(str, enum.Enum):
    factual = "factual"
    reasoning = "reasoning"
    code = "code"
    analysis = "analysis"
    creative = "creative"
    instruction = "instruction"
    unknown = "unknown"


# Weighted markers per task type
_FACTUAL_MARKERS = [
    (r"\b(what is|who is|when did|where is|how many|how much|which)\b", 2.0),
    (r"\b(capital of|symbol for|largest|smallest|first|last)\b", 2.0),
    (r"\b(name the|list the|what year|what country)\b", 1.5),
    (r"\b(define|definition of)\b", 1.5),
    (r"\?$", 0.5),
]

_REASONING_MARKERS = [
    (r"\b(if .+ then|can we conclude|does it follow|is it valid)\b", 3.0),
    (r"\b(therefore|logically|implies|argument|premise|syllogism)\b", 2.0),
    (r"\b(how (many|long|much) would|cost .+ total)\b", 2.0),
    (r"\b(solve|deduce|infer|prove|disprove)\b", 2.0),
    (r"\b(puzzle|riddle|trick question|brain teaser)\b", 1.5),
    (r"\ball .+ are .+ and\b", 2.0),
]

_CODE_MARKERS = [
    (r"\b(write a|implement|code|function|class|method|script)\b", 3.0),
    (r"\b(python|javascript|rust|java|golang|typescript|sql|bash)\b", 2.5),
    (r"\b(algorithm|data structure|api|endpoint|regex)\b", 1.5),
    (r"\b(one-liner|snippet|refactor|debug|fix the bug)\b", 2.0),
    (r"\b(def |return |import |for .+ in |while )\b", 2.0),
    (r"```", 3.0),
]

_ANALYSIS_MARKERS = [
    (r"\b(difference[s]? between|compare|contrast|pros and cons)\b", 3.5),
    (r"\b(explain the|how does .+ work|why does|what causes)\b", 2.0),
    (r"\b(advantage|disadvantage|trade.?off|implication)\b", 1.5),
    (r"\b(architecture|design|pattern|principle|theorem)\b", 1.5),
    (r"\b(relationship between|impact of|role of)\b", 1.5),
    (r"\b(compare and|versus|vs\.?)\b", 2.5),
]

_CREATIVE_MARKERS = [
    (r"\b(write a (poem|story|essay|song|joke|haiku|limerick))\b", 4.0),
    (r"\b(creative|imaginative|fiction|narrative|metaphor)\b", 2.0),
    (r"\b(brainstorm|generate ideas|come up with)\b", 1.5),
    (r"\b(rewrite .+ as|in the style of|tone of)\b", 2.0),
]

_INSTRUCTION_MARKERS = [
    (r"\b(how to|step.?by.?step|tutorial|guide|instructions)\b", 3.0),
    (r"\b(walk me through|show me how|teach me)\b", 2.0),
    (r"\b(best practice|recommended|should I)\b", 1.5),
    (r"\b(setup|install|configure|deploy|migrate)\b", 1.5),
]

_ALL_MARKERS = [
    (TaskType.factual, _FACTUAL_MARKERS),
    (TaskType.reasoning, _REASONING_MARKERS),
    (TaskType.code, _CODE_MARKERS),
    (TaskType.analysis, _ANALYSIS_MARKERS),
    (TaskType.creative, _CREATIVE_MARKERS),
    (TaskType.instruction, _INSTRUCTION_MARKERS),
]


class TaskClassifier:
    """Classify prompt into task type via heuristics."""

    def classify(self, text: str) -> tuple[TaskType, float]:
        """Returns (task_type, confidence 0-1)."""
        if not text or not text.strip():
            return TaskType.unknown, 0.0

        lower = text.lower()
        scores: dict[TaskType, float] = {tt: 0.0 for tt in TaskType if tt != TaskType.unknown}

        for task_type, markers in _ALL_MARKERS:
            for pattern, weight in markers:
                if re.search(pattern, lower, re.IGNORECASE):
                    scores[task_type] += weight

        best_type = max(scores, key=scores.get)
        best_score = scores[best_type]

        if best_score == 0:
            return TaskType.unknown, 0.5

        sorted_scores = sorted(scores.values(), reverse=True)
        second = sorted_scores[1] if len(sorted_scores) > 1 else 0.0
        total = sum(scores.values()) or 1.0
        confidence = min((best_score - second) / total + 0.5, 1.0)

        return best_type, round(confidence, 3)
