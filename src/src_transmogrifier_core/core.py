"""Core orchestrator: Transmogrifier class, enums, result types."""

from __future__ import annotations

import enum
import time
import uuid

from pydantic import BaseModel, Field

from .detector import RegisterDetector
from .profiles import ProfileCache
from .rules import RuleEngine
from .system_prompts import get_system_prompt
from .task_classifier import TaskClassifier, TaskType


class Register(str, enum.Enum):
    direct = "direct"
    casual = "casual"
    technical = "technical"
    academic = "academic"
    narrative = "narrative"


class TranslationLevel(int, enum.Enum):
    system_prompt = 1
    rule_rewrite = 2
    llm_translate = 3


class TranslationResult(BaseModel):
    input_text: str
    output_text: str
    detected_register: Register
    target_register: Register
    detected_task: str = ""
    level_applied: TranslationLevel
    system_prompt: str | None = None
    semantic_similarity: float | None = None
    skipped: bool = False
    skip_reason: str | None = None
    elapsed_ms: float = 0.0
    trace_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])


class TranslationConfig(BaseModel):
    target_register: Register | None = None
    max_level: TranslationLevel = TranslationLevel.rule_rewrite
    semantic_threshold: float = 0.95
    spread_threshold_pp: float = 10.0
    passthrough_on_failure: bool = True
    task_aware: bool = True  # use per-task register when available


class Transmogrifier:
    """Main entry point for register-aware prompt translation.

    Usage:
        t = Transmogrifier()
        result = t.translate("yo what's the deal with TCP", model="claude-opus-4")
        # result.output_text  -> rewritten prompt
        # result.system_prompt -> Level 1 injection to prepend
    """

    def __init__(
        self,
        profile_cache: ProfileCache | None = None,
        config: TranslationConfig | None = None,
    ) -> None:
        self._detector = RegisterDetector()
        self._task_classifier = TaskClassifier()
        self._profile_cache = profile_cache or ProfileCache()
        self._rule_engine = RuleEngine()
        self._config = config or TranslationConfig()

    def translate(
        self,
        text: str,
        model: str = "",
        config: TranslationConfig | None = None,
    ) -> TranslationResult:
        """Translate text to optimal register. Zero API calls."""
        t0 = time.perf_counter()
        config = config or self._config

        detected, confidence = self._detector.detect(text)
        task_type, task_conf = self._task_classifier.classify(text)
        profile = self._profile_cache.get(model) if model else None

        # Skip for invariant models — but check per-task too
        if profile and profile.is_invariant:
            # Even if aggregate is invariant, per-task might not be
            task_spread = profile.spread_for_task(task_type.value) if config.task_aware else 0
            if task_spread < 2.0:
                return TranslationResult(
                    input_text=text,
                    output_text=text,
                    detected_register=detected,
                    target_register=detected,
                    detected_task=task_type.value,
                    level_applied=TranslationLevel.system_prompt,
                    skipped=True,
                    skip_reason=f"invariant model ({profile.spread_pp:.1f}pp spread)",
                    elapsed_ms=(time.perf_counter() - t0) * 1000,
                )

        # Determine target register
        if config.target_register:
            target = config.target_register
        elif profile and config.task_aware and task_type != TaskType.unknown:
            # Use per-task optimal register if available
            target = profile.best_register_for_task(task_type.value)
        elif profile:
            target = profile.best_register
        else:
            target = Register.direct

        # Ensure target is a Register enum
        if isinstance(target, str):
            target = Register(target)

        # Level 1: always generate system prompt
        sys_prompt = get_system_prompt(detected, target)

        # Level 2: rule-based rewrite if source != target
        output_text = text
        level = TranslationLevel.system_prompt
        if detected != target:
            output_text = self._rule_engine.rewrite(text, detected, target)
            level = TranslationLevel.rule_rewrite

        elapsed = (time.perf_counter() - t0) * 1000
        return TranslationResult(
            input_text=text,
            output_text=output_text,
            detected_register=detected,
            target_register=target,
            detected_task=task_type.value,
            level_applied=level,
            system_prompt=sys_prompt,
            elapsed_ms=elapsed,
        )
