"""Tests for core orchestrator."""

import time

from transmogrifier.core import (
    Register,
    Transmogrifier,
    TranslationConfig,
    TranslationLevel,
    TranslationResult,
)


def test_register_enum():
    assert Register.casual.value == "casual"
    assert Register("direct") == Register.direct
    assert len(Register) == 5


def test_translation_result_serialization():
    r = TranslationResult(
        input_text="yo what",
        output_text="What",
        detected_register=Register.casual,
        target_register=Register.direct,
        level_applied=TranslationLevel.rule_rewrite,
    )
    data = r.model_dump()
    assert data["detected_register"] == "casual"
    assert data["level_applied"] == 2
    assert len(data["trace_id"]) == 12


def test_translation_config_defaults():
    c = TranslationConfig()
    assert c.target_register is None
    assert c.max_level == TranslationLevel.rule_rewrite
    assert c.semantic_threshold == 0.95
    assert c.passthrough_on_failure is True


def test_translate_casual_to_direct():
    t = Transmogrifier()
    result = t.translate("yo so like, what's the deal with TCP")
    assert result.detected_register == Register.casual
    assert result.target_register == Register.direct
    assert result.level_applied == TranslationLevel.rule_rewrite
    assert result.system_prompt  # Level 1 should produce a prompt
    assert "yo" not in result.output_text.lower()


def test_translate_direct_stays_direct():
    t = Transmogrifier()
    result = t.translate("What is TCP?")
    assert result.detected_register == Register.direct
    assert result.level_applied == TranslationLevel.system_prompt
    assert result.output_text == "What is TCP?"


def test_translate_skips_invariant_model():
    t = Transmogrifier()
    result = t.translate("yo what's TCP", model="gpt-4o-mini")
    assert result.skipped is True
    assert "invariant" in result.skip_reason


def test_translate_with_override():
    t = Transmogrifier()
    config = TranslationConfig(target_register=Register.technical)
    result = t.translate("yo what's TCP", config=config)
    assert result.target_register == Register.technical


def test_translate_performance():
    t = Transmogrifier()
    start = time.perf_counter()
    for _ in range(1000):
        t.translate("yo so like, what's the deal with activation fingerprinting")
    elapsed_ms = (time.perf_counter() - start) * 1000
    assert elapsed_ms < 5000, f"1000 translations took {elapsed_ms:.0f}ms (>5s)"
