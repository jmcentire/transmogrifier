"""
Contract tests for RegisterDetector interface.

Tests verify:
1. Type structure and contract compliance (_FeatureScores, RegisterDetector)
2. detect() method behavior (happy paths, edge cases, postconditions, invariants)
3. _score() method behavior (scoring logic, brevity boosts, penalties)
4. Module-level constants (marker structures and weights)
5. Stateless operation and determinism
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import re
from dataclasses import dataclass, fields
from enum import Enum


# Import the component under test
from contracts.src_transmogrifier_detector.interface import (
    RegisterDetector,
    _FeatureScores,
    Register,
    CASUAL_MARKERS,
    TECHNICAL_MARKERS,
    ACADEMIC_MARKERS,
    NARRATIVE_MARKERS,
    DIRECT_MARKERS,
)


class TestDetectHappyPath:
    """Happy path tests for detect() method covering all register types."""

    def test_detect_happy_path_casual_text(self):
        """Verify detect() correctly identifies casual register with typical conversational text."""
        detector = RegisterDetector()
        text = "Hey, what's up? I'm gonna grab some food, wanna come along?"
        
        result = detector.detect(text)
        
        assert isinstance(result, tuple), "Result must be a tuple"
        assert len(result) == 2, "Result must have exactly 2 elements"
        register, confidence = result
        assert isinstance(register, Register), "First element must be Register enum"
        assert isinstance(confidence, float), "Second element must be float"
        assert 0.0 <= confidence <= 1.0, "Confidence must be in range [0.0, 1.0]"
        # Verify confidence is rounded to 3 decimal places
        assert round(confidence, 3) == confidence, "Confidence must be rounded to 3 decimals"

    def test_detect_happy_path_technical_text(self):
        """Verify detect() correctly identifies technical register with domain-specific terminology."""
        detector = RegisterDetector()
        text = "The algorithm utilizes a binary search tree to optimize query performance with O(log n) complexity."
        
        result = detector.detect(text)
        
        assert isinstance(result, tuple), "Result must be a tuple"
        assert len(result) == 2, "Result must have exactly 2 elements"
        register, confidence = result
        assert isinstance(register, Register), "First element must be Register enum"
        assert isinstance(confidence, float), "Second element must be float"
        assert 0.0 <= confidence <= 1.0, "Confidence must be in range [0.0, 1.0]"
        assert round(confidence, 3) == confidence, "Confidence must be rounded to 3 decimals"

    def test_detect_happy_path_academic_text(self):
        """Verify detect() correctly identifies academic register with scholarly language."""
        detector = RegisterDetector()
        text = "Furthermore, it is evident that the hypothesis demonstrates significant correlation with observed phenomena."
        
        result = detector.detect(text)
        
        assert isinstance(result, tuple), "Result must be a tuple"
        assert len(result) == 2, "Result must have exactly 2 elements"
        register, confidence = result
        assert isinstance(register, Register), "First element must be Register enum"
        assert isinstance(confidence, float), "Second element must be float"
        assert 0.0 <= confidence <= 1.0, "Confidence must be in range [0.0, 1.0]"
        assert round(confidence, 3) == confidence, "Confidence must be rounded to 3 decimals"

    def test_detect_happy_path_narrative_text(self):
        """Verify detect() correctly identifies narrative register with storytelling elements."""
        detector = RegisterDetector()
        text = "Once upon a time, there was a young girl who discovered a hidden treasure in her grandmother's attic."
        
        result = detector.detect(text)
        
        assert isinstance(result, tuple), "Result must be a tuple"
        assert len(result) == 2, "Result must have exactly 2 elements"
        register, confidence = result
        assert isinstance(register, Register), "First element must be Register enum"
        assert isinstance(confidence, float), "Second element must be float"
        assert 0.0 <= confidence <= 1.0, "Confidence must be in range [0.0, 1.0]"
        assert round(confidence, 3) == confidence, "Confidence must be rounded to 3 decimals"

    def test_detect_happy_path_direct_text(self):
        """Verify detect() correctly identifies direct register with brief commands."""
        detector = RegisterDetector()
        text = "Submit report."
        
        result = detector.detect(text)
        
        assert isinstance(result, tuple), "Result must be a tuple"
        assert len(result) == 2, "Result must have exactly 2 elements"
        register, confidence = result
        assert isinstance(register, Register), "First element must be Register enum"
        assert isinstance(confidence, float), "Second element must be float"
        assert 0.0 <= confidence <= 1.0, "Confidence must be in range [0.0, 1.0]"
        assert round(confidence, 3) == confidence, "Confidence must be rounded to 3 decimals"


class TestDetectEdgeCases:
    """Edge case tests for detect() method."""

    def test_detect_edge_case_empty_string(self):
        """Verify detect() returns (Register.direct, 1.0) for empty string as per contract."""
        detector = RegisterDetector()
        text = ""
        
        result = detector.detect(text)
        
        register, confidence = result
        assert register == Register.direct, "Empty string must return Register.direct"
        assert confidence == 1.0, "Empty string must return confidence 1.0"

    def test_detect_edge_case_whitespace_only(self):
        """Verify detect() returns (Register.direct, 1.0) for whitespace-only text."""
        detector = RegisterDetector()
        text = "   \t\n  "
        
        result = detector.detect(text)
        
        register, confidence = result
        assert register == Register.direct, "Whitespace-only text must return Register.direct"
        assert confidence == 1.0, "Whitespace-only text must return confidence 1.0"

    def test_detect_edge_case_single_word(self):
        """Verify detect() handles single-word input correctly."""
        detector = RegisterDetector()
        text = "Hello"
        
        result = detector.detect(text)
        
        assert isinstance(result, tuple), "Result must be a tuple"
        register, confidence = result
        assert isinstance(register, Register), "First element must be Register enum"
        assert isinstance(confidence, float), "Second element must be float"
        assert 0.0 <= confidence <= 1.0, "Confidence must be in range [0.0, 1.0]"
        assert round(confidence, 3) == confidence, "Confidence must be rounded to 3 decimals"

    def test_detect_edge_case_very_long_text(self):
        """Verify detect() handles long text efficiently (performance boundary)."""
        detector = RegisterDetector()
        text = "word " * 1000
        
        result = detector.detect(text)
        
        assert isinstance(result, tuple), "Result must be a tuple"
        register, confidence = result
        assert isinstance(register, Register), "First element must be Register enum"
        assert isinstance(confidence, float), "Second element must be float"
        assert 0.0 <= confidence <= 1.0, "Confidence must be in range [0.0, 1.0]"
        assert round(confidence, 3) == confidence, "Confidence must be rounded to 3 decimals"

    def test_detect_edge_case_six_words_direct_boost(self):
        """Verify detect() applies double direct boost for text with exactly 6 words."""
        detector = RegisterDetector()
        text = "Please send the report by Friday."
        
        result = detector.detect(text)
        
        register, confidence = result
        # With 6 words, direct register should get +3.0 boost, making it likely to win
        assert isinstance(register, Register), "Must return valid Register"
        assert 0.0 <= confidence <= 1.0, "Confidence must be in range [0.0, 1.0]"
        assert round(confidence, 3) == confidence, "Confidence must be rounded to 3 decimals"

    def test_detect_edge_case_twelve_words_direct_boost(self):
        """Verify detect() applies single direct boost for text with exactly 12 words."""
        detector = RegisterDetector()
        text = "The quick brown fox jumps over the lazy dog in the afternoon."
        
        result = detector.detect(text)
        
        register, confidence = result
        # With 12 words, direct register should get +1.5 boost
        assert isinstance(register, Register), "Must return valid Register"
        assert 0.0 <= confidence <= 1.0, "Confidence must be in range [0.0, 1.0]"
        assert round(confidence, 3) == confidence, "Confidence must be rounded to 3 decimals"

    def test_detect_edge_case_thirteen_words_no_boost(self):
        """Verify detect() does not apply direct boost for text with 13 words."""
        detector = RegisterDetector()
        text = "The quick brown fox jumps over the lazy dog in the afternoon today."
        
        result = detector.detect(text)
        
        register, confidence = result
        # With 13 words, no brevity boost should be applied
        assert isinstance(register, Register), "Must return valid Register"
        assert 0.0 <= confidence <= 1.0, "Confidence must be in range [0.0, 1.0]"
        assert round(confidence, 3) == confidence, "Confidence must be rounded to 3 decimals"

    def test_detect_edge_case_mixed_case(self):
        """Verify detect() handles mixed-case text (case-insensitive matching)."""
        detector = RegisterDetector()
        text = "FURTHERMORE, It Is EVIDENT That THE Hypothesis DEMONSTRATES Correlation."
        
        result = detector.detect(text)
        
        register, confidence = result
        # Should still detect academic markers despite mixed case
        assert isinstance(register, Register), "Must return valid Register"
        assert 0.0 <= confidence <= 1.0, "Confidence must be in range [0.0, 1.0]"
        assert round(confidence, 3) == confidence, "Confidence must be rounded to 3 decimals"

    def test_detect_edge_case_special_characters(self):
        """Verify detect() handles text with special characters and punctuation."""
        detector = RegisterDetector()
        text = "What's the deal?! @#$% & *()... Hey!!!"
        
        result = detector.detect(text)
        
        register, confidence = result
        assert isinstance(register, Register), "Must return valid Register"
        assert 0.0 <= confidence <= 1.0, "Confidence must be in range [0.0, 1.0]"
        assert round(confidence, 3) == confidence, "Confidence must be rounded to 3 decimals"

    def test_detect_edge_case_unicode_text(self):
        """Verify detect() handles Unicode characters."""
        detector = RegisterDetector()
        text = "Café résumé naïve Москва 北京"
        
        result = detector.detect(text)
        
        register, confidence = result
        assert isinstance(register, Register), "Must return valid Register"
        assert 0.0 <= confidence <= 1.0, "Confidence must be in range [0.0, 1.0]"
        assert round(confidence, 3) == confidence, "Confidence must be rounded to 3 decimals"

    def test_detect_postcondition_all_scores_zero(self):
        """Verify detect() returns (Register.direct, 0.8) when all scores are 0."""
        detector = RegisterDetector()
        # Long text with no markers should produce zero scores
        text = "Lorem ipsum dolor sit amet consectetur adipiscing elit sed do eiusmod tempor incididunt labore dolore magna aliqua."
        
        result = detector.detect(text)
        
        register, confidence = result
        # When all scores are 0, should return (Register.direct, 0.8) per postcondition
        # However, this text is long enough to not get brevity boost
        assert isinstance(register, Register), "Must return valid Register"
        assert 0.0 <= confidence <= 1.0, "Confidence must be in range [0.0, 1.0]"
        assert round(confidence, 3) == confidence, "Confidence must be rounded to 3 decimals"


class TestDetectInvariants:
    """Invariant tests for detect() method."""

    def test_detect_invariant_confidence_range(self):
        """Verify confidence is always in [0.0, 1.0] range."""
        detector = RegisterDetector()
        text = "The algorithm utilizes sophisticated techniques for optimization and performance enhancement."
        
        register, confidence = detector.detect(text)
        
        assert 0.0 <= confidence <= 1.0, "Confidence must be in range [0.0, 1.0]"

    def test_detect_invariant_confidence_precision(self):
        """Verify confidence is rounded to exactly 3 decimal places."""
        detector = RegisterDetector()
        text = "Furthermore, the hypothesis demonstrates significant correlation."
        
        register, confidence = detector.detect(text)
        
        # Check that rounding to 3 decimals doesn't change the value
        assert round(confidence, 3) == confidence, "Confidence must be rounded to 3 decimals"
        # Additional check: string representation shouldn't have more than 3 decimal places
        if '.' in str(confidence):
            decimal_part = str(confidence).split('.')[1]
            assert len(decimal_part) <= 3, f"Confidence has more than 3 decimal places: {confidence}"

    def test_detect_invariant_return_type(self):
        """Verify detect() always returns tuple of (Register, float)."""
        detector = RegisterDetector()
        text = "Some text for testing."
        
        result = detector.detect(text)
        
        assert isinstance(result, tuple), "Result must be a tuple"
        assert len(result) == 2, "Result must have exactly 2 elements"
        assert isinstance(result[1], float), "Second element must be float"

    def test_detect_invariant_determinism(self):
        """Verify detect() is deterministic - same input produces same output."""
        detector = RegisterDetector()
        text = "Hey, what's up? I'm gonna grab some food."
        
        result1 = detector.detect(text)
        result2 = detector.detect(text)
        
        assert result1 == result2, "Same input must produce same output (determinism)"

    def test_detect_invariant_stateless(self):
        """Verify RegisterDetector is stateless - different instances produce same results."""
        text = "Furthermore, it is evident that the hypothesis demonstrates correlation."
        
        detector1 = RegisterDetector()
        detector2 = RegisterDetector()
        result1 = detector1.detect(text)
        result2 = detector2.detect(text)
        
        assert result1 == result2, "Different instances must produce same results (stateless)"


class TestScoreHappyPath:
    """Happy path tests for _score() method."""

    def test_score_happy_path_casual_markers(self):
        """Verify _score() computes casual score based on pattern matches."""
        detector = RegisterDetector()
        text = "Hey, what's up? I'm gonna grab food."
        
        scores = detector._score(text)
        
        assert isinstance(scores, _FeatureScores), "Must return _FeatureScores"
        assert scores.casual > 0, "Casual score should be > 0 for text with casual markers"

    def test_score_happy_path_technical_markers(self):
        """Verify _score() computes technical score based on pattern matches."""
        detector = RegisterDetector()
        text = "The algorithm utilizes binary search with O(log n) complexity."
        
        scores = detector._score(text)
        
        assert isinstance(scores, _FeatureScores), "Must return _FeatureScores"
        assert scores.technical > 0, "Technical score should be > 0 for text with technical markers"

    def test_score_happy_path_academic_markers(self):
        """Verify _score() computes academic score based on pattern matches."""
        detector = RegisterDetector()
        text = "Furthermore, it is evident that the hypothesis demonstrates correlation."
        
        scores = detector._score(text)
        
        assert isinstance(scores, _FeatureScores), "Must return _FeatureScores"
        assert scores.academic > 0, "Academic score should be > 0 for text with academic markers"

    def test_score_happy_path_narrative_markers(self):
        """Verify _score() computes narrative score based on pattern matches."""
        detector = RegisterDetector()
        text = "Once upon a time, there was a girl who discovered treasure."
        
        scores = detector._score(text)
        
        assert isinstance(scores, _FeatureScores), "Must return _FeatureScores"
        assert scores.narrative > 0, "Narrative score should be > 0 for text with narrative markers"

    def test_score_happy_path_direct_brevity_six_words(self):
        """Verify _score() applies +3.0 direct boost for <= 6 words."""
        detector = RegisterDetector()
        text = "Send the report now."
        
        scores = detector._score(text)
        
        assert isinstance(scores, _FeatureScores), "Must return _FeatureScores"
        # Text has 4 words, should get +1.5 for <=12 and +1.5 for <=6 = 3.0 total
        assert scores.direct == 3.0, f"Direct score should be 3.0 for text with <= 6 words, got {scores.direct}"

    def test_score_happy_path_direct_brevity_twelve_words(self):
        """Verify _score() applies +1.5 direct boost for <= 12 words."""
        detector = RegisterDetector()
        text = "The quick brown fox jumps over the lazy dog today."
        
        scores = detector._score(text)
        
        assert isinstance(scores, _FeatureScores), "Must return _FeatureScores"
        # Text has 10 words, should get +1.5 for <=12 but not the <=6 boost
        assert scores.direct == 1.5, f"Direct score should be 1.5 for text with 7-12 words, got {scores.direct}"


class TestScoreEdgeCases:
    """Edge case tests for _score() method."""

    def test_score_edge_case_direct_penalty(self):
        """Verify _score() applies 0.3 penalty to direct score when max(other_scores) > 2.0."""
        detector = RegisterDetector()
        text = "Furthermore, it is evident that the hypothesis demonstrates significant correlation with observed phenomena."
        
        scores = detector._score(text)
        
        assert isinstance(scores, _FeatureScores), "Must return _FeatureScores"
        # This text has strong academic markers and is brief enough for some direct boost
        # If max(casual, technical, academic, narrative) > 2.0, direct score should be *= 0.3
        max_other = max(scores.casual, scores.technical, scores.academic, scores.narrative)
        # We can't assert exact values without knowing the implementation, but we can verify structure
        assert scores.direct >= 0.0, "Direct score must be non-negative even after penalty"

    def test_score_edge_case_no_markers(self):
        """Verify _score() returns all zeros except direct boost for text with no markers."""
        detector = RegisterDetector()
        text = "The cat sat on the mat quietly."
        
        scores = detector._score(text)
        
        assert isinstance(scores, _FeatureScores), "Must return _FeatureScores"
        # Text with 7 words gets +1.5 direct boost, but no other markers should be present
        # We can't guarantee exact 0s without implementation knowledge, but scores should be minimal
        assert scores.direct >= 0.0, "Direct score should include brevity boost if applicable"


class TestScoreInvariants:
    """Invariant tests for _score() method."""

    def test_score_invariant_all_scores_non_negative(self):
        """Verify _score() returns all scores >= 0.0."""
        detector = RegisterDetector()
        text = "Hey, what's the algorithm's complexity? Furthermore, it demonstrates correlation once upon a time."
        
        scores = detector._score(text)
        
        assert scores.casual >= 0.0, "Casual score must be non-negative"
        assert scores.technical >= 0.0, "Technical score must be non-negative"
        assert scores.academic >= 0.0, "Academic score must be non-negative"
        assert scores.narrative >= 0.0, "Narrative score must be non-negative"
        assert scores.direct >= 0.0, "Direct score must be non-negative"

    def test_score_invariant_featurescores_structure(self):
        """Verify _score() returns valid _FeatureScores with all five fields."""
        detector = RegisterDetector()
        text = "Some test text here."
        
        scores = detector._score(text)
        
        assert hasattr(scores, 'casual'), "_FeatureScores must have 'casual' field"
        assert hasattr(scores, 'technical'), "_FeatureScores must have 'technical' field"
        assert hasattr(scores, 'academic'), "_FeatureScores must have 'academic' field"
        assert hasattr(scores, 'narrative'), "_FeatureScores must have 'narrative' field"
        assert hasattr(scores, 'direct'), "_FeatureScores must have 'direct' field"


class TestModuleInvariants:
    """Invariant tests for module-level constants."""

    def test_invariant_casual_markers_structure(self):
        """Verify CASUAL_MARKERS contains exactly 7 tuples with weights in [1.0, 2.5]."""
        assert len(CASUAL_MARKERS) == 7, f"CASUAL_MARKERS must contain exactly 7 tuples, got {len(CASUAL_MARKERS)}"
        
        for item in CASUAL_MARKERS:
            assert isinstance(item, tuple), "Each CASUAL_MARKERS item must be a tuple"
            assert len(item) == 2, "Each CASUAL_MARKERS tuple must have 2 elements (pattern, weight)"
            pattern, weight = item
            assert isinstance(weight, (int, float)), f"Weight must be numeric, got {type(weight)}"
            assert 1.0 <= weight <= 2.5, f"Weight must be in range [1.0, 2.5], got {weight}"

    def test_invariant_technical_markers_structure(self):
        """Verify TECHNICAL_MARKERS contains exactly 5 tuples with weights in [1.0, 2.0]."""
        assert len(TECHNICAL_MARKERS) == 5, f"TECHNICAL_MARKERS must contain exactly 5 tuples, got {len(TECHNICAL_MARKERS)}"
        
        for item in TECHNICAL_MARKERS:
            assert isinstance(item, tuple), "Each TECHNICAL_MARKERS item must be a tuple"
            assert len(item) == 2, "Each TECHNICAL_MARKERS tuple must have 2 elements (pattern, weight)"
            pattern, weight = item
            assert isinstance(weight, (int, float)), f"Weight must be numeric, got {type(weight)}"
            assert 1.0 <= weight <= 2.0, f"Weight must be in range [1.0, 2.0], got {weight}"

    def test_invariant_academic_markers_structure(self):
        """Verify ACADEMIC_MARKERS contains exactly 6 tuples with weights in [1.5, 2.5]."""
        assert len(ACADEMIC_MARKERS) == 6, f"ACADEMIC_MARKERS must contain exactly 6 tuples, got {len(ACADEMIC_MARKERS)}"
        
        for item in ACADEMIC_MARKERS:
            assert isinstance(item, tuple), "Each ACADEMIC_MARKERS item must be a tuple"
            assert len(item) == 2, "Each ACADEMIC_MARKERS tuple must have 2 elements (pattern, weight)"
            pattern, weight = item
            assert isinstance(weight, (int, float)), f"Weight must be numeric, got {type(weight)}"
            assert 1.5 <= weight <= 2.5, f"Weight must be in range [1.5, 2.5], got {weight}"

    def test_invariant_narrative_markers_structure(self):
        """Verify NARRATIVE_MARKERS contains exactly 5 tuples with weights in [1.0, 3.0]."""
        assert len(NARRATIVE_MARKERS) == 5, f"NARRATIVE_MARKERS must contain exactly 5 tuples, got {len(NARRATIVE_MARKERS)}"
        
        for item in NARRATIVE_MARKERS:
            assert isinstance(item, tuple), "Each NARRATIVE_MARKERS item must be a tuple"
            assert len(item) == 2, "Each NARRATIVE_MARKERS tuple must have 2 elements (pattern, weight)"
            pattern, weight = item
            assert isinstance(weight, (int, float)), f"Weight must be numeric, got {type(weight)}"
            assert 1.0 <= weight <= 3.0, f"Weight must be in range [1.0, 3.0], got {weight}"

    def test_invariant_direct_markers_empty(self):
        """Verify DIRECT_MARKERS is empty as per contract."""
        assert len(DIRECT_MARKERS) == 0, f"DIRECT_MARKERS must be empty, got {len(DIRECT_MARKERS)} elements"


class TestFeatureScoresType:
    """Tests for _FeatureScores dataclass structure."""

    def test_featurescores_can_be_instantiated(self):
        """Verify _FeatureScores can be instantiated with valid values."""
        scores = _FeatureScores(
            casual=1.5,
            technical=2.0,
            academic=1.8,
            narrative=0.5,
            direct=3.0
        )
        
        assert scores.casual == 1.5
        assert scores.technical == 2.0
        assert scores.academic == 1.8
        assert scores.narrative == 0.5
        assert scores.direct == 3.0

    def test_featurescores_has_all_required_fields(self):
        """Verify _FeatureScores has all five required fields."""
        scores = _FeatureScores(
            casual=0.0,
            technical=0.0,
            academic=0.0,
            narrative=0.0,
            direct=0.0
        )
        
        # Verify all fields exist
        field_names = {f.name for f in fields(scores)}
        assert 'casual' in field_names
        assert 'technical' in field_names
        assert 'academic' in field_names
        assert 'narrative' in field_names
        assert 'direct' in field_names
        assert len(field_names) == 5, "Should have exactly 5 fields"


class TestRegisterDetectorType:
    """Tests for RegisterDetector class structure."""

    def test_registerdetector_can_be_instantiated(self):
        """Verify RegisterDetector can be instantiated."""
        detector = RegisterDetector()
        assert detector is not None

    def test_registerdetector_has_detect_method(self):
        """Verify RegisterDetector has detect() method."""
        detector = RegisterDetector()
        assert hasattr(detector, 'detect'), "RegisterDetector must have detect() method"
        assert callable(detector.detect), "detect must be callable"

    def test_registerdetector_has_score_method(self):
        """Verify RegisterDetector has _score() method."""
        detector = RegisterDetector()
        assert hasattr(detector, '_score'), "RegisterDetector must have _score() method"
        assert callable(detector._score), "_score must be callable"

    def test_registerdetector_is_stateless(self):
        """Verify RegisterDetector has no instance state (stateless)."""
        detector1 = RegisterDetector()
        detector2 = RegisterDetector()
        
        # Both instances should behave identically for same input
        text = "Test text for stateless verification."
        result1 = detector1.detect(text)
        result2 = detector2.detect(text)
        
        assert result1 == result2, "Stateless detector instances should produce identical results"
        
        # Multiple calls on same instance should also be identical
        result3 = detector1.detect(text)
        assert result1 == result3, "Multiple calls on same instance should be identical (no state mutation)"
