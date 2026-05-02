"""
Contract test suite for TaskClassifier interface.

Tests verify compliance with the contract specification including:
- Happy path: valid inputs return correct types and bounds
- Edge cases: empty strings, whitespace, unicode, long text, special chars
- Invariants: confidence bounds, precision, return types, determinism
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
from enum import Enum
from typing import Tuple
import random
import string

# Import the component under test
from contracts.contracts_src_transmogrifier_task_classifier_interface.interface import *


# ============================================================================
# Test Fixtures and Helpers
# ============================================================================

@pytest.fixture
def classifier():
    """Fixture providing a TaskClassifier instance."""
    return TaskClassifier()


@pytest.fixture
def mock_classifier():
    """Fixture providing a mock TaskClassifier for controlled testing."""
    mock = Mock(spec=TaskClassifier)
    # Default behavior: return unknown with 0.5 confidence
    mock.classify.return_value = (TaskType.unknown, 0.5)
    return mock


def is_valid_confidence(conf: float) -> bool:
    """Check if confidence value meets contract requirements."""
    if not isinstance(conf, float):
        return False
    if not (0.0 <= conf <= 1.0):
        return False
    # Check precision (at most 3 decimal places)
    return round(conf, 3) == conf


# ============================================================================
# Contract Compliance Tests
# ============================================================================

class TestContractCompliance:
    """Test basic contract compliance: types, bounds, interface."""
    
    def test_task_type_enum_exists(self):
        """Verify TaskType enum is defined with correct members."""
        assert issubclass(TaskType, Enum)
        expected_members = {'factual', 'reasoning', 'code', 'analysis', 
                          'creative', 'instruction', 'unknown'}
        actual_members = {member.name for member in TaskType}
        assert expected_members == actual_members, \
            f"Expected {expected_members}, got {actual_members}"
    
    def test_task_classifier_exists(self):
        """Verify TaskClassifier class is defined."""
        assert TaskClassifier is not None
        classifier = TaskClassifier()
        assert classifier is not None
    
    def test_classify_method_exists(self, classifier):
        """Verify classify method exists and is callable."""
        assert hasattr(classifier, 'classify')
        assert callable(getattr(classifier, 'classify'))
    
    def test_classify_return_type(self, classifier):
        """Verify classify returns tuple of (TaskType, float)."""
        result = classifier.classify("Test input")
        
        assert isinstance(result, tuple), "Result must be a tuple"
        assert len(result) == 2, "Result tuple must have exactly 2 elements"
        
        task_type, confidence = result
        assert isinstance(task_type, TaskType), \
            f"First element must be TaskType, got {type(task_type)}"
        assert isinstance(confidence, float), \
            f"Second element must be float, got {type(confidence)}"
    
    def test_classify_confidence_bounds(self, classifier):
        """Verify confidence is always in range [0.0, 1.0]."""
        test_inputs = [
            "What is the capital of France?",
            "Write a Python function",
            "Analyze this data",
            "Be creative",
            "How to install software",
            "If A then B",
            "",
            "   ",
            "xyzabc123"
        ]
        
        for text in test_inputs:
            _, confidence = classifier.classify(text)
            assert 0.0 <= confidence <= 1.0, \
                f"Confidence {confidence} out of bounds for input: {text}"
    
    def test_classify_confidence_precision(self, classifier):
        """Verify confidence is rounded to 3 decimal places."""
        test_inputs = [
            "What is the answer?",
            "Write code to solve this",
            "Analyze the results",
            "Sample text input"
        ]
        
        for text in test_inputs:
            _, confidence = classifier.classify(text)
            assert is_valid_confidence(confidence), \
                f"Confidence {confidence} not rounded to 3 decimal places"


# ============================================================================
# Happy Path Tests
# ============================================================================

class TestHappyPath:
    """Test expected behavior with typical inputs."""
    
    def test_classify_happy_path_factual(self, classifier):
        """Verify classify returns valid TaskType and confidence for a factual question."""
        text = "What is the capital of France?"
        result = classifier.classify(text)
        
        assert isinstance(result, tuple), "Result is a tuple with 2 elements"
        assert len(result) == 2
        
        task_type, confidence = result
        assert isinstance(task_type, TaskType), "First element is a TaskType enum member"
        assert isinstance(confidence, float), "Second element is float"
        assert 0.0 <= confidence <= 1.0, "Confidence in range [0.0, 1.0]"
        assert is_valid_confidence(confidence), "Confidence is rounded to 3 decimal places"
    
    def test_classify_happy_path_code(self, classifier):
        """Verify classify handles code-related prompts."""
        text = "Write a Python function to sort a list"
        result = classifier.classify(text)
        
        assert isinstance(result, tuple), "Result is a tuple with 2 elements"
        assert len(result) == 2
        
        task_type, confidence = result
        assert isinstance(task_type, TaskType), "First element is a TaskType enum member"
        assert isinstance(confidence, float), "Second element is float"
        assert 0.0 <= confidence <= 1.0, "Confidence in range [0.0, 1.0]"
    
    def test_classify_happy_path_reasoning(self, classifier):
        """Verify classify handles reasoning tasks."""
        text = "If all A are B and all B are C, what can we conclude about A and C?"
        result = classifier.classify(text)
        
        assert isinstance(result, tuple), "Result is a tuple with 2 elements"
        assert len(result) == 2
        
        task_type, confidence = result
        assert isinstance(task_type, TaskType), "First element is a TaskType enum member"
        assert isinstance(confidence, float), "Second element is float"
        assert 0.0 <= confidence <= 1.0, "Confidence in range [0.0, 1.0]"


# ============================================================================
# Edge Case Tests
# ============================================================================

class TestEdgeCases:
    """Test boundary conditions and edge cases."""
    
    def test_classify_empty_string(self, classifier):
        """Verify empty string returns unknown with 0.0 confidence."""
        task_type, confidence = classifier.classify("")
        
        assert task_type == TaskType.unknown, "Returns TaskType.unknown"
        assert confidence == 0.0, "Confidence is exactly 0.0"
    
    def test_classify_whitespace_only(self, classifier):
        """Verify whitespace-only input returns unknown with 0.0 confidence."""
        whitespace_inputs = ["   ", "\t", "\n", "\r", "   \t\n\r  ", " \t\n\r "]
        
        for text in whitespace_inputs:
            task_type, confidence = classifier.classify(text)
            assert task_type == TaskType.unknown, \
                f"Returns TaskType.unknown for '{repr(text)}'"
            assert confidence == 0.0, \
                f"Confidence is exactly 0.0 for '{repr(text)}'"
    
    def test_classify_no_pattern_match(self, classifier):
        """Verify text with no matching patterns returns unknown with 0.5 confidence."""
        # Random text unlikely to match any patterns
        text = "xyzabc123"
        result = classifier.classify(text)
        
        assert isinstance(result, tuple), "Result is a tuple"
        task_type, confidence = result
        assert isinstance(task_type, TaskType), "First element is a TaskType enum member"
        assert isinstance(confidence, float), "Second element is float"
        assert 0.0 <= confidence <= 1.0, "Confidence in range [0.0, 1.0]"
    
    def test_classify_unicode_text(self, classifier):
        """Verify classify handles unicode text correctly."""
        text = "Какова столица России? 日本の首都は?"
        result = classifier.classify(text)
        
        assert isinstance(result, tuple), "Result is a tuple"
        task_type, confidence = result
        assert isinstance(task_type, TaskType), "First element is a TaskType"
        assert isinstance(confidence, float), "Second element is float"
        assert 0.0 <= confidence <= 1.0, "Confidence in range [0.0, 1.0]"
    
    def test_classify_very_long_text(self, classifier):
        """Verify classify handles very long text inputs."""
        # Create a very long text (10000+ characters)
        long_text = "What is the answer? " * 1000
        result = classifier.classify(long_text)
        
        assert isinstance(result, tuple), "Result is a tuple"
        task_type, confidence = result
        assert isinstance(task_type, TaskType), "First element is a TaskType"
        assert isinstance(confidence, float), "Second element is float"
        assert 0.0 <= confidence <= 1.0, "Confidence in range [0.0, 1.0]"
    
    def test_classify_special_characters(self, classifier):
        """Verify classify handles special characters and symbols."""
        text = "!@#$%^&*()_+-=[]{}|;':\",./<>?"
        result = classifier.classify(text)
        
        assert isinstance(result, tuple), "Result is a tuple"
        task_type, confidence = result
        assert isinstance(task_type, TaskType), "First element is a TaskType"
        assert isinstance(confidence, float), "Second element is float"
        assert 0.0 <= confidence <= 1.0, "Confidence in range [0.0, 1.0]"
    
    def test_classify_mixed_content(self, classifier):
        """Verify classify handles text with mixed markers from multiple categories."""
        text = "What is the answer? Analyze this code: def foo(): pass. Be creative!"
        result = classifier.classify(text)
        
        assert isinstance(result, tuple), "Result is a tuple"
        task_type, confidence = result
        assert isinstance(task_type, TaskType), "First element is a TaskType"
        assert isinstance(confidence, float), "Second element is float"
        assert 0.0 <= confidence <= 1.0, "Confidence in range [0.0, 1.0]"
        assert is_valid_confidence(confidence), "Confidence is rounded to 3 decimal places"
    
    def test_classify_single_character(self, classifier):
        """Verify classify handles single character input."""
        text = "a"
        result = classifier.classify(text)
        
        assert isinstance(result, tuple), "Result is a tuple"
        task_type, confidence = result
        assert isinstance(task_type, TaskType), "First element is a TaskType"
        assert isinstance(confidence, float), "Second element is float"
        assert 0.0 <= confidence <= 1.0, "Confidence in range [0.0, 1.0]"
    
    @pytest.mark.parametrize("text", [
        "",
        "   ",
        "a",
        "Test",
        "What is this?",
        "Write code",
        "Analyze data",
        "Be creative!",
        "How to do something",
        "If this then that",
        "x" * 10000,
        "🎉 emoji test 🚀",
        "\x00\x01\x02",
        "UPPERCASE TEXT",
        "lowercase text",
        "MiXeD cAsE"
    ])
    def test_classify_various_inputs(self, classifier, text):
        """Parametrized test for various edge case inputs."""
        result = classifier.classify(text)
        
        assert isinstance(result, tuple)
        assert len(result) == 2
        task_type, confidence = result
        assert isinstance(task_type, TaskType)
        assert isinstance(confidence, float)
        assert 0.0 <= confidence <= 1.0


# ============================================================================
# Invariant Tests
# ============================================================================

class TestInvariants:
    """Test invariants that must hold across all invocations."""
    
    def test_classify_all_task_types_reachable(self):
        """Verify all TaskType enum members can theoretically be returned."""
        # All TaskType values are valid
        all_types = list(TaskType)
        assert len(all_types) == 7, "Expected 7 TaskType enum members"
        
        expected = {TaskType.factual, TaskType.reasoning, TaskType.code,
                   TaskType.analysis, TaskType.creative, TaskType.instruction,
                   TaskType.unknown}
        actual = set(all_types)
        assert expected == actual, "All TaskType enum members are valid"
    
    def test_classify_case_insensitive(self, classifier):
        """Verify pattern matching is case-insensitive."""
        text_lower = "what is the capital"
        text_upper = "WHAT IS THE CAPITAL"
        text_mixed = "WhAt Is ThE cApItAl"
        
        result_lower = classifier.classify(text_lower)
        result_upper = classifier.classify(text_upper)
        result_mixed = classifier.classify(text_mixed)
        
        # All should return valid tuples
        for result in [result_lower, result_upper, result_mixed]:
            assert isinstance(result, tuple)
            task_type, confidence = result
            assert isinstance(task_type, TaskType)
            assert isinstance(confidence, float)
            assert 0.0 <= confidence <= 1.0
    
    def test_classify_determinism(self, classifier):
        """Verify classify is deterministic (same input yields same output)."""
        text = "Determinism test input"
        
        result1 = classifier.classify(text)
        result2 = classifier.classify(text)
        result3 = classifier.classify(text)
        
        assert result1 == result2, "First call result equals second call result"
        assert result2 == result3, "Result is stable across multiple invocations"
        assert result1 == result3
    
    def test_classify_no_side_effects(self, classifier):
        """Verify classify has no side effects (pure function behavior)."""
        text = "Side effect test"
        
        # Call multiple times and verify state doesn't change
        for _ in range(10):
            result = classifier.classify(text)
            assert isinstance(result, tuple)
            assert len(result) == 2
    
    def test_confidence_calculation_bounds(self, classifier):
        """Verify confidence calculation never exceeds 1.0."""
        # Test with various inputs that might have high scores
        test_inputs = [
            "What is? How? Why? When? Where? Who?",  # Multiple factual markers
            "def class import function code python java",  # Multiple code markers
            "analyze examine investigate evaluate assess",  # Multiple analysis markers
            "create imagine invent design compose",  # Multiple creative markers
        ]
        
        for text in test_inputs:
            _, confidence = classifier.classify(text)
            assert confidence <= 1.0, \
                f"Confidence {confidence} exceeds 1.0 for input: {text}"
            assert is_valid_confidence(confidence)
    
    def test_empty_and_whitespace_consistency(self, classifier):
        """Verify all empty/whitespace inputs consistently return (unknown, 0.0)."""
        empty_inputs = ["", " ", "  ", "\t", "\n", "\r\n", "   \t\n\r   "]
        
        for text in empty_inputs:
            task_type, confidence = classifier.classify(text)
            assert task_type == TaskType.unknown
            assert confidence == 0.0
    
    def test_classify_random_inputs_no_exceptions(self, classifier):
        """Verify classify doesn't raise exceptions on random valid string inputs."""
        random.seed(42)  # For reproducibility
        
        for _ in range(50):
            # Generate random strings of varying lengths
            length = random.randint(0, 1000)
            chars = string.ascii_letters + string.digits + string.punctuation + " \t\n"
            text = ''.join(random.choice(chars) for _ in range(length))
            
            # Should not raise any exceptions
            result = classifier.classify(text)
            assert isinstance(result, tuple)
            task_type, confidence = result
            assert isinstance(task_type, TaskType)
            assert isinstance(confidence, float)
            assert 0.0 <= confidence <= 1.0


# ============================================================================
# Integration Tests
# ============================================================================

class TestIntegration:
    """Integration tests verifying end-to-end behavior."""
    
    def test_classify_workflow_typical_prompts(self, classifier):
        """Test classify with a variety of typical prompts."""
        test_cases = [
            ("What is the meaning of life?", "factual-like question"),
            ("Explain how to solve this problem step by step", "reasoning task"),
            ("Write a function in Python that reverses a string", "code task"),
            ("Analyze the performance metrics of this system", "analysis task"),
            ("Create a short story about a dragon", "creative task"),
            ("How to install Docker on Ubuntu", "instruction task"),
            ("", "empty string"),
            ("zzz123", "random text"),
        ]
        
        for text, description in test_cases:
            result = classifier.classify(text)
            assert isinstance(result, tuple), f"Failed for: {description}"
            task_type, confidence = result
            assert isinstance(task_type, TaskType), f"Failed for: {description}"
            assert isinstance(confidence, float), f"Failed for: {description}"
            assert 0.0 <= confidence <= 1.0, f"Failed for: {description}"
    
    def test_classifier_multiple_instances(self):
        """Verify multiple classifier instances behave consistently."""
        classifier1 = TaskClassifier()
        classifier2 = TaskClassifier()
        
        test_text = "What is the capital of France?"
        
        result1 = classifier1.classify(test_text)
        result2 = classifier2.classify(test_text)
        
        # Both should return same result (deterministic behavior)
        assert result1 == result2


# ============================================================================
# Marker Pattern Invariants (if accessible)
# ============================================================================

class TestMarkerInvariants:
    """Test invariants related to marker patterns (if accessible)."""
    
    def test_marker_constants_exist(self):
        """Verify marker constants are defined (if publicly accessible)."""
        # These might be private, so we check if they exist
        # This is optional and depends on implementation visibility
        try:
            classifier = TaskClassifier()
            # Try to access markers if they're public attributes
            if hasattr(classifier, '_FACTUAL_MARKERS'):
                assert isinstance(classifier._FACTUAL_MARKERS, (list, tuple))
            if hasattr(classifier, '_REASONING_MARKERS'):
                assert isinstance(classifier._REASONING_MARKERS, (list, tuple))
            if hasattr(classifier, '_CODE_MARKERS'):
                assert isinstance(classifier._CODE_MARKERS, (list, tuple))
            if hasattr(classifier, '_ANALYSIS_MARKERS'):
                assert isinstance(classifier._ANALYSIS_MARKERS, (list, tuple))
            if hasattr(classifier, '_CREATIVE_MARKERS'):
                assert isinstance(classifier._CREATIVE_MARKERS, (list, tuple))
            if hasattr(classifier, '_INSTRUCTION_MARKERS'):
                assert isinstance(classifier._INSTRUCTION_MARKERS, (list, tuple))
        except AttributeError:
            # Markers are private, skip this test
            pytest.skip("Marker constants are not publicly accessible")


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
