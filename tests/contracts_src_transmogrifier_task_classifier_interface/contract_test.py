"""
Contract tests for TaskClassifier interface.

Tests cover:
1. Contract tests - type signatures and return value structure
2. Happy path tests - one per TaskType category
3. Edge cases - empty/whitespace, no matches, unicode, long text, case sensitivity
4. Invariant tests - determinism, confidence bounds, enum membership
5. Boundary/robustness tests - special characters, mixed scripts, ambiguous inputs
"""

import pytest
import re
from unittest.mock import Mock, patch
from contracts.src_transmogrifier_task_classifier.interface import TaskType, TaskClassifier


class TestClassifyHappyPath:
    """Happy path tests for each TaskType category."""
    
    def test_classify_factual_happy_path(self):
        """Happy path: Classify a clear factual question."""
        classifier = TaskClassifier()
        text = "What is the capital of France?"
        
        result = classifier.classify(text)
        
        assert isinstance(result, tuple), "result is a tuple"
        assert len(result) == 2, "result is a tuple of length 2"
        assert result[0] is TaskType.factual, "result[0] is TaskType.factual"
        assert 0.0 <= result[1] <= 1.0, "0.0 <= result[1] <= 1.0"
        assert isinstance(result[1], float), "isinstance(result[1], float)"
    
    def test_classify_reasoning_happy_path(self):
        """Happy path: Classify a reasoning task."""
        classifier = TaskClassifier()
        text = "Why does the moon orbit the Earth? Explain the underlying physics."
        
        result = classifier.classify(text)
        
        assert isinstance(result, tuple), "result is a tuple"
        assert len(result) == 2, "result is a tuple of length 2"
        assert result[0] is TaskType.reasoning, "result[0] is TaskType.reasoning"
        assert 0.0 <= result[1] <= 1.0, "0.0 <= result[1] <= 1.0"
    
    def test_classify_code_happy_path(self):
        """Happy path: Classify a code-related task."""
        classifier = TaskClassifier()
        text = "Write a Python function to calculate fibonacci numbers"
        
        result = classifier.classify(text)
        
        assert isinstance(result, tuple), "result is a tuple"
        assert len(result) == 2, "result is a tuple of length 2"
        assert result[0] is TaskType.code, "result[0] is TaskType.code"
        assert 0.0 <= result[1] <= 1.0, "0.0 <= result[1] <= 1.0"
    
    def test_classify_analysis_happy_path(self):
        """Happy path: Classify an analytical task."""
        classifier = TaskClassifier()
        text = "Analyze the differences between REST and GraphQL APIs"
        
        result = classifier.classify(text)
        
        assert isinstance(result, tuple), "result is a tuple"
        assert len(result) == 2, "result is a tuple of length 2"
        assert result[0] is TaskType.analysis, "result[0] is TaskType.analysis"
        assert 0.0 <= result[1] <= 1.0, "0.0 <= result[1] <= 1.0"
    
    def test_classify_creative_happy_path(self):
        """Happy path: Classify a creative task."""
        classifier = TaskClassifier()
        text = "Write a short story about a time traveler"
        
        result = classifier.classify(text)
        
        assert isinstance(result, tuple), "result is a tuple"
        assert len(result) == 2, "result is a tuple of length 2"
        assert result[0] is TaskType.creative, "result[0] is TaskType.creative"
        assert 0.0 <= result[1] <= 1.0, "0.0 <= result[1] <= 1.0"
    
    def test_classify_instruction_happy_path(self):
        """Happy path: Classify an instructional task."""
        classifier = TaskClassifier()
        text = "How to bake a chocolate cake step by step"
        
        result = classifier.classify(text)
        
        assert isinstance(result, tuple), "result is a tuple"
        assert len(result) == 2, "result is a tuple of length 2"
        assert result[0] is TaskType.instruction, "result[0] is TaskType.instruction"
        assert 0.0 <= result[1] <= 1.0, "0.0 <= result[1] <= 1.0"


class TestClassifyEdgeCases:
    """Edge case tests for boundary conditions."""
    
    def test_classify_empty_string(self):
        """Edge case: Empty string input returns unknown with 0.0 confidence."""
        classifier = TaskClassifier()
        text = ""
        
        result = classifier.classify(text)
        
        assert result == (TaskType.unknown, 0.0), "result == (TaskType.unknown, 0.0)"
    
    def test_classify_whitespace_only_spaces(self):
        """Edge case: Whitespace-only input (spaces) returns unknown with 0.0 confidence."""
        classifier = TaskClassifier()
        text = "   "
        
        result = classifier.classify(text)
        
        assert result == (TaskType.unknown, 0.0), "result == (TaskType.unknown, 0.0)"
    
    def test_classify_whitespace_only_tabs(self):
        """Edge case: Whitespace-only input (tabs) returns unknown with 0.0 confidence."""
        classifier = TaskClassifier()
        text = "\t\t\t"
        
        result = classifier.classify(text)
        
        assert result == (TaskType.unknown, 0.0), "result == (TaskType.unknown, 0.0)"
    
    def test_classify_whitespace_only_newlines(self):
        """Edge case: Whitespace-only input (newlines) returns unknown with 0.0 confidence."""
        classifier = TaskClassifier()
        text = "\n\n\n"
        
        result = classifier.classify(text)
        
        assert result == (TaskType.unknown, 0.0), "result == (TaskType.unknown, 0.0)"
    
    def test_classify_whitespace_mixed(self):
        """Edge case: Mixed whitespace returns unknown with 0.0 confidence."""
        classifier = TaskClassifier()
        text = " \t\n \r "
        
        result = classifier.classify(text)
        
        assert result == (TaskType.unknown, 0.0), "result == (TaskType.unknown, 0.0)"
    
    def test_classify_no_patterns_match(self):
        """Edge case: Text with no matching patterns returns unknown with 0.5 confidence."""
        classifier = TaskClassifier()
        text = "asdfghjkl qwertyuiop zxcvbnm"
        
        result = classifier.classify(text)
        
        assert result[0] == TaskType.unknown, "result[0] == TaskType.unknown"
        assert result[1] == 0.5, "result[1] == 0.5"
    
    def test_classify_confidence_rounded_to_3_decimals(self):
        """Edge case: Verify confidence is rounded to 3 decimal places."""
        classifier = TaskClassifier()
        text = "What is Python? Write a function."
        
        result = classifier.classify(text)
        
        # Check that confidence has at most 3 decimal places
        confidence_str = f"{result[1]:.10f}"  # Get full precision
        if '.' in confidence_str:
            decimal_part = confidence_str.split('.')[1].rstrip('0')
            assert len(decimal_part) <= 3 or result[1] in [0.0, 1.0], \
                f"Confidence {result[1]} has more than 3 decimal places: {decimal_part}"
    
    def test_classify_confidence_bounds(self):
        """Edge case: Confidence is always in [0.0, 1.0]."""
        classifier = TaskClassifier()
        text = "Explain how to write code that analyzes creative stories with factual instructions"
        
        result = classifier.classify(text)
        
        assert 0.0 <= result[1] <= 1.0, "0.0 <= result[1] <= 1.0"
    
    def test_classify_unicode_text(self):
        """Edge case: Handle unicode characters."""
        classifier = TaskClassifier()
        text = "What is the capital of France? 👋🌍"
        
        result = classifier.classify(text)
        
        assert isinstance(result, tuple), "isinstance(result, tuple)"
        assert len(result) == 2, "len(result) == 2"
        assert isinstance(result[0], TaskType), "isinstance(result[0], TaskType)"
        assert 0.0 <= result[1] <= 1.0, "0.0 <= result[1] <= 1.0"
    
    def test_classify_special_characters(self):
        """Edge case: Handle special characters."""
        classifier = TaskClassifier()
        text = "@#$%^&*()"
        
        result = classifier.classify(text)
        
        assert isinstance(result, tuple), "isinstance(result, tuple)"
        assert len(result) == 2, "len(result) == 2"
        assert isinstance(result[0], TaskType), "isinstance(result[0], TaskType)"
        assert 0.0 <= result[1] <= 1.0, "0.0 <= result[1] <= 1.0"
    
    def test_classify_very_long_text(self):
        """Edge case: Handle very long text (>10KB)."""
        classifier = TaskClassifier()
        text = "What is the meaning of life? " * 1000
        
        result = classifier.classify(text)
        
        assert isinstance(result, tuple), "isinstance(result, tuple)"
        assert len(result) == 2, "len(result) == 2"
        assert isinstance(result[0], TaskType), "isinstance(result[0], TaskType)"
        assert 0.0 <= result[1] <= 1.0, "0.0 <= result[1] <= 1.0"
    
    def test_classify_case_insensitive(self):
        """Edge case: Verify case-insensitive pattern matching."""
        classifier = TaskClassifier()
        text = "WHAT IS THE CAPITAL OF FRANCE?"
        
        result = classifier.classify(text)
        
        assert result[0] == TaskType.factual, "result[0] == TaskType.factual"
    
    def test_classify_ambiguous_multi_category(self):
        """Edge case: Text matching multiple categories."""
        classifier = TaskClassifier()
        text = "What is a sorting algorithm? Write code to implement it and explain why it works."
        
        result = classifier.classify(text)
        
        assert isinstance(result[0], TaskType), "isinstance(result[0], TaskType)"
        assert 0.0 <= result[1] <= 1.0, "0.0 <= result[1] <= 1.0"
    
    def test_classify_single_word(self):
        """Edge case: Single word input."""
        classifier = TaskClassifier()
        text = "function"
        
        result = classifier.classify(text)
        
        assert isinstance(result, tuple), "isinstance(result, tuple)"
        assert len(result) == 2, "len(result) == 2"
        assert isinstance(result[0], TaskType), "isinstance(result[0], TaskType)"
        assert 0.0 <= result[1] <= 1.0, "0.0 <= result[1] <= 1.0"
    
    def test_classify_numbers_only(self):
        """Edge case: Text with only numbers."""
        classifier = TaskClassifier()
        text = "123456789"
        
        result = classifier.classify(text)
        
        assert isinstance(result, tuple), "isinstance(result, tuple)"
        assert len(result) == 2, "len(result) == 2"
        assert isinstance(result[0], TaskType), "isinstance(result[0], TaskType)"
        assert 0.0 <= result[1] <= 1.0, "0.0 <= result[1] <= 1.0"
    
    def test_classify_mixed_script(self):
        """Edge case: Mixed script text (Latin + Cyrillic)."""
        classifier = TaskClassifier()
        text = "What is привет мир?"
        
        result = classifier.classify(text)
        
        assert isinstance(result, tuple), "isinstance(result, tuple)"
        assert len(result) == 2, "len(result) == 2"
        assert isinstance(result[0], TaskType), "isinstance(result[0], TaskType)"
        assert 0.0 <= result[1] <= 1.0, "0.0 <= result[1] <= 1.0"


class TestClassifyInvariants:
    """Invariant tests for contract guarantees."""
    
    def test_classify_determinism(self):
        """Invariant: Same input produces same output (determinism)."""
        classifier = TaskClassifier()
        text = "Write a Python function"
        
        # Call multiple times
        result1 = classifier.classify(text)
        result2 = classifier.classify(text)
        result3 = classifier.classify(text)
        
        assert result1 == result2 == result3, "Multiple calls return same result"
    
    def test_classify_return_type_structure(self):
        """Invariant: Return type is always tuple[TaskType, float]."""
        classifier = TaskClassifier()
        text = "Random text input"
        
        result = classifier.classify(text)
        
        assert isinstance(result, tuple), "isinstance(result, tuple)"
        assert len(result) == 2, "len(result) == 2"
        assert isinstance(result[0], TaskType), "isinstance(result[0], TaskType)"
        assert isinstance(result[1], float), "isinstance(result[1], float)"
    
    def test_classify_enum_membership(self):
        """Invariant: Returned TaskType is always valid enum member."""
        classifier = TaskClassifier()
        text = "Any input text"
        
        result = classifier.classify(text)
        
        assert result[0] in TaskType, "result[0] in TaskType"
        # Verify it's one of the defined enum values
        valid_types = [TaskType.factual, TaskType.reasoning, TaskType.code, 
                      TaskType.analysis, TaskType.creative, TaskType.instruction, 
                      TaskType.unknown]
        assert result[0] in valid_types, f"TaskType {result[0]} is a valid enum member"
    
    def test_classify_confidence_always_in_range(self):
        """Invariant: Confidence is always in [0.0, 1.0] for any input."""
        classifier = TaskClassifier()
        
        # Test various inputs
        test_inputs = [
            "",
            "   ",
            "a",
            "What is X?",
            "Write code",
            "Explain why",
            "asdfghjkl",
            "What is X? Write code. Explain why. Analyze this. Create story. How to do it.",
            "x" * 10000,
        ]
        
        for text in test_inputs:
            result = classifier.classify(text)
            assert 0.0 <= result[1] <= 1.0, \
                f"Confidence {result[1]} for input '{text[:50]}...' is in [0.0, 1.0]"
    
    def test_classify_return_structure_various_inputs(self):
        """Invariant: Return structure is consistent across diverse inputs."""
        classifier = TaskClassifier()
        
        test_inputs = [
            "What is the capital?",
            "",
            "123",
            "@#$%",
            "Write a function",
            "   \n\t   ",
        ]
        
        for text in test_inputs:
            result = classifier.classify(text)
            assert isinstance(result, tuple), f"Result for '{text}' is a tuple"
            assert len(result) == 2, f"Result for '{text}' has length 2"
            assert isinstance(result[0], TaskType), f"First element for '{text}' is TaskType"
            assert isinstance(result[1], float), f"Second element for '{text}' is float"


class TestClassifyPerformance:
    """Performance and robustness tests."""
    
    @pytest.mark.timeout(1)
    def test_classify_performance_typical_text(self):
        """Performance: Typical text (<1KB) completes within timeout."""
        classifier = TaskClassifier()
        text = "What is machine learning and how does it work? " * 10
        
        result = classifier.classify(text)
        
        assert isinstance(result, tuple)
        assert len(result) == 2
    
    @pytest.mark.timeout(5)
    def test_classify_performance_large_text(self):
        """Performance: Large text (>100KB) completes within reasonable time."""
        classifier = TaskClassifier()
        text = "What is the capital of France? " * 5000  # ~150KB
        
        result = classifier.classify(text)
        
        assert isinstance(result, tuple)
        assert len(result) == 2


class TestTaskTypeEnum:
    """Tests for TaskType enum."""
    
    def test_task_type_enum_members(self):
        """Verify all TaskType enum members exist."""
        assert hasattr(TaskType, 'factual')
        assert hasattr(TaskType, 'reasoning')
        assert hasattr(TaskType, 'code')
        assert hasattr(TaskType, 'analysis')
        assert hasattr(TaskType, 'creative')
        assert hasattr(TaskType, 'instruction')
        assert hasattr(TaskType, 'unknown')
    
    def test_task_type_enum_count(self):
        """Verify TaskType has exactly 7 members."""
        assert len(list(TaskType)) == 7


class TestClassifierAdditionalEdgeCases:
    """Additional edge case coverage."""
    
    def test_classify_punctuation_heavy(self):
        """Edge case: Text with heavy punctuation."""
        classifier = TaskClassifier()
        text = "What??? Is!!! This????"
        
        result = classifier.classify(text)
        
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], TaskType)
        assert 0.0 <= result[1] <= 1.0
    
    def test_classify_repeated_words(self):
        """Edge case: Repeated words."""
        classifier = TaskClassifier()
        text = "what what what what what"
        
        result = classifier.classify(text)
        
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], TaskType)
        assert 0.0 <= result[1] <= 1.0
    
    def test_classify_mixed_case(self):
        """Edge case: Mixed case text."""
        classifier = TaskClassifier()
        text = "WhAt Is ThE cApItAl Of FrAnCe?"
        
        result = classifier.classify(text)
        
        assert result[0] == TaskType.factual
        assert 0.0 <= result[1] <= 1.0
    
    def test_classify_with_urls(self):
        """Edge case: Text with URLs."""
        classifier = TaskClassifier()
        text = "What is https://example.com about?"
        
        result = classifier.classify(text)
        
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], TaskType)
        assert 0.0 <= result[1] <= 1.0
    
    def test_classify_with_code_snippet(self):
        """Edge case: Text with code snippet."""
        classifier = TaskClassifier()
        text = "Write a function: def hello(): return 'world'"
        
        result = classifier.classify(text)
        
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], TaskType)
        assert 0.0 <= result[1] <= 1.0


class TestConfidenceCalculation:
    """Tests specifically for confidence calculation logic."""
    
    def test_classify_empty_gives_zero_confidence(self):
        """Verify empty input gives 0.0 confidence."""
        classifier = TaskClassifier()
        
        result = classifier.classify("")
        assert result[1] == 0.0
    
    def test_classify_no_match_gives_half_confidence(self):
        """Verify no pattern match gives 0.5 confidence."""
        classifier = TaskClassifier()
        
        # Text that shouldn't match any patterns
        result = classifier.classify("zxcvbnm asdfghjkl qwertyuiop")
        assert result[1] == 0.5
    
    def test_classify_strong_match_high_confidence(self):
        """Verify strong single-category match gives high confidence."""
        classifier = TaskClassifier()
        
        # Very code-specific text
        result = classifier.classify("def function(): pass class MyClass: method()")
        assert result[0] == TaskType.code
        assert result[1] > 0.5  # Should have confidence above baseline
    
    def test_classify_confidence_is_float(self):
        """Verify confidence is always a float, never int."""
        classifier = TaskClassifier()
        
        test_inputs = ["", "What is X?", "asdfghjkl"]
        for text in test_inputs:
            result = classifier.classify(text)
            assert type(result[1]) is float, f"Confidence for '{text}' is float type"
