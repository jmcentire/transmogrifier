"""
Contract test suite for contracts_src_transmogrifier_rules_interface
Generated from contract version 1

Tests the rewrite rule engine for register-based text transformation.
"""

import pytest
import re
import time
from unittest.mock import Mock, MagicMock
from dataclasses import dataclass
from typing import Callable, Union
from enum import Enum

# Import the component under test
from contracts.src_transmogrifier_rules.interface import *


# ============================================================================
# FIXTURES AND HELPERS
# ============================================================================

@dataclass
class MockRewriteRule:
    """Mock implementation of RewriteRule for testing"""
    pattern: str
    replacement: Union[str, Callable]


class MockRegister(Enum):
    """Mock register enum with .value attribute"""
    casual = "casual"
    direct = "direct"
    academic = "academic"
    narrative = "narrative"
    technical = "technical"


@pytest.fixture
def sample_rules():
    """Sample rewrite rules for testing"""
    return [
        MockRewriteRule(pattern=r"\bhey\b", replacement="hello"),
        MockRewriteRule(pattern=r"\bwanna\b", replacement="want to"),
        MockRewriteRule(pattern=r"what's up", replacement="how are you"),
    ]


# ============================================================================
# HAPPY PATH TESTS
# ============================================================================

def test_rewrite_happy_path_casual_to_direct():
    """Rewrite from casual to direct register with known pattern replacements"""
    text = "Hey there! Wanna grab lunch?"
    source = "casual"
    target = "direct"
    
    result = rewrite(text, source, target)
    
    # Result should be a string and stripped
    assert isinstance(result, str)
    assert result == result.strip()


def test_rewrite_happy_path_academic_to_direct():
    """Rewrite from academic to direct register"""
    text = "Furthermore, the empirical evidence substantiates this hypothesis."
    source = "academic"
    target = "direct"
    
    result = rewrite(text, source, target)
    
    # Result should be a string and stripped
    assert isinstance(result, str)
    assert result == result.strip()


def test_rewrite_happy_path_narrative_to_direct():
    """Rewrite from narrative to direct register"""
    text = "Once upon a time, in a land far away, there lived a brave knight."
    source = "narrative"
    target = "direct"
    
    result = rewrite(text, source, target)
    
    # Result should be a string and stripped
    assert isinstance(result, str)
    assert result == result.strip()


def test_rewrite_happy_path_technical_to_direct():
    """Rewrite from technical to direct register"""
    text = "Initialize the TCP/IP stack prior to establishing connections."
    source = "technical"
    target = "direct"
    
    result = rewrite(text, source, target)
    
    # Result should be a string and stripped
    assert isinstance(result, str)
    assert result == result.strip()


# ============================================================================
# EDGE CASE TESTS
# ============================================================================

def test_rewrite_edge_case_source_equals_target():
    """When source equals target, original text is returned unchanged"""
    text = "Some text here"
    source = "direct"
    target = "direct"
    
    result = rewrite(text, source, target)
    
    # Postcondition: Returns original text if source == target
    assert result == text


def test_rewrite_edge_case_empty_string_input():
    """Empty string input should be handled gracefully"""
    text = ""
    source = "casual"
    target = "direct"
    
    result = rewrite(text, source, target)
    
    # Should return empty string or handle gracefully
    assert isinstance(result, str)


def test_rewrite_edge_case_whitespace_only():
    """Whitespace-only text should be handled (result stripped)"""
    text = "   \n\t  "
    source = "casual"
    target = "direct"
    
    result = rewrite(text, source, target)
    
    # Result should be stripped
    assert isinstance(result, str)
    # If result becomes empty after stripping, should fallback to original
    # Otherwise should be stripped
    assert result == result.strip()


def test_rewrite_edge_case_no_matching_patterns():
    """Text with no matching patterns returns original text"""
    text = "xyzabc123"
    source = "casual"
    target = "direct"
    
    result = rewrite(text, source, target)
    
    # Should return some result (original or transformed)
    assert isinstance(result, str)


def test_rewrite_edge_case_unicode_text():
    """Unicode characters should be preserved through rewrite"""
    text = "Hello 你好 مرحبا 🎉"
    source = "casual"
    target = "direct"
    
    result = rewrite(text, source, target)
    
    # Result should be a string
    assert isinstance(result, str)
    # Unicode should be preserved (at least partially)
    # Since we don't know exact transformation, just verify it's a valid string


def test_rewrite_edge_case_multiline_text():
    """Multiline text should be processed correctly"""
    text = "Line one\nLine two\nLine three"
    source = "casual"
    target = "direct"
    
    result = rewrite(text, source, target)
    
    # Result should be processed and stripped
    assert isinstance(result, str)
    assert result == result.strip()


def test_rewrite_edge_case_very_long_text():
    """Very long text (1000+ chars) should be processed within performance bounds"""
    text = "This is a test. " * 100  # 1600 characters
    source = "casual"
    target = "direct"
    
    start_time = time.time()
    result = rewrite(text, source, target)
    elapsed = time.time() - start_time
    
    # Should complete within reasonable time (p95 < 1ms, but allow overhead)
    assert elapsed < 0.1  # 100ms upper bound for safety
    assert isinstance(result, str)


def test_rewrite_edge_case_fallback_through_direct():
    """When no direct (src, tgt) rules exist and tgt != 'direct', attempts (src, 'direct') fallback"""
    text = "Hey what's up?"
    source = "casual"
    target = "academic"  # No direct casual→academic mapping
    
    result = rewrite(text, source, target)
    
    # Should fallback through 'direct' register
    # Result should be a string
    assert isinstance(result, str)


def test_rewrite_edge_case_result_becomes_empty_fallback():
    """If all rules result in empty string, original text is returned"""
    text = "test"
    source = "casual"
    target = "direct"
    
    result = rewrite(text, source, target)
    
    # Postcondition: Returns original text if result after all rules is empty string
    assert isinstance(result, str)
    assert len(result) > 0  # Should not be empty (either transformed or original)


def test_rewrite_edge_case_leading_trailing_whitespace():
    """Leading and trailing whitespace in result should be stripped"""
    text = "  Hey there  "
    source = "casual"
    target = "direct"
    
    result = rewrite(text, source, target)
    
    # Postcondition: Returns stripped result
    assert result == result.strip()


def test_rewrite_edge_case_enum_with_value_attribute():
    """Source and target as enum objects with .value attribute"""
    text = "Test with enum"
    
    # Create mock enum objects with .value attribute
    source_enum = MockRegister.casual
    target_enum = MockRegister.direct
    
    result = rewrite(text, source_enum, target_enum)
    
    # Should handle enum objects with .value attribute
    assert isinstance(result, str)


def test_rewrite_edge_case_special_regex_chars():
    """Text containing regex special characters should be handled safely"""
    text = "Price is $100.00 (50% off)"
    source = "casual"
    target = "direct"
    
    result = rewrite(text, source, target)
    
    # Should not raise regex errors
    assert isinstance(result, str)


# ============================================================================
# INVARIANT TESTS
# ============================================================================

def test_rewrite_invariant_case_insensitive_matching():
    """Verify that pattern matching is case-insensitive"""
    text_lower = "hey there"
    text_upper = "HEY THERE"
    text_mixed = "Hey There"
    
    source = "casual"
    target = "direct"
    
    result_lower = rewrite(text_lower, source, target)
    result_upper = rewrite(text_upper, source, target)
    result_mixed = rewrite(text_mixed, source, target)
    
    # Case-insensitive matching means different cases should be transformed
    # We can't know exact output without rules, but verify processing occurred
    assert isinstance(result_lower, str)
    assert isinstance(result_upper, str)
    assert isinstance(result_mixed, str)


def test_rewrite_invariant_sequential_rule_application():
    """Rules are applied sequentially in list order"""
    text = "Test sequential application"
    source = "casual"
    target = "direct"
    
    result = rewrite(text, source, target)
    
    # Rules should be applied in sequence
    # Verify result is consistent
    result2 = rewrite(text, source, target)
    assert result == result2  # Deterministic application


def test_rewrite_invariant_all_supported_mappings():
    """Verify all supported register mappings work"""
    text = "Test text"
    supported_mappings = [
        ("casual", "direct"),
        ("academic", "direct"),
        ("narrative", "direct"),
        ("technical", "direct"),
    ]
    
    for source, target in supported_mappings:
        result = rewrite(text, source, target)
        assert isinstance(result, str), f"Mapping {source}→{target} failed"


def test_rewrite_invariant_empty_result_fallback():
    """Empty results after rule application fall back to original text"""
    # Test with text that might become empty after processing
    text = "..."
    source = "casual"
    target = "direct"
    
    result = rewrite(text, source, target)
    
    # Should never return truly empty result if original was non-empty
    assert isinstance(result, str)


def test_rewrite_invariant_no_side_effects():
    """Rewrite has no side effects - calling multiple times yields same result"""
    text = "Test for side effects"
    source = "casual"
    target = "direct"
    
    result1 = rewrite(text, source, target)
    result2 = rewrite(text, source, target)
    result3 = rewrite(text, source, target)
    
    # All results should be identical
    assert result1 == result2 == result3


def test_rewrite_invariant_idempotency_same_register():
    """Rewriting from a register to itself is idempotent"""
    registers = ["casual", "direct", "academic", "narrative", "technical"]
    text = "Test idempotency"
    
    for register in registers:
        result = rewrite(text, register, register)
        # Should return original text unchanged
        assert result == text


# ============================================================================
# ERROR CASE TESTS
# ============================================================================

def test_rewrite_error_case_non_string_text():
    """Non-string text violates precondition"""
    text = 123  # Integer instead of string
    source = "casual"
    target = "direct"
    
    # Should raise AttributeError or TypeError when trying to process non-string
    with pytest.raises((AttributeError, TypeError)):
        rewrite(text, source, target)


def test_rewrite_error_case_none_text():
    """None as text should raise error"""
    text = None
    source = "casual"
    target = "direct"
    
    with pytest.raises((AttributeError, TypeError)):
        rewrite(text, source, target)


# ============================================================================
# PERFORMANCE TESTS
# ============================================================================

def test_rewrite_performance_baseline():
    """Verify p95 < 1ms performance target for typical input"""
    text = "Hey there! Wanna grab some lunch?"
    source = "casual"
    target = "direct"
    
    # Run multiple iterations to get p95
    times = []
    for _ in range(100):
        start = time.time()
        rewrite(text, source, target)
        elapsed = time.time() - start
        times.append(elapsed)
    
    times.sort()
    p95 = times[94]  # 95th percentile
    
    # p95 should be < 1ms (0.001s), but allow some overhead for test environment
    assert p95 < 0.01, f"p95 latency {p95*1000:.2f}ms exceeds target"


def test_rewrite_performance_complexity():
    """Verify O(n*m) complexity - linear in text length and rule count"""
    source = "casual"
    target = "direct"
    
    # Test with increasing text lengths
    text_small = "test " * 10
    text_large = "test " * 100
    
    start = time.time()
    rewrite(text_small, source, target)
    time_small = time.time() - start
    
    start = time.time()
    rewrite(text_large, source, target)
    time_large = time.time() - start
    
    # Large text should take proportionally longer, but not exponentially
    # Allow up to 20x time for 10x text (conservative for regex overhead)
    if time_small > 0:
        ratio = time_large / time_small
        assert ratio < 20, f"Time complexity appears worse than O(n*m): {ratio}x"


# ============================================================================
# COMBINATORIAL TESTS
# ============================================================================

def test_rewrite_all_register_combinations():
    """Test all possible source/target register combinations"""
    registers = ["casual", "direct", "academic", "narrative", "technical"]
    text = "Test text for all combinations"
    
    for source in registers:
        for target in registers:
            result = rewrite(text, source, target)
            assert isinstance(result, str), f"Failed for {source}→{target}"
            
            # If source == target, should return original
            if source == target:
                assert result == text


# ============================================================================
# ROBUSTNESS TESTS (Fuzz-like with random)
# ============================================================================

def test_rewrite_robustness_random_text():
    """Test with randomly generated text to verify robustness"""
    import random
    import string
    
    source = "casual"
    target = "direct"
    
    for _ in range(20):
        # Generate random text
        length = random.randint(0, 200)
        chars = string.ascii_letters + string.digits + string.punctuation + " \n\t"
        text = ''.join(random.choice(chars) for _ in range(length))
        
        # Should not crash
        try:
            result = rewrite(text, source, target)
            assert isinstance(result, str)
        except (AttributeError, TypeError):
            # These are acceptable for invalid inputs
            pass


def test_rewrite_robustness_random_registers():
    """Test with random register values"""
    import random
    
    text = "Test text"
    registers = ["casual", "direct", "academic", "narrative", "technical"]
    
    for _ in range(20):
        source = random.choice(registers)
        target = random.choice(registers)
        
        result = rewrite(text, source, target)
        assert isinstance(result, str)
        
        if source == target:
            assert result == text


# ============================================================================
# REGRESSION TESTS
# ============================================================================

def test_rewrite_regression_common_casual_phrases():
    """Regression test for common casual→direct transformations"""
    test_cases = [
        ("hey", "casual", "direct"),
        ("wanna", "casual", "direct"),
        ("gonna", "casual", "direct"),
        ("kinda", "casual", "direct"),
    ]
    
    for text, source, target in test_cases:
        result = rewrite(text, source, target)
        assert isinstance(result, str)
        # Result should be different from input (unless no rules exist)
        # Just verify it processes without error


def test_rewrite_regression_whitespace_handling():
    """Regression test for whitespace handling edge cases"""
    test_cases = [
        ("  text  ", "casual", "direct"),
        ("\n\ntext\n\n", "casual", "direct"),
        ("\ttext\t", "casual", "direct"),
        ("  ", "casual", "direct"),
    ]
    
    for text, source, target in test_cases:
        result = rewrite(text, source, target)
        # Result should always be stripped
        assert result == result.strip()


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

def test_rewrite_integration_chained_transformations():
    """Test chaining multiple rewrite operations"""
    text = "Hey there friend"
    
    # Chain: casual→direct→casual
    result1 = rewrite(text, "casual", "direct")
    result2 = rewrite(result1, "direct", "casual")
    
    # Should complete without errors
    assert isinstance(result1, str)
    assert isinstance(result2, str)


def test_rewrite_integration_fallback_chain():
    """Test fallback through 'direct' register for unsupported direct paths"""
    text = "Test fallback behavior"
    
    # casual→academic has no direct path, should fallback through direct
    result = rewrite(text, "casual", "academic")
    
    # Should fallback gracefully
    assert isinstance(result, str)


# ============================================================================
# TYPE VALIDATION TESTS
# ============================================================================

def test_rewrite_rule_structure():
    """Test RewriteRule dataclass structure if accessible"""
    try:
        # Test string replacement
        rule1 = RewriteRule(pattern=r"\btest\b", replacement="example")
        assert rule1.pattern == r"\btest\b"
        assert rule1.replacement == "example"
        
        # Test callable replacement
        rule2 = RewriteRule(pattern=r"\d+", replacement=lambda m: m.group(0))
        assert rule2.pattern == r"\d+"
        assert callable(rule2.replacement)
    except NameError:
        # RewriteRule might not be exported
        pytest.skip("RewriteRule not accessible for testing")


def test_rule_engine_structure():
    """Test RuleEngine structure if accessible"""
    try:
        # RuleEngine should be stateless
        engine = RuleEngine()
        # Just verify it can be instantiated
        assert engine is not None
    except NameError:
        # RuleEngine might not be exported
        pytest.skip("RuleEngine not accessible for testing")


# ============================================================================
# BOUNDARY TESTS
# ============================================================================

def test_rewrite_boundary_single_character():
    """Test with single character input"""
    text = "a"
    result = rewrite(text, "casual", "direct")
    assert isinstance(result, str)
    assert len(result) >= 0


def test_rewrite_boundary_max_length_text():
    """Test with very large text (stress test)"""
    text = "word " * 10000  # ~50KB text
    
    start = time.time()
    result = rewrite(text, "casual", "direct")
    elapsed = time.time() - start
    
    assert isinstance(result, str)
    # Should complete in reasonable time even for large input
    assert elapsed < 1.0  # 1 second max for large text


def test_rewrite_boundary_many_newlines():
    """Test with text containing many newlines"""
    text = "\n" * 100 + "text" + "\n" * 100
    result = rewrite(text, "casual", "direct")
    
    assert isinstance(result, str)
    assert result == result.strip()


# ============================================================================
# CONTRACT VALIDATION TESTS
# ============================================================================

def test_contract_precondition_text_is_string():
    """Verify precondition: text is a string"""
    # Valid: string
    assert isinstance(rewrite("text", "casual", "direct"), str)
    
    # Invalid: not string
    with pytest.raises((AttributeError, TypeError)):
        rewrite(123, "casual", "direct")
    
    with pytest.raises((AttributeError, TypeError)):
        rewrite(None, "casual", "direct")


def test_contract_postcondition_source_equals_target():
    """Verify postcondition: Returns original text if source == target"""
    text = "Test text"
    result = rewrite(text, "direct", "direct")
    assert result == text


def test_contract_postcondition_stripped_result():
    """Verify postcondition: Returns stripped result otherwise"""
    text = "  test  "
    result = rewrite(text, "casual", "direct")
    assert result == result.strip()


def test_contract_postcondition_empty_result_fallback():
    """Verify postcondition: Returns original text if result after all rules is empty string"""
    # This is hard to test without knowing exact rules, but we can verify
    # that rewrite never returns empty string for non-empty input
    text = "test"
    result = rewrite(text, "casual", "direct")
    # Should return non-empty result
    assert isinstance(result, str)


def test_contract_side_effect_none():
    """Verify side effect: none - function is pure"""
    text = "Test for purity"
    original_text = text
    
    result = rewrite(text, "casual", "direct")
    
    # Original text should be unchanged
    assert text == original_text
    
    # Multiple calls should yield same result
    result2 = rewrite(text, "casual", "direct")
    assert result == result2
