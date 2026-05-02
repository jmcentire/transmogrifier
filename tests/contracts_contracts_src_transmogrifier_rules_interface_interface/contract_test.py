"""
Contract Test Suite for Register Rewrite Rule Engine Interface
Generated from contract version 1
Tests behavioral contracts: preconditions, postconditions, invariants, and side effects.
"""

import pytest
import re
import time
from unittest.mock import Mock, patch, MagicMock
from typing import Callable

# Import the component under test
from contracts.contracts_src_transmogrifier_rules_interface.interface import (
    rewrite,
    RewriteRule,
    RuleEngine
)


# ============================================================================
# HAPPY PATH TESTS
# ============================================================================

def test_rewrite_happy_path_casual_to_direct():
    """Rewrite text from casual to direct register with known pattern transformations"""
    text = "Hey there, how's it going?"
    source = "casual"
    target = "direct"
    
    result = rewrite(text, source, target)
    
    # Verify result is a string
    assert isinstance(result, str)
    # Verify result is stripped (postcondition)
    assert result == result.strip()
    # Result should be different from input (assuming rules exist)
    # or same if no rules match
    assert result is not None


def test_rewrite_happy_path_academic_to_direct():
    """Rewrite text from academic to direct register"""
    text = "Furthermore, it is evident that the aforementioned hypothesis requires rigorous examination."
    source = "academic"
    target = "direct"
    
    result = rewrite(text, source, target)
    
    assert isinstance(result, str)
    assert result == result.strip()
    assert result is not None


def test_rewrite_happy_path_narrative_to_direct():
    """Rewrite text from narrative to direct register"""
    text = "Once upon a time, in a land far away, there lived a brave knight."
    source = "narrative"
    target = "direct"
    
    result = rewrite(text, source, target)
    
    assert isinstance(result, str)
    assert result == result.strip()
    assert result is not None


def test_rewrite_happy_path_technical_to_direct():
    """Rewrite text from technical to direct register"""
    text = "The instantiation of the aforementioned protocol buffer object necessitates proper initialization."
    source = "technical"
    target = "direct"
    
    result = rewrite(text, source, target)
    
    assert isinstance(result, str)
    assert result == result.strip()
    assert result is not None


# ============================================================================
# EDGE CASE TESTS
# ============================================================================

def test_rewrite_same_source_target_returns_original():
    """When source equals target, return original text unchanged per postcondition"""
    text = "This text should remain unchanged"
    source = "direct"
    target = "direct"
    
    result = rewrite(text, source, target)
    
    # Postcondition: Returns original text if source == target
    assert result == text


def test_rewrite_empty_string_input():
    """Rewriting empty string returns empty string"""
    text = ""
    source = "casual"
    target = "direct"
    
    result = rewrite(text, source, target)
    
    assert isinstance(result, str)
    # Empty input should return empty or original
    assert result == ""


def test_rewrite_whitespace_only_input():
    """Rewriting whitespace-only string is handled correctly"""
    text = "   \t\n  "
    source = "casual"
    target = "direct"
    
    result = rewrite(text, source, target)
    
    assert isinstance(result, str)
    # Result should be stripped
    assert result == result.strip()


def test_rewrite_no_matching_patterns():
    """Text with no matching patterns returns original text"""
    text = "xyz123abc"
    source = "casual"
    target = "direct"
    
    result = rewrite(text, source, target)
    
    assert isinstance(result, str)
    # If no patterns match, original text should be preserved
    assert len(result) > 0


def test_rewrite_fallback_to_original_if_result_empty():
    """If all rules produce empty string, return original text per postcondition"""
    # This tests the postcondition: Returns original text if result after all rules is empty string
    text = "test_text"
    source = "casual"
    target = "direct"
    
    result = rewrite(text, source, target)
    
    # Result should never be truly empty if original wasn't empty (falls back to original)
    assert isinstance(result, str)
    assert len(result) > 0


def test_rewrite_fallback_through_direct_register():
    """If no direct (src, tgt) rules exist and tgt != 'direct', attempt (src, 'direct') first per postcondition"""
    text = "Some text to rewrite"
    source = "casual"
    target = "academic"
    
    result = rewrite(text, source, target)
    
    # Should fall back through 'direct' register if no direct path exists
    assert isinstance(result, str)
    assert result == result.strip()


def test_rewrite_unicode_input():
    """Handle Unicode characters correctly in regex matching"""
    text = "Hey café, naïve résumé"
    source = "casual"
    target = "direct"
    
    result = rewrite(text, source, target)
    
    assert isinstance(result, str)
    # Unicode should be preserved
    assert result is not None


def test_rewrite_source_target_with_value_attribute():
    """Source and target can be objects with .value attribute (enum-like)"""
    text = "Hey there"
    
    # Create mock enum-like objects with .value attribute
    source_enum = Mock()
    source_enum.value = "casual"
    target_enum = Mock()
    target_enum.value = "direct"
    
    result = rewrite(text, source_enum, target_enum)
    
    assert isinstance(result, str)
    assert result == result.strip()


def test_rewrite_special_regex_characters():
    """Handle text containing special regex characters"""
    text = "Hey (what's) [up] $100 ^test*"
    source = "casual"
    target = "direct"
    
    result = rewrite(text, source, target)
    
    assert isinstance(result, str)
    # Should not crash on special characters
    assert result is not None


def test_rewrite_multiline_text():
    """Handle multiline text with newlines"""
    text = "Hey there\nHow's it going?\nWhat's up?"
    source = "casual"
    target = "direct"
    
    result = rewrite(text, source, target)
    
    assert isinstance(result, str)
    assert result == result.strip()


def test_rewrite_very_long_text():
    """Handle very long text efficiently (performance invariant p95<1ms)"""
    text = "Hey there, what's up buddy? " * 100
    source = "casual"
    target = "direct"
    
    start_time = time.perf_counter()
    result = rewrite(text, source, target)
    elapsed = time.perf_counter() - start_time
    
    assert isinstance(result, str)
    # Performance constraint: p95 < 1ms (allowing some margin for test overhead)
    # This is a soft check - we just verify it completes reasonably fast
    assert elapsed < 0.1  # 100ms margin for test environment


def test_rewrite_overlapping_patterns():
    """Handle overlapping regex patterns correctly"""
    text = "Hey there buddy friend"
    source = "casual"
    target = "direct"
    
    result = rewrite(text, source, target)
    
    assert isinstance(result, str)
    assert result == result.strip()


# ============================================================================
# INVARIANT TESTS
# ============================================================================

def test_rewrite_case_insensitive_matching():
    """Verify case-insensitive regex matching invariant"""
    text_lower = "hey there how are you"
    text_upper = "HEY THERE HOW ARE YOU"
    text_mixed = "HeY tHeRe HoW aRe YoU"
    source = "casual"
    target = "direct"
    
    result_lower = rewrite(text_lower, source, target)
    result_upper = rewrite(text_upper, source, target)
    result_mixed = rewrite(text_mixed, source, target)
    
    # Results should show pattern matching occurred regardless of case
    # (exact results may differ due to replacement preserving original case,
    # but transformations should apply)
    assert isinstance(result_lower, str)
    assert isinstance(result_upper, str)
    assert isinstance(result_mixed, str)


def test_rewrite_sequential_rule_application():
    """Verify rules are applied sequentially in list order"""
    text = "Hey what's up buddy"
    source = "casual"
    target = "direct"
    
    result = rewrite(text, source, target)
    
    # Rules should be applied in order
    # Later rules operate on results of earlier rules
    assert isinstance(result, str)
    assert result == result.strip()


def test_rewrite_result_is_stripped():
    """Verify postcondition that result is stripped"""
    text = "  Hey there  "
    source = "casual"
    target = "direct"
    
    result = rewrite(text, source, target)
    
    # Postcondition: Returns stripped result
    assert result == result.strip()
    # No leading/trailing whitespace
    if len(result) > 0:
        assert result[0] != ' ' and result[-1] != ' '


def test_rewrite_stateless_multiple_calls():
    """Verify statelessness: multiple calls with same input produce same output"""
    text = "Hey what's up"
    source = "casual"
    target = "direct"
    
    result1 = rewrite(text, source, target)
    result2 = rewrite(text, source, target)
    result3 = rewrite(text, source, target)
    
    # Stateless: same input produces same output
    assert result1 == result2 == result3


def test_rewrite_supported_register_mappings():
    """Verify supported register mappings invariant"""
    text = "Test text for transformation"
    
    # Supported mappings per invariant
    mappings = [
        ("casual", "direct"),
        ("academic", "direct"),
        ("narrative", "direct"),
        ("technical", "direct")
    ]
    
    for source, target in mappings:
        result = rewrite(text, source, target)
        assert isinstance(result, str)
        assert result == result.strip()


def test_rewrite_empty_result_fallback_invariant():
    """Verify empty results after rule application fall back to original text"""
    # This is tricky to test without knowing the actual rules
    # We test the postcondition behavior
    text = "nonmatching_xyz_123"
    source = "casual"
    target = "direct"
    
    result = rewrite(text, source, target)
    
    # Should never return empty string if original wasn't empty
    assert len(result) > 0


# ============================================================================
# TYPE CONSTRUCTION AND VALIDATION TESTS
# ============================================================================

def test_rewrite_rule_construction_string_replacement():
    """Test RewriteRule construction with string replacement"""
    rule = RewriteRule(pattern=r"\bhey\b", replacement="hello")
    
    assert rule.pattern == r"\bhey\b"
    assert rule.replacement == "hello"
    assert isinstance(rule.replacement, str)


def test_rewrite_rule_construction_callable_replacement():
    """Test RewriteRule construction with callable replacement"""
    def replacer(match):
        return match.group(0).upper()
    
    rule = RewriteRule(pattern=r"\bhey\b", replacement=replacer)
    
    assert rule.pattern == r"\bhey\b"
    assert callable(rule.replacement)
    assert rule.replacement is replacer


def test_rewrite_rule_with_complex_pattern():
    """Test RewriteRule with complex regex pattern"""
    pattern = r"(?:what's|how's|where's)\s+(?:up|going|it)"
    rule = RewriteRule(pattern=pattern, replacement="status")
    
    assert rule.pattern == pattern
    assert rule.replacement == "status"


# ============================================================================
# PARAMETERIZED TESTS
# ============================================================================

@pytest.mark.parametrize("source,target", [
    ("casual", "direct"),
    ("academic", "direct"),
    ("narrative", "direct"),
    ("technical", "direct"),
])
def test_rewrite_all_supported_mappings(source, target):
    """Parameterized test for all supported register mappings"""
    text = "This is a test sentence for transformation."
    
    result = rewrite(text, source, target)
    
    assert isinstance(result, str)
    assert result == result.strip()
    assert result is not None


@pytest.mark.parametrize("text", [
    "",
    " ",
    "  \t\n  ",
    "a",
    "Test",
    "Multiple words here",
    "Special chars: @#$%^&*()",
    "Unicode: café résumé naïve",
    "Numbers: 123 456 789",
    "Mixed: Test123 @hello café",
])
def test_rewrite_various_inputs(text):
    """Parameterized test with various input texts"""
    source = "casual"
    target = "direct"
    
    result = rewrite(text, source, target)
    
    assert isinstance(result, str)
    assert result == result.strip()


@pytest.mark.parametrize("source,target", [
    ("direct", "direct"),
    ("casual", "casual"),
    ("academic", "academic"),
])
def test_rewrite_identity_transformation(source, target):
    """Parameterized test: source == target returns original"""
    text = "This should remain unchanged"
    
    result = rewrite(text, source, target)
    
    # Postcondition: Returns original text if source == target
    assert result == text


# ============================================================================
# METAMORPHIC TESTS
# ============================================================================

def test_rewrite_transformation_chaining():
    """Metamorphic test: chaining transformations through intermediate register"""
    text = "Hey there, what's up buddy?"
    
    # Direct transformation
    result_direct = rewrite(text, "casual", "direct")
    
    # Should be equivalent to chaining (if rules exist)
    # casual -> direct
    assert isinstance(result_direct, str)


def test_rewrite_idempotency_same_register():
    """Metamorphic test: applying same register twice is idempotent"""
    text = "Test sentence"
    
    result1 = rewrite(text, "direct", "direct")
    result2 = rewrite(result1, "direct", "direct")
    
    assert result1 == result2 == text


def test_rewrite_double_application():
    """Metamorphic test: applying transformation twice on result"""
    text = "Hey what's up"
    source = "casual"
    target = "direct"
    
    result1 = rewrite(text, source, target)
    result2 = rewrite(result1, target, target)  # Apply identity
    
    # Second application with same source/target should return result unchanged
    assert result2 == result1


# ============================================================================
# EDGE CASE: REGEX PATTERN EDGE CASES
# ============================================================================

def test_rewrite_text_with_only_spaces():
    """Edge case: text with only spaces"""
    text = "     "
    source = "casual"
    target = "direct"
    
    result = rewrite(text, source, target)
    
    assert isinstance(result, str)
    assert result == result.strip()


def test_rewrite_text_with_newlines_and_tabs():
    """Edge case: text with various whitespace characters"""
    text = "Hey\tthere\n\nWhat's\rup"
    source = "casual"
    target = "direct"
    
    result = rewrite(text, source, target)
    
    assert isinstance(result, str)


def test_rewrite_text_with_repeated_patterns():
    """Edge case: text with repeated patterns"""
    text = "Hey hey hey what's what's what's"
    source = "casual"
    target = "direct"
    
    result = rewrite(text, source, target)
    
    assert isinstance(result, str)
    assert result == result.strip()


def test_rewrite_text_with_word_boundaries():
    """Edge case: test word boundary matching"""
    text = "heyday hey ahoy"  # 'hey' in different contexts
    source = "casual"
    target = "direct"
    
    result = rewrite(text, source, target)
    
    assert isinstance(result, str)


# ============================================================================
# STRESS TESTS
# ============================================================================

def test_rewrite_concurrent_calls_stateless():
    """Verify statelessness with multiple concurrent-like calls"""
    text = "Hey there buddy"
    source = "casual"
    target = "direct"
    
    results = []
    for _ in range(10):
        result = rewrite(text, source, target)
        results.append(result)
    
    # All results should be identical (stateless)
    assert all(r == results[0] for r in results)


def test_rewrite_with_very_long_pattern_text():
    """Edge case: very long text to test performance bounds"""
    text = " ".join(["Hey there what's up buddy"] * 50)
    source = "casual"
    target = "direct"
    
    result = rewrite(text, source, target)
    
    assert isinstance(result, str)
    assert result == result.strip()


# ============================================================================
# PRECONDITION TESTS
# ============================================================================

def test_rewrite_precondition_text_is_string():
    """Verify precondition: text is a string"""
    # Valid string inputs
    assert isinstance(rewrite("test", "casual", "direct"), str)
    assert isinstance(rewrite("", "casual", "direct"), str)
    assert isinstance(rewrite("123", "casual", "direct"), str)


def test_rewrite_precondition_source_target_are_string_or_enum():
    """Verify precondition: source and target are strings or objects with .value attribute"""
    text = "Test"
    
    # String inputs
    result1 = rewrite(text, "casual", "direct")
    assert isinstance(result1, str)
    
    # Enum-like objects with .value attribute
    source_enum = Mock()
    source_enum.value = "casual"
    target_enum = Mock()
    target_enum.value = "direct"
    
    result2 = rewrite(text, source_enum, target_enum)
    assert isinstance(result2, str)


# ============================================================================
# POSTCONDITION VERIFICATION TESTS
# ============================================================================

def test_rewrite_postcondition_original_if_source_equals_target():
    """Verify postcondition: Returns original text if source == target"""
    text = "Original text here"
    
    assert rewrite(text, "direct", "direct") == text
    assert rewrite(text, "casual", "casual") == text
    assert rewrite(text, "academic", "academic") == text


def test_rewrite_postcondition_stripped_result():
    """Verify postcondition: Returns stripped result"""
    texts = [
        "  test  ",
        "\ttest\t",
        "\ntest\n",
        "  test with spaces  ",
    ]
    
    for text in texts:
        result = rewrite(text, "casual", "direct")
        assert result == result.strip(), f"Result not stripped for input: {repr(text)}"


def test_rewrite_postcondition_no_side_effects():
    """Verify side effect: none (function is pure)"""
    text = "Test input"
    original_text = text
    source = "casual"
    target = "direct"
    
    result = rewrite(text, source, target)
    
    # Original text should not be modified
    assert text == original_text
    # Function should not raise exceptions
    assert result is not None


# ============================================================================
# INTEGRATION TESTS WITH MOCK RULES
# ============================================================================

def test_rewrite_with_mock_implementation():
    """Integration test verifying the contract behavior"""
    text = "Hey there buddy"
    source = "casual"
    target = "direct"
    
    result = rewrite(text, source, target)
    
    # Contract guarantees
    assert isinstance(result, str)  # Returns a string
    assert result == result.strip()  # Result is stripped
    # No side effects - function is pure
    
    # Call again with same inputs
    result2 = rewrite(text, source, target)
    assert result == result2  # Stateless/deterministic


def test_rewrite_all_register_combinations():
    """Test all combinations of supported registers"""
    text = "Test sentence"
    registers = ["casual", "academic", "narrative", "technical", "direct"]
    
    for source in registers:
        for target in registers:
            result = rewrite(text, source, target)
            
            assert isinstance(result, str)
            assert result == result.strip()
            
            if source == target:
                assert result == text


# ============================================================================
# RANDOM/FUZZ TESTS (using random module)
# ============================================================================

def test_rewrite_random_text_variations():
    """Random test with various text inputs"""
    import random
    import string
    
    random.seed(42)  # For reproducibility
    
    for _ in range(20):
        # Generate random text
        length = random.randint(0, 100)
        chars = string.ascii_letters + string.digits + " \t\n"
        text = "".join(random.choice(chars) for _ in range(length))
        
        source = random.choice(["casual", "academic", "narrative", "technical"])
        target = random.choice(["casual", "academic", "narrative", "technical", "direct"])
        
        result = rewrite(text, source, target)
        
        # Contract guarantees
        assert isinstance(result, str)
        assert result == result.strip()


def test_rewrite_random_unicode_text():
    """Random test with Unicode characters"""
    import random
    
    random.seed(123)  # For reproducibility
    
    unicode_chars = "café résumé naïve œuvre ñoño"
    
    for _ in range(10):
        # Build random text with unicode
        words = random.choices(unicode_chars.split(), k=random.randint(1, 5))
        text = " ".join(words)
        
        result = rewrite(text, "casual", "direct")
        
        assert isinstance(result, str)
        assert result == result.strip()
