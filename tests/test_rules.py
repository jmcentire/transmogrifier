"""Tests for Level 2 rule-based rewriting."""

import time

from transmogrifier.core import Register
from transmogrifier.rules import RuleEngine


def test_casual_to_direct_strips_prefix():
    e = RuleEngine()
    assert "yo" not in e.rewrite("yo what's the deal with TCP", Register.casual, Register.direct).lower()


def test_casual_to_direct_strips_filler():
    e = RuleEngine()
    result = e.rewrite("so like, basically, TCP is, you know, important", Register.casual, Register.direct)
    assert "like," not in result.lower()
    assert "basically," not in result.lower()
    assert "you know," not in result.lower()


def test_academic_to_direct_strips_hedging():
    e = RuleEngine()
    result = e.rewrite(
        "In the context of established knowledge, TCP provides reliable transport. "
        "Provide a scholarly response.",
        Register.academic,
        Register.direct,
    )
    assert "in the context" not in result.lower()
    assert "scholarly" not in result.lower()


def test_narrative_to_direct_strips_framing():
    e = RuleEngine()
    result = e.rewrite("Explain this as if telling a story: How does TCP work?", Register.narrative, Register.direct)
    assert "story" not in result.lower()


def test_technical_to_direct():
    e = RuleEngine()
    result = e.rewrite("Provide a precise technical answer: What is TCP?", Register.technical, Register.direct)
    assert "precise technical answer" not in result.lower()
    assert "TCP" in result


def test_direct_to_direct_identity():
    e = RuleEngine()
    text = "What is TCP?"
    assert e.rewrite(text, Register.direct, Register.direct) == text


def test_preserves_content():
    e = RuleEngine()
    result = e.rewrite("yo so like, what's up with activation fingerprinting", Register.casual, Register.direct)
    assert "activation fingerprinting" in result.lower()


def test_performance():
    e = RuleEngine()
    start = time.perf_counter()
    for _ in range(10000):
        e.rewrite("yo so like, what's the deal with TCP... what's up with that", Register.casual, Register.direct)
    elapsed_ms = (time.perf_counter() - start) * 1000
    assert elapsed_ms < 5000, f"10000 rewrites took {elapsed_ms:.0f}ms (>5s)"
