"""Tests for register detection."""

import time

from transmogrifier.core import Register
from transmogrifier.detector import RegisterDetector


def test_detect_casual():
    d = RegisterDetector()
    reg, conf = d.detect("yo so like, what's the deal with TCP... what's up with that")
    assert reg == Register.casual
    assert conf > 0.5


def test_detect_technical():
    d = RegisterDetector()
    reg, conf = d.detect("Provide a precise technical answer: What is TCP?")
    assert reg == Register.technical
    assert conf > 0.5


def test_detect_academic():
    d = RegisterDetector()
    reg, conf = d.detect(
        "In the context of established knowledge, the TCP protocol "
        "is characterized by its reliability. Provide a scholarly response."
    )
    assert reg == Register.academic
    assert conf > 0.5


def test_detect_narrative():
    d = RegisterDetector()
    reg, conf = d.detect("Explain this as if telling a story: How does TCP work?")
    assert reg == Register.narrative
    assert conf > 0.5


def test_detect_direct():
    d = RegisterDetector()
    reg, conf = d.detect("What is TCP?")
    assert reg == Register.direct


def test_detect_empty():
    d = RegisterDetector()
    reg, conf = d.detect("")
    assert reg == Register.direct
    assert conf == 1.0


def test_detect_performance():
    d = RegisterDetector()
    texts = [
        "yo what's the deal with this",
        "Provide a precise technical answer",
        "In the context of established knowledge",
        "Explain this as if telling a story",
        "What is TCP?",
    ] * 200  # 1000 detections

    start = time.perf_counter()
    for text in texts:
        d.detect(text)
    elapsed_ms = (time.perf_counter() - start) * 1000
    assert elapsed_ms < 1000, f"1000 detections took {elapsed_ms:.0f}ms (>1s)"
