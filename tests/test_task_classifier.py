"""Tests for task type classifier."""

import time

from transmogrifier.task_classifier import TaskClassifier, TaskType


def test_classify_factual():
    c = TaskClassifier()
    tt, conf = c.classify("What is the capital of France?")
    assert tt == TaskType.factual


def test_classify_reasoning():
    c = TaskClassifier()
    tt, conf = c.classify("If all dogs are mammals and all mammals breathe air, does it follow that all dogs breathe air?")
    assert tt == TaskType.reasoning


def test_classify_code():
    c = TaskClassifier()
    tt, conf = c.classify("Write a Python function that checks if a number is prime")
    assert tt == TaskType.code


def test_classify_analysis():
    c = TaskClassifier()
    tt, conf = c.classify("What are the key differences between TCP and UDP?")
    assert tt == TaskType.analysis


def test_classify_creative():
    c = TaskClassifier()
    tt, conf = c.classify("Write a poem about the ocean")
    assert tt == TaskType.creative


def test_classify_instruction():
    c = TaskClassifier()
    tt, conf = c.classify("How do you set up a virtual environment in Python step by step?")
    assert tt == TaskType.instruction


def test_classify_empty():
    c = TaskClassifier()
    tt, conf = c.classify("")
    assert tt == TaskType.unknown


def test_classify_performance():
    c = TaskClassifier()
    texts = [
        "What is the capital of France?",
        "Write a Python function",
        "Compare TCP and UDP",
        "If A then B, does C follow?",
        "Write a haiku about code",
    ] * 200
    start = time.perf_counter()
    for text in texts:
        c.classify(text)
    elapsed_ms = (time.perf_counter() - start) * 1000
    assert elapsed_ms < 1000, f"1000 classifications took {elapsed_ms:.0f}ms"
