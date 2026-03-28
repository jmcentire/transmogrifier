"""Tests for calibration runner."""

from transmogrifier.calibrate import BENCHMARK_TASKS, CalibrationRunner, score_response
from transmogrifier.profiles import ProfileCache


def test_benchmark_has_enough_tasks():
    assert len(BENCHMARK_TASKS) >= 50


def test_benchmark_categories():
    categories = set(t["category"] for t in BENCHMARK_TASKS)
    assert "factual" in categories
    assert "reasoning" in categories
    assert "code" in categories
    assert "analysis" in categories
    assert "creative" in categories
    assert "instruction" in categories


def test_benchmark_category_sizes():
    from collections import Counter
    counts = Counter(t["category"] for t in BENCHMARK_TASKS)
    for cat, count in counts.items():
        assert count >= 6, f"{cat} has only {count} tasks (need >= 6)"


def test_score_response_correct():
    task = {"category": "factual", "prompt": "Capital of France?", "accept": ["paris"], "reject": []}
    assert score_response("The capital of France is Paris.", task) is True


def test_score_response_wrong():
    task = {"category": "factual", "prompt": "Capital of France?", "accept": ["paris"], "reject": []}
    assert score_response("The capital is London.", task) is False


def test_score_response_reasoning_reject_overrides():
    task = {
        "category": "reasoning",
        "prompt": "Bat and ball",
        "accept": ["0.05", "5 cents"],
        "reject": ["0.10", "10 cents"],
    }
    # Contains both accept and reject pattern — reject wins for reasoning
    assert score_response("The ball costs $0.10, wait no, $0.05", task) is False


def test_calibration_with_mock_backend(tmp_path):
    """Test calibration runner with a mock backend that always returns 'Paris'."""
    class MockBackend:
        def complete(self, system, messages, max_tokens=300):
            return "Paris"

    cache = ProfileCache(cache_dir=tmp_path)
    runner = CalibrationRunner(MockBackend(), cache)

    # Run with just 3 tasks for speed
    tasks = BENCHMARK_TASKS[:3]
    profile = runner.run(
        model_name="mock-model",
        model_version="v1",
        provider="mock",
        tasks=tasks,
        delay=0,
    )

    assert profile.model_name == "mock-model"
    assert len(profile.accuracies) == 5  # 5 registers
    assert profile.calibration_version == "2.0"
    assert profile.calibrated_at  # timestamp set
    assert (tmp_path / "mock-model.json").exists()
