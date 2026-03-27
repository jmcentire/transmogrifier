"""Tests for model profile cache."""

from transmogrifier.profiles import ModelProfile, ProfileCache, RegisterAccuracy


def test_model_profile_spread():
    p = ModelProfile(
        model_name="test",
        accuracies=[
            RegisterAccuracy(register="direct", accuracy=0.9),
            RegisterAccuracy(register="casual", accuracy=0.7),
        ],
    )
    assert abs(p.spread_pp - 20.0) < 0.01
    assert p.best_register == "direct"
    assert p.worst_register == "casual"
    assert not p.is_invariant


def test_invariant_model():
    p = ModelProfile(
        model_name="test",
        accuracies=[
            RegisterAccuracy(register="direct", accuracy=0.875),
            RegisterAccuracy(register="casual", accuracy=0.875),
        ],
    )
    assert p.spread_pp == 0.0
    assert p.is_invariant


def test_preseeded_profiles():
    cache = ProfileCache()

    opus = cache.get("claude-opus-4")
    assert opus is not None
    assert opus.spread_pp > 15

    gpt = cache.get("gpt-4o-mini")
    assert gpt is not None
    assert gpt.is_invariant

    gemini = cache.get("gemini-2-5-flash")
    assert gemini is not None
    assert gemini.spread_pp > 50


def test_alias_resolution():
    cache = ProfileCache()
    p = cache.get("claude-opus-4-20250514")
    assert p is not None
    assert p.model_name == "claude-opus-4"


def test_cache_put_get(tmp_path):
    cache = ProfileCache(cache_dir=tmp_path)
    profile = ModelProfile(
        model_name="test-model",
        model_version="v1",
        accuracies=[
            RegisterAccuracy(register="direct", accuracy=0.9),
            RegisterAccuracy(register="casual", accuracy=0.6),
        ],
    )
    cache.put(profile)

    loaded = cache.get("test-model")
    assert loaded is not None
    assert abs(loaded.spread_pp - 30.0) < 0.01


def test_missing_profile():
    cache = ProfileCache()
    assert cache.get("nonexistent-model-xyz") is None
