"""Model profile cache: register sensitivity per model, pre-seeded from experiments."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, computed_field

logger = logging.getLogger(__name__)


class RegisterAccuracy(BaseModel):
    model_config = {"ignored_types": ()}
    register: str  # noqa: shadows BaseModel.register (benign)
    accuracy: float
    sample_size: int = 16


class ModelProfile(BaseModel):
    model_name: str
    model_version: str = ""
    provider: str = ""
    accuracies: list[RegisterAccuracy]
    calibrated_at: str = ""
    ttl_hours: int = 720
    calibration_version: str = "1.0"

    @computed_field
    @property
    def spread_pp(self) -> float:
        if not self.accuracies:
            return 0.0
        accs = [a.accuracy for a in self.accuracies]
        return (max(accs) - min(accs)) * 100

    @computed_field
    @property
    def is_invariant(self) -> bool:
        return self.spread_pp < 2.0

    @computed_field
    @property
    def best_register(self) -> str:
        if not self.accuracies:
            return "direct"
        return max(self.accuracies, key=lambda a: a.accuracy).register

    @computed_field
    @property
    def worst_register(self) -> str:
        if not self.accuracies:
            return "direct"
        return min(self.accuracies, key=lambda a: a.accuracy).register

    @property
    def is_expired(self) -> bool:
        if not self.calibrated_at:
            return False
        try:
            cal = datetime.fromisoformat(self.calibrated_at)
            if cal.tzinfo is None:
                cal = cal.replace(tzinfo=timezone.utc)
            now = datetime.now(timezone.utc)
            hours = (now - cal).total_seconds() / 3600
            return hours > self.ttl_hours
        except (ValueError, TypeError):
            return False


# Pre-seeded profiles from experiments (2026-03-27)
_PRESEEDED: dict[str, ModelProfile] = {
    "claude-opus-4": ModelProfile(
        model_name="claude-opus-4",
        model_version="20250514",
        provider="anthropic",
        calibrated_at="2026-03-27T00:00:00Z",
        accuracies=[
            RegisterAccuracy(register="direct", accuracy=0.938),
            RegisterAccuracy(register="technical", accuracy=0.938),
            RegisterAccuracy(register="academic", accuracy=0.875),
            RegisterAccuracy(register="narrative", accuracy=0.875),
            RegisterAccuracy(register="casual", accuracy=0.750),
        ],
    ),
    "claude-haiku-4-5": ModelProfile(
        model_name="claude-haiku-4-5",
        model_version="20251001",
        provider="anthropic",
        calibrated_at="2026-03-27T00:00:00Z",
        accuracies=[
            RegisterAccuracy(register="direct", accuracy=0.938),
            RegisterAccuracy(register="technical", accuracy=0.875),
            RegisterAccuracy(register="academic", accuracy=0.875),
            RegisterAccuracy(register="narrative", accuracy=0.875),
            RegisterAccuracy(register="casual", accuracy=0.875),
        ],
    ),
    "gpt-4o-mini": ModelProfile(
        model_name="gpt-4o-mini",
        model_version="2024-07-18",
        provider="openai",
        calibrated_at="2026-03-27T00:00:00Z",
        accuracies=[
            RegisterAccuracy(register="direct", accuracy=0.875),
            RegisterAccuracy(register="technical", accuracy=0.875),
            RegisterAccuracy(register="academic", accuracy=0.875),
            RegisterAccuracy(register="narrative", accuracy=0.875),
            RegisterAccuracy(register="casual", accuracy=0.875),
        ],
    ),
    "gemini-2-5-flash": ModelProfile(
        model_name="gemini-2-5-flash",
        model_version="2025",
        provider="gemini",
        calibrated_at="2026-03-27T00:00:00Z",
        accuracies=[
            RegisterAccuracy(register="direct", accuracy=0.562),
            RegisterAccuracy(register="technical", accuracy=0.562),
            RegisterAccuracy(register="academic", accuracy=0.312),
            RegisterAccuracy(register="narrative", accuracy=0.000),
            RegisterAccuracy(register="casual", accuracy=0.125),
        ],
    ),
}

# Alias map: model IDs to canonical profile names
_ALIASES: dict[str, str] = {
    "claude-opus-4-20250514": "claude-opus-4",
    "claude-haiku-4-5-20251001": "claude-haiku-4-5",
    "gpt-4o-mini-2024-07-18": "gpt-4o-mini",
    "gemini-2.5-flash": "gemini-2-5-flash",
    "gemini-2-5-flash": "gemini-2-5-flash",
}


class ProfileCache:
    """File-based cache for model register sensitivity profiles."""

    def __init__(self, cache_dir: Path | None = None) -> None:
        self._cache_dir = cache_dir or Path.home() / ".transmogrifier" / "profiles"
        self._memory: dict[str, ModelProfile] = {}

    def get(self, model_name: str) -> ModelProfile | None:
        canonical = _ALIASES.get(model_name, model_name)

        if canonical in self._memory:
            profile = self._memory[canonical]
            if not profile.is_expired:
                return profile

        # Check file cache
        profile = self._load_file(canonical)
        if profile and not profile.is_expired:
            self._memory[canonical] = profile
            return profile

        # Fall back to pre-seeded
        if canonical in _PRESEEDED:
            return _PRESEEDED[canonical]

        # Try partial match on pre-seeded keys
        for key, profile in _PRESEEDED.items():
            if key in canonical or canonical in key:
                return profile

        return None

    def put(self, profile: ModelProfile) -> Path:
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        filename = f"{profile.model_name}.json"
        path = self._cache_dir / filename
        path.write_text(profile.model_dump_json(indent=2))
        self._memory[profile.model_name] = profile
        return path

    def invalidate(self, model_name: str) -> bool:
        canonical = _ALIASES.get(model_name, model_name)
        self._memory.pop(canonical, None)
        path = self._cache_dir / f"{canonical}.json"
        if path.exists():
            path.unlink()
            return True
        return False

    def list_profiles(self) -> list[ModelProfile]:
        profiles = list(_PRESEEDED.values())
        if self._cache_dir.exists():
            for f in self._cache_dir.glob("*.json"):
                try:
                    data = json.loads(f.read_text())
                    p = ModelProfile(**data)
                    if not any(ep.model_name == p.model_name for ep in profiles):
                        profiles.append(p)
                except Exception:
                    continue
        return profiles

    def _load_file(self, model_name: str) -> ModelProfile | None:
        path = self._cache_dir / f"{model_name}.json"
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text())
            return ModelProfile(**data)
        except Exception as e:
            logger.debug("Failed to load profile %s: %s", path, e)
            return None
