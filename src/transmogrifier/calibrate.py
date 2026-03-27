"""Calibration benchmark runner. Stub — Phase 8."""

from __future__ import annotations


class CalibrationRunner:
    """Runs the register sensitivity benchmark for a model. Requires API backend."""

    def __init__(self, backend=None, profile_cache=None):
        self._backend = backend
        self._cache = profile_cache

    async def run(self, model_name: str, model_version: str = "", provider: str = ""):
        raise NotImplementedError("Calibration requires Phase 8 implementation")
