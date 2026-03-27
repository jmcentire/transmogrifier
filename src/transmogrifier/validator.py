"""Semantic equivalence validation (optional, requires sentence-transformers)."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


class SemanticValidator:
    """Embedding-based similarity check between input and output."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self._model_name = model_name
        self._model = None

    def _load(self):
        if self._model is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self._model_name)
        except ImportError:
            logger.warning("sentence-transformers not installed; semantic validation disabled")
            self._model = False

    def validate(self, input_text: str, output_text: str) -> float | None:
        self._load()
        if self._model is False:
            return None
        import numpy as np
        embeddings = self._model.encode([input_text, output_text], normalize_embeddings=True)
        return float(np.dot(embeddings[0], embeddings[1]))

    def is_valid(self, input_text: str, output_text: str, threshold: float = 0.95) -> bool | None:
        sim = self.validate(input_text, output_text)
        if sim is None:
            return None
        return sim >= threshold
