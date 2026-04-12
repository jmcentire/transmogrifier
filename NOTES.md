# Transmogrifier Notes

## Validator removal (2026-04-11)

The optional `validation` extra and `src/transmogrifier/validator.py` were removed because `sentence-transformers` transitively pulls `torch` and `scikit-learn`, whose macOS PyPI wheels ship mutually-incompatible `libomp.dylib` install names and cause the OpenMP runtime to call `abort()` when both load into one Python process. This made `pip install "transmogrifier[validation]"` unsafe on fresh macOS machines.

The validator was a single-purpose optional feature: embedding-similarity check (>0.95 threshold) on translation output to catch semantic drift during Level 3 LLM translation. Nothing else in transmogrifier depended on it, and kindex never called it.

## Future direction: Voyage-based replacement

The cleanest re-introduction is a new optional extra `validation-voyage` that uses Voyage AI's embeddings API instead of a local sentence-transformers model. Voyage is Anthropic's officially recommended embeddings provider, ships as a pure-Python HTTP client (no native deps, no libomp drama), and has a generous free tier (200M tokens on `voyage-3.5`) that effectively makes the validator free for any realistic use.

Sketch of the replacement:

```python
import os, voyageai

class SemanticValidator:
    def __init__(self, model: str = "voyage-3.5"):
        self._client = voyageai.Client()  # reads VOYAGE_API_KEY
        self._model = model

    def validate(self, input_text: str, output_text: str) -> float | None:
        try:
            r = self._client.embed(
                [input_text, output_text],
                model=self._model,
                input_type="document",
            )
        except Exception:
            return None
        a, b = r.embeddings
        return sum(x * y for x, y in zip(a, b))  # voyage vectors are unit-normalized

    def is_valid(self, input_text: str, output_text: str, threshold: float = 0.95) -> bool | None:
        sim = self.validate(input_text, output_text)
        return None if sim is None else sim >= threshold
```

Pricing (as of April 2026): `voyage-3.5` at $0.06 per 1M tokens, 200M free tier per account. Each validation embeds two strings of typically 50–400 tokens, so one call is roughly $0.00002 and the free tier covers ~500,000 validations before billing starts.

`VOYAGE_API_KEY` is already set in `~/.profile` on this machine awaiting use.

See kindex memory node `voyage-transmogrifier-validator-future` for broader context and rationale.
