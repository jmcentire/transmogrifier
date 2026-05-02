"""Optional Kindex integration. Graceful degradation when kindex not installed."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_store = None
_checked = False


def is_available() -> bool:
    global _checked
    if _checked:
        return _store is not None
    _checked = True
    return _init_store()


def _init_store() -> bool:
    global _store
    try:
        from kindex.config import load_config
        from kindex.store import Store
        config = load_config()
        _store = Store(config)
        return True
    except Exception as e:
        logger.debug("Kindex not available: %s", e)
        return False


def close() -> None:
    global _store, _checked
    if _store is not None:
        _store.close()
    _store = None
    _checked = False
