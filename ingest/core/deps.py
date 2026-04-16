"""Third-party and optional dependency imports for ingestion.

Author: deviprasad
"""
from __future__ import annotations

from typing import Any, Dict

try:
    import chromadb
except ImportError as exc:  # pragma: no cover
    raise SystemExit("chromadb is required") from exc

try:
    import requests
except ImportError:  # pragma: no cover
    requests = None  # type: ignore

try:
    import aiohttp  # type: ignore
except ImportError:  # pragma: no cover
    aiohttp = None  # type: ignore

try:
    import pathspec  # type: ignore
except ImportError:  # pragma: no cover
    pathspec = None  # type: ignore

try:
    from bs4 import BeautifulSoup
except ImportError:  # pragma: no cover
    BeautifulSoup = None  # type: ignore

try:
    from sanitizer import sanitize as sanitize_pii
except ImportError:  # pragma: no cover
    sanitize_pii = None  # type: ignore

# ---------------------------------------------------------------------------
# Optional tree-sitter
# ---------------------------------------------------------------------------
_TS_LANG: Dict[str, Any] = {}


def _load_ts_language(name: str, mod_name: str) -> Any:
    if name in _TS_LANG:
        return _TS_LANG[name]
    try:
        mod = __import__(mod_name, fromlist=["language"])
        from tree_sitter import Language as TSLanguage  # type: ignore

        lang = TSLanguage(getattr(mod, "language")())
        _TS_LANG[name] = lang
        return lang
    except Exception:  # pragma: no cover
        return None


def _ts_parser_for(lang_name: str, mod_name: str):
    from tree_sitter import Parser  # type: ignore

    lang = _load_ts_language(lang_name, mod_name)
    if lang is None:
        return None  # pragma: no cover
    try:
        return Parser(lang)  # tree-sitter-python >=0.21
    except TypeError:  # pragma: no cover
        p = Parser()
        if hasattr(p, "set_language"):
            p.set_language(lang)
        else:
            p.language = lang  # type: ignore[attr-defined]
        return p
