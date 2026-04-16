"""Shared span helpers for RFC-specific anchor parsers (see ``rfcNNNN_*.py``).

Author: deviprasad
"""
from __future__ import annotations

import re
from typing import Callable, Dict, List, Optional, Tuple

from ingest.chunking.rfc.rfc_struct_parser import strip_spans


def span_from_to(
    body: str,
    start_re: str,
    end_re: str,
    *,
    flags: int = re.MULTILINE,
) -> Optional[Tuple[int, int]]:
    """
    Return ``(start, end)`` with ``body[start:end]`` from first ``start_re`` match
    up to (but not including) the first ``end_re`` match after that, else ``len(body)``.
    """
    m0 = re.search(start_re, body, flags)
    if not m0:
        return None
    tail = body[m0.end() :]
    m1 = re.search(end_re, tail, flags)
    end = m0.end() + m1.start() if m1 else len(body)
    return (m0.start(), end)


def strip_one(body: str, span: Optional[Tuple[int, int]]) -> str:
    return strip_spans(body, [span]) if span else body


def chunk_from_span(
    body: str,
    span: Optional[Tuple[int, int]],
    *,
    chunk_type: str,
    hierarchical_field: str,
) -> List[Tuple[str, Dict[str, str]]]:
    if not span:
        return []
    s, e = span
    chunk = body[s:e].strip()
    if len(chunk) < 24:
        return []
    meta: Dict[str, str] = {"chunk_type": chunk_type, "hierarchical_field": hierarchical_field}
    return [(chunk, meta)]


def strip_from_spans(body: str, spans: List[Optional[Tuple[int, int]]]) -> str:
    good = [sp for sp in spans if sp]
    return strip_spans(body, good) if good else body
