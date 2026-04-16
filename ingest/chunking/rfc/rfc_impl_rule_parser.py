"""Normative requirement lines (RFC 2119) as ``implementation_rule`` chunks.

Author: deviprasad
"""
from __future__ import annotations

import re
from typing import Dict, List, Tuple

from ingest.chunking.rfc.rfc_struct_parser import strip_spans

_KEYWORD = re.compile(
    r"\b(MUST\s+NOT|MUST|SHALL\s+NOT|SHALL|SHOULD\s+NOT|SHOULD|"
    r"RECOMMENDED|NOT\s+RECOMMENDED|MAY|REQUIRED|OPTIONAL)\b",
    re.I,
)

_PARA_SPLIT = re.compile(r"(\n(?:\s*\n)+)")


def parse_implementation_rules(body: str) -> List[Tuple[str, Dict[str, str], Tuple[int, int]]]:
    """One chunk per paragraph that contains a normative keyword; spans are index-accurate."""
    if not body.strip():
        return []
    tokens = _PARA_SPLIT.split(body)
    out: List[Tuple[str, Dict[str, str], Tuple[int, int]]] = []
    idx = 0
    for j in range(0, len(tokens), 2):
        p = tokens[j]
        sep = tokens[j + 1] if j + 1 < len(tokens) else ""
        span_start = idx
        span_end = idx + len(p)
        idx = span_end + len(sep)
        p_stripped = p.strip()
        if len(p_stripped) < 12 or not _KEYWORD.search(p_stripped):
            continue
        inner_start = p.index(p_stripped) if p_stripped else 0
        inner_end = inner_start + len(p_stripped)
        span = (span_start + inner_start, span_start + inner_end)
        out.append((p_stripped, {"chunk_type": "implementation_rule"}, span))
    return out


def parse_all_impl_regions(body: str) -> Tuple[List[Tuple[str, Dict[str, str]]], List[Tuple[int, int]]]:
    chunks: List[Tuple[str, Dict[str, str]]] = []
    spans: List[Tuple[int, int]] = []
    for ch, meta, sp in parse_implementation_rules(body):
        chunks.append((ch, meta))
        spans.append(sp)
    return chunks, spans


def strip_impl_spans(body: str, spans: List[Tuple[int, int]]) -> str:
    return strip_spans(body, spans)
