"""Shared helpers: pre-run struct parsers on matching sections (strip before generic pass).

Author: deviprasad
"""
from __future__ import annotations

from typing import Dict, List, Tuple

from ingest.chunking.rfc.rfc_struct_parser import parse_all_struct_regions, strip_spans


def struct_prefilter_parse(body: str, sec: str) -> List[Tuple[str, Dict[str, str]]]:
    chunks, _ = parse_all_struct_regions(body)
    return chunks


def struct_prefilter_strip(body: str, sec: str) -> str:
    _, spans = parse_all_struct_regions(body)
    return strip_spans(body, spans)
