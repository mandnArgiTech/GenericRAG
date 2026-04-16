"""RFC 8200 IPv6 — base header figure (Section 3) as one chunk; struct on §4.

Author: deviprasad
"""
from __future__ import annotations

from typing import Dict, List, Tuple

from ingest.chunking.rfc.rfc_plugin_struct_prefilter import struct_prefilter_parse, struct_prefilter_strip
from ingest.chunking.rfc.rfc_registry import RfcSectionParser, register_rfc_parser


def parse_rfc8200_section3_atomic(body: str, sec: str) -> List[Tuple[str, Dict[str, str]]]:
    text = body.strip()
    if len(text) < 80 or "+-+" not in text:
        return []
    meta: Dict[str, str] = {
        "chunk_type": "struct_definition",
        "hierarchical_field": "IPv6 base header (Section 3)",
    }
    return [(text, meta)]


def strip_rfc8200_section3_atomic(_body: str, _sec: str) -> str:
    return ""


register_rfc_parser(
    RfcSectionParser(
        rfc_number="8200",
        section_pattern=r"^3$",
        parse_fn=parse_rfc8200_section3_atomic,
        strip_fn=strip_rfc8200_section3_atomic,
        priority=8,
    )
)
register_rfc_parser(
    RfcSectionParser(
        rfc_number="8200",
        section_pattern=r"^(4\.|4$)",
        parse_fn=struct_prefilter_parse,
        strip_fn=struct_prefilter_strip,
        priority=3,
    )
)
