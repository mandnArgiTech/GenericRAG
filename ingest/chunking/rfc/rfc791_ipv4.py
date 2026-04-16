"""RFC 791 IPv4 — diagram-heavy sections (header, options, fragmentation).

Section ``3.1`` is emitted as one atomic chunk (header figure + field prose)
so paragraph splitting does not tear the specification apart; ``3.2`` keeps
the generic struct prefilter for remaining diagrams.

Author: deviprasad
"""
from __future__ import annotations

from typing import Dict, List, Tuple

from ingest.chunking.rfc.rfc_plugin_struct_prefilter import struct_prefilter_parse, struct_prefilter_strip
from ingest.chunking.rfc.rfc_registry import RfcSectionParser, register_rfc_parser


def parse_rfc791_section31_atomic(body: str, sec: str) -> List[Tuple[str, Dict[str, str]]]:
    text = body.strip()
    if len(text) < 80 or "+-+" not in text:
        return []
    meta: Dict[str, str] = {
        "chunk_type": "struct_definition",
        "hierarchical_field": "IPv4 internet header (Section 3.1)",
    }
    return [(text, meta)]


def strip_rfc791_section31_atomic(_body: str, _sec: str) -> str:
    return ""


register_rfc_parser(
    RfcSectionParser(
        rfc_number="791",
        section_pattern=r"^3\.1$",
        parse_fn=parse_rfc791_section31_atomic,
        strip_fn=strip_rfc791_section31_atomic,
        priority=8,
    )
)
register_rfc_parser(
    RfcSectionParser(
        rfc_number="791",
        section_pattern=r"^3\.2(\D|$)",
        parse_fn=struct_prefilter_parse,
        strip_fn=struct_prefilter_strip,
        priority=3,
    )
)
