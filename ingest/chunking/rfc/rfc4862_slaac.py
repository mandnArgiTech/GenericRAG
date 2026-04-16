"""RFC 4862 SLAAC — Duplicate Address Detection section as one normative chunk.

Author: deviprasad
"""
from __future__ import annotations

from typing import Dict, List, Tuple

from ingest.chunking.rfc.rfc_plugin_struct_prefilter import struct_prefilter_parse, struct_prefilter_strip
from ingest.chunking.rfc.rfc_registry import RfcSectionParser, register_rfc_parser


def parse_rfc4862_section54_atomic(body: str, sec: str) -> List[Tuple[str, Dict[str, str]]]:
    text = body.strip()
    if len(text) < 200:
        return []
    meta: Dict[str, str] = {
        "chunk_type": "implementation_rule",
        "hierarchical_field": "Duplicate Address Detection (Section 5.4)",
    }
    return [(text, meta)]


def strip_rfc4862_section54_atomic(_body: str, _sec: str) -> str:
    return ""


register_rfc_parser(
    RfcSectionParser(
        rfc_number="4862",
        section_pattern=r"^5\.4$",
        parse_fn=parse_rfc4862_section54_atomic,
        strip_fn=strip_rfc4862_section54_atomic,
        priority=8,
    )
)
register_rfc_parser(
    RfcSectionParser(
        rfc_number="4862",
        section_pattern=r"^5\.",
        parse_fn=struct_prefilter_parse,
        strip_fn=struct_prefilter_strip,
        priority=3,
    )
)
