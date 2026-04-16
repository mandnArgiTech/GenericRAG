"""RFC 9293 TCP — header figure (§3.1) as one chunk; struct prefilter on later §3.x.

Author: deviprasad
"""
from __future__ import annotations

from typing import Dict, List, Tuple

from ingest.chunking.rfc.rfc_plugin_struct_prefilter import struct_prefilter_parse, struct_prefilter_strip
from ingest.chunking.rfc.rfc_registry import RfcSectionParser, register_rfc_parser


def parse_rfc9293_section31_atomic(body: str, sec: str) -> List[Tuple[str, Dict[str, str]]]:
    text = body.strip()
    if len(text) < 120 or "+-+" not in text:
        return []
    meta: Dict[str, str] = {
        "chunk_type": "struct_definition",
        "hierarchical_field": "TCP header format (Section 3.1)",
    }
    return [(text, meta)]


def strip_rfc9293_section31_atomic(_body: str, _sec: str) -> str:
    return ""


register_rfc_parser(
    RfcSectionParser(
        rfc_number="9293",
        section_pattern=r"^3\.1$",
        parse_fn=parse_rfc9293_section31_atomic,
        strip_fn=strip_rfc9293_section31_atomic,
        priority=8,
    )
)
register_rfc_parser(
    RfcSectionParser(
        rfc_number="9293",
        section_pattern=r"^3\.(2|3|4|5|6|7|8|9|10)(\D|$)",
        parse_fn=struct_prefilter_parse,
        strip_fn=struct_prefilter_strip,
        priority=3,
    )
)
