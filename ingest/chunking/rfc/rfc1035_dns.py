"""RFC 1035 DNS — header bit diagram (§4.1.1) as one atomic chunk.

Author: deviprasad
"""
from __future__ import annotations

from typing import Dict, List, Tuple

from ingest.chunking.rfc.rfc_plugin_struct_prefilter import struct_prefilter_parse, struct_prefilter_strip
from ingest.chunking.rfc.rfc_registry import RfcSectionParser, register_rfc_parser


def parse_rfc1035_section411_atomic(body: str, sec: str) -> List[Tuple[str, Dict[str, str]]]:
    text = body.strip()
    if len(text) < 80 or "+--+" not in text:
        return []
    meta: Dict[str, str] = {
        "chunk_type": "struct_definition",
        "hierarchical_field": "DNS message header (Section 4.1.1)",
    }
    return [(text, meta)]


def strip_rfc1035_section411_atomic(_body: str, _sec: str) -> str:
    return ""


register_rfc_parser(
    RfcSectionParser(
        rfc_number="1035",
        section_pattern=r"^4\.1\.1$",
        parse_fn=parse_rfc1035_section411_atomic,
        strip_fn=strip_rfc1035_section411_atomic,
        priority=8,
    )
)
register_rfc_parser(
    RfcSectionParser(
        rfc_number="1035",
        section_pattern=r"^3\.2|^4\.1",
        parse_fn=struct_prefilter_parse,
        strip_fn=struct_prefilter_strip,
        priority=3,
    )
)
