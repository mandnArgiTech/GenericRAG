"""RFC 4443 ICMPv6 — general format and Destination Unreachable as atomic chunks.

Author: deviprasad
"""
from __future__ import annotations

from typing import Dict, List, Tuple

from ingest.chunking.rfc.rfc_plugin_struct_prefilter import struct_prefilter_parse, struct_prefilter_strip
from ingest.chunking.rfc.rfc_registry import RfcSectionParser, register_rfc_parser


def _atomic_struct_chunk(body: str, hierarchical_field: str) -> List[Tuple[str, Dict[str, str]]]:
    text = body.strip()
    if len(text) < 60 or "+-+" not in text:
        return []
    meta: Dict[str, str] = {"chunk_type": "struct_definition", "hierarchical_field": hierarchical_field}
    return [(text, meta)]


def parse_rfc4443_s21(body: str, sec: str) -> List[Tuple[str, Dict[str, str]]]:
    return _atomic_struct_chunk(body, "ICMPv6 general message format (Section 2.1)")


def strip_rfc4443_s21(_body: str, _sec: str) -> str:
    return ""


def parse_rfc4443_s31(body: str, sec: str) -> List[Tuple[str, Dict[str, str]]]:
    return _atomic_struct_chunk(body, "ICMPv6 Destination Unreachable (Section 3.1)")


def strip_rfc4443_s31(_body: str, _sec: str) -> str:
    return ""


register_rfc_parser(
    RfcSectionParser(
        rfc_number="4443",
        section_pattern=r"^2\.1$",
        parse_fn=parse_rfc4443_s21,
        strip_fn=strip_rfc4443_s21,
        priority=8,
    )
)
register_rfc_parser(
    RfcSectionParser(
        rfc_number="4443",
        section_pattern=r"^3\.1$",
        parse_fn=parse_rfc4443_s31,
        strip_fn=strip_rfc4443_s31,
        priority=8,
    )
)
register_rfc_parser(
    RfcSectionParser(
        rfc_number="4443",
        section_pattern="*",
        parse_fn=struct_prefilter_parse,
        strip_fn=struct_prefilter_strip,
        priority=2,
    )
)
