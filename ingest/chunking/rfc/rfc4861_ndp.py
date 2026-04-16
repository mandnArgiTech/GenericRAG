"""RFC 4861 NDP — Neighbor Solicitation / Advertisement figures as atomic chunks.

Author: deviprasad
"""
from __future__ import annotations

from typing import Dict, List, Tuple

from ingest.chunking.rfc.rfc_plugin_struct_prefilter import struct_prefilter_parse, struct_prefilter_strip
from ingest.chunking.rfc.rfc_registry import RfcSectionParser, register_rfc_parser


def _atomic_if_icmp_diagram(body: str, hierarchical_field: str) -> List[Tuple[str, Dict[str, str]]]:
    text = body.strip()
    if len(text) < 100 or "+-+" not in text:
        return []
    meta: Dict[str, str] = {"chunk_type": "struct_definition", "hierarchical_field": hierarchical_field}
    return [(text, meta)]


def parse_rfc4861_s43(body: str, sec: str) -> List[Tuple[str, Dict[str, str]]]:
    return _atomic_if_icmp_diagram(body, "Neighbor Solicitation message format (Section 4.3)")


def strip_rfc4861_s43(_body: str, _sec: str) -> str:
    return ""


def parse_rfc4861_s44(body: str, sec: str) -> List[Tuple[str, Dict[str, str]]]:
    return _atomic_if_icmp_diagram(body, "Neighbor Advertisement message format (Section 4.4)")


def strip_rfc4861_s44(_body: str, _sec: str) -> str:
    return ""


register_rfc_parser(
    RfcSectionParser(
        rfc_number="4861",
        section_pattern=r"^4\.3$",
        parse_fn=parse_rfc4861_s43,
        strip_fn=strip_rfc4861_s43,
        priority=8,
    )
)
register_rfc_parser(
    RfcSectionParser(
        rfc_number="4861",
        section_pattern=r"^4\.4$",
        parse_fn=parse_rfc4861_s44,
        strip_fn=strip_rfc4861_s44,
        priority=8,
    )
)
register_rfc_parser(
    RfcSectionParser(
        rfc_number="4861",
        section_pattern=r"^4\.",
        parse_fn=struct_prefilter_parse,
        strip_fn=struct_prefilter_strip,
        priority=3,
    )
)
