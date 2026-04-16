"""RFC 826 ARP — packet layout between ``Packet format`` and ``Packet Generation``.

RFC 826 has no numbered sections; the anchor parser uses ``*`` on the full body.

Author: deviprasad
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from ingest.chunking.rfc.rfc_anchor_utils import chunk_from_span, strip_one, span_from_to
from ingest.chunking.rfc.rfc_plugin_struct_prefilter import struct_prefilter_parse, struct_prefilter_strip
from ingest.chunking.rfc.rfc_registry import RfcSectionParser, register_rfc_parser


def _find_packet_format_span(body: str) -> Optional[Tuple[int, int]]:
    return span_from_to(
        body,
        r"(?m)^Packet format:\s*$",
        r"(?m)^Packet Generation:\s*$",
    )


def parse_rfc826_packet_format(body: str, sec: str) -> List[Tuple[str, Dict[str, str]]]:
    sp = _find_packet_format_span(body)
    return chunk_from_span(
        body,
        sp,
        chunk_type="struct_definition",
        hierarchical_field="ARP / Ethernet packet format (RFC 826)",
    )


def strip_rfc826_packet_format(body: str, sec: str) -> str:
    return strip_one(body, _find_packet_format_span(body))


register_rfc_parser(
    RfcSectionParser(
        rfc_number="826",
        section_pattern="*",
        parse_fn=parse_rfc826_packet_format,
        strip_fn=strip_rfc826_packet_format,
        priority=8,
    )
)
register_rfc_parser(
    RfcSectionParser(
        rfc_number="826",
        section_pattern="*",
        parse_fn=struct_prefilter_parse,
        strip_fn=struct_prefilter_strip,
        priority=2,
    )
)
