"""RFC 768 UDP — anchor span for header + pseudo-header block.

RFC 768 has no numbered section headings; the priority-8 ``*`` parser runs on
the full document when :func:`chunk_rfc` finds no section list.

Author: deviprasad
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

from ingest.chunking.rfc.rfc_anchor_utils import chunk_from_span, strip_one, span_from_to
from ingest.chunking.rfc.rfc_plugin_struct_prefilter import struct_prefilter_parse, struct_prefilter_strip
from ingest.chunking.rfc.rfc_registry import RfcSectionParser, register_rfc_parser


def _find_udp_format_span(body: str) -> Optional[Tuple[int, int]]:
    if not re.search(r"(?m)^Format\s*$", body):
        return None
    return span_from_to(body, r"(?m)^Format\s*$", r"(?m)^User Interface\s*$")


def parse_rfc768_udp_diagrams(body: str, sec: str) -> List[Tuple[str, Dict[str, str]]]:
    sp = _find_udp_format_span(body)
    return chunk_from_span(
        body,
        sp,
        chunk_type="struct_definition",
        hierarchical_field="UDP header and pseudo-header (RFC 768)",
    )


def strip_rfc768_udp_diagrams(body: str, sec: str) -> str:
    return strip_one(body, _find_udp_format_span(body))


register_rfc_parser(
    RfcSectionParser(
        rfc_number="768",
        section_pattern="*",
        parse_fn=parse_rfc768_udp_diagrams,
        strip_fn=strip_rfc768_udp_diagrams,
        priority=8,
    )
)
register_rfc_parser(
    RfcSectionParser(
        rfc_number="768",
        section_pattern="*",
        parse_fn=struct_prefilter_parse,
        strip_fn=struct_prefilter_strip,
        priority=2,
    )
)
