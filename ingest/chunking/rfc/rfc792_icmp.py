"""RFC 792 ICMP — anchor spans for common message figures.

RFC 792 has no numbered sections; parsers use ``*`` on the full document body.

Author: deviprasad
"""
from __future__ import annotations

from typing import Dict, List, Tuple

from ingest.chunking.rfc.rfc_anchor_utils import chunk_from_span, span_from_to, strip_from_spans
from ingest.chunking.rfc.rfc_plugin_struct_prefilter import struct_prefilter_parse, struct_prefilter_strip
from ingest.chunking.rfc.rfc_registry import RfcSectionParser, register_rfc_parser

_RFC792_ANCHORS: Tuple[Tuple[str, str, str], ...] = (
    (
        r"(?m)^\s*Destination Unreachable Message\s*$",
        r"(?m)^\s*Time Exceeded Message\s*$",
        "ICMP Destination Unreachable (type 3)",
    ),
    (
        r"(?m)^\s*Echo or Echo Reply Message\s*$",
        r"(?m)^\s*Timestamp or Timestamp Reply Message\s*$",
        "ICMP Echo / Echo Reply (types 0, 8)",
    ),
)


def parse_rfc792_diagram_anchors(body: str, sec: str) -> List[Tuple[str, Dict[str, str]]]:
    out: List[Tuple[str, Dict[str, str]]] = []
    for start_re, end_re, hf in _RFC792_ANCHORS:
        sp = span_from_to(body, start_re, end_re)
        out.extend(
            chunk_from_span(body, sp, chunk_type="struct_definition", hierarchical_field=hf)
        )
    return out


def strip_rfc792_diagram_anchors(body: str, sec: str) -> str:
    spans = [span_from_to(body, a[0], a[1]) for a in _RFC792_ANCHORS]
    return strip_from_spans(body, list(spans))


register_rfc_parser(
    RfcSectionParser(
        rfc_number="792",
        section_pattern="*",
        parse_fn=parse_rfc792_diagram_anchors,
        strip_fn=strip_rfc792_diagram_anchors,
        priority=8,
    )
)
register_rfc_parser(
    RfcSectionParser(
        rfc_number="792",
        section_pattern="*",
        parse_fn=struct_prefilter_parse,
        strip_fn=struct_prefilter_strip,
        priority=2,
    )
)
