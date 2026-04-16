"""RFC 2131 DHCP — message figure + client state diagram as anchored chunks.

Author: deviprasad
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from ingest.chunking.rfc.rfc_anchor_utils import chunk_from_span, strip_one, span_from_to
from ingest.chunking.rfc.rfc_plugin_struct_prefilter import struct_prefilter_parse, struct_prefilter_strip
from ingest.chunking.rfc.rfc_registry import RfcSectionParser, register_rfc_parser


def _span_figure1(body: str) -> Optional[Tuple[int, int]]:
    return span_from_to(
        body,
        r"(?m)^\s*\|\s*op \(1\)\s*\|",
        r"(?m)^\s*Table 1:\s+Description of fields in a DHCP message\s*$",
    )


def parse_rfc2131_figure1(body: str, sec: str) -> List[Tuple[str, Dict[str, str]]]:
    sp = _span_figure1(body)
    return chunk_from_span(
        body,
        sp,
        chunk_type="struct_definition",
        hierarchical_field="DHCP message format (Figure 1, Table 1)",
    )


def strip_rfc2131_figure1(body: str, sec: str) -> str:
    return strip_one(body, _span_figure1(body))


def _span_figure5(body: str) -> Optional[Tuple[int, int]]:
    # Same sentence continues on the line ("...client.  A client..."); do not anchor EOL.
    return span_from_to(
        body,
        r"(?m)^\s*Figure 5 gives a state-transition diagram for a DHCP client\.",
        r"(?m)^4\.4\.1\s+",
    )


def parse_rfc2131_figure5_fsm(body: str, sec: str) -> List[Tuple[str, Dict[str, str]]]:
    sp = _span_figure5(body)
    return chunk_from_span(
        body,
        sp,
        chunk_type="struct_definition",
        hierarchical_field="DHCP client state-transition diagram (Figure 5)",
    )


def strip_rfc2131_figure5_fsm(body: str, sec: str) -> str:
    return strip_one(body, _span_figure5(body))


register_rfc_parser(
    RfcSectionParser(
        rfc_number="2131",
        section_pattern=r"^2$",
        parse_fn=parse_rfc2131_figure1,
        strip_fn=strip_rfc2131_figure1,
        priority=8,
    )
)
register_rfc_parser(
    RfcSectionParser(
        rfc_number="2131",
        section_pattern=r"^4\.4$",
        parse_fn=parse_rfc2131_figure5_fsm,
        strip_fn=strip_rfc2131_figure5_fsm,
        priority=8,
    )
)
register_rfc_parser(
    RfcSectionParser(
        rfc_number="2131",
        section_pattern=r"^2\.|^4\.4",
        parse_fn=struct_prefilter_parse,
        strip_fn=struct_prefilter_strip,
        priority=3,
    )
)
