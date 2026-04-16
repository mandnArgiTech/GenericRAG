"""RFC 1332 IPCP — configuration option figures as atomic section chunks.

Author: deviprasad
"""
from __future__ import annotations

from typing import Dict, List, Tuple

from ingest.chunking.rfc.rfc_plugin_struct_prefilter import struct_prefilter_parse, struct_prefilter_strip
from ingest.chunking.rfc.rfc_registry import RfcSectionParser, register_rfc_parser


def _atomic_if_diagram(
    body: str,
    *,
    hierarchical_field: str,
    min_len: int = 80,
) -> List[Tuple[str, Dict[str, str]]]:
    text = body.strip()
    if len(text) < min_len or "+-+" not in text:
        return []
    meta: Dict[str, str] = {
        "chunk_type": "struct_definition",
        "hierarchical_field": hierarchical_field,
    }
    return [(text, meta)]


def parse_rfc1332_s32(body: str, sec: str) -> List[Tuple[str, Dict[str, str]]]:
    return _atomic_if_diagram(
        body,
        hierarchical_field="IPCP IP-Compression-Protocol option (Section 3.2)",
    )


def strip_rfc1332_s32(_body: str, _sec: str) -> str:
    return ""


def parse_rfc1332_s33(body: str, sec: str) -> List[Tuple[str, Dict[str, str]]]:
    return _atomic_if_diagram(
        body,
        hierarchical_field="IPCP IP-Address option (Section 3.3)",
    )


def strip_rfc1332_s33(_body: str, _sec: str) -> str:
    return ""


def parse_rfc1332_s41(body: str, sec: str) -> List[Tuple[str, Dict[str, str]]]:
    return _atomic_if_diagram(
        body,
        hierarchical_field="Van Jacobson compression option format (Section 4.1)",
    )


def strip_rfc1332_s41(_body: str, _sec: str) -> str:
    return ""


for _pat, _p, _s in (
    (r"^3\.2$", parse_rfc1332_s32, strip_rfc1332_s32),
    (r"^3\.3$", parse_rfc1332_s33, strip_rfc1332_s33),
    (r"^4\.1$", parse_rfc1332_s41, strip_rfc1332_s41),
):
    register_rfc_parser(
        RfcSectionParser(
            rfc_number="1332",
            section_pattern=_pat,
            parse_fn=_p,
            strip_fn=_s,
            priority=8,
        )
    )

register_rfc_parser(
    RfcSectionParser(
        rfc_number="1332",
        section_pattern=r"^3\.",
        parse_fn=struct_prefilter_parse,
        strip_fn=struct_prefilter_strip,
        priority=3,
    )
)
