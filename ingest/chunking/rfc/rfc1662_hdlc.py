"""RFC 1662 PPP HDLC-like framing — dedicated anchors for frame, escapes, ACCM.

Unlike generic ``struct_prefilter`` alone, this module pulls RFC 1662-specific
blocks (§3.1 frame figure + field prose, §4.2 escape examples, §7.1 ACCM option)
into atomic chunks so they are not split by paragraph logic.

Author: deviprasad
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

from ingest.chunking.rfc.rfc_registry import RfcSectionParser, register_rfc_parser


def _span(body: str, start: int, end: int) -> Tuple[str, Tuple[int, int]]:
    return body[start:end].strip(), (start, end)


def _find_frame_format_span(body: str) -> Optional[Tuple[int, int]]:
    """§3.1: ASCII frame figure through prose before §3.2."""
    m_start = re.search(
        r"(?ms)^\s*\+[-+]+\+[-+]+\+[-+]+\+\s*\n\s*\|\s*Flag\s*\|",
        body,
    )
    m_flag = re.search(r"(?m)^\s*\|\s*Flag\s*\|\s*Address\s*\|\s*Control\s*\|", body)
    if not m_flag:
        return None
    start = m_start.start() if m_start else m_flag.start()
    m_end = re.search(r"(?m)^\s*3\.2\.\s+Modification\b", body[m_flag.start() :])
    if m_end:
        end = m_flag.start() + m_end.start()
    else:
        end = len(body)
    return (start, end)


def _find_escape_examples_span(body: str) -> Optional[Tuple[int, int]]:
    """§4.2: hex encoding examples block before §4.3."""
    m0 = re.search(r"(?m)^\s*0x7e\s+is\s+encoded\s+as\s+0x7d", body)
    if not m0:
        return None
    m1 = re.search(r"(?m)^\s*4\.3\.\s+Invalid\b", body[m0.start() :])
    end = m0.start() + m1.start() if m1 else len(body)
    return (m0.start(), end)


def _find_accm_option_span(body: str) -> Optional[Tuple[int, int]]:
    """§7.1: ACCM configuration option summary line through diagram and field definitions."""
    m0 = re.search(
        r"(?m)^\s*A\s+summary\s+of\s+the\s+Async-Control-Character-Map\s+Configuration\s+Option",
        body,
    )
    if not m0:
        m0 = re.search(r"(?m)^\s*0\s+1\s+2\s+3\s+4\s+5\s+6\s+7\s+8\s+9\s+0\s+1\s+2\s+3", body)
    if not m0:
        return None
    tail = body[m0.start() :]
    m1 = re.search(r"(?m)^\s*A\.\s+Recommended\b", tail)
    rel_end = m1.start() if m1 else len(tail)
    end = m0.start() + rel_end
    return (m0.start(), min(end, len(body)))


def _strip_spans(body: str, spans: List[Tuple[int, int]]) -> str:
    if not spans:
        return body
    spans = sorted(spans)
    merged: List[Tuple[int, int]] = []
    for s, e in spans:
        if not merged or s > merged[-1][1] + 1:
            merged.append((s, e))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
    t = body
    for s, e in reversed(merged):
        s = max(0, min(s, len(t)))
        e = max(s, min(e, len(t)))
        t = t[:s].rstrip() + "\n\n" + t[e:].lstrip()
    return re.sub(r"\n{3,}", "\n\n", t).strip()


def parse_rfc1662_section31_frame(body: str, sec: str) -> List[Tuple[str, Dict[str, str]]]:
    sp = _find_frame_format_span(body)
    if not sp:
        return []
    s, e = sp
    chunk, _ = _span(body, s, e)
    if len(chunk) < 40:
        return []
    meta: Dict[str, str] = {
        "chunk_type": "struct_definition",
        "hierarchical_field": "PPP HDLC-like frame (Flag, Address, Control, Protocol, FCS)",
    }
    return [(chunk, meta)]


def strip_rfc1662_section31_frame(body: str, sec: str) -> str:
    sp = _find_frame_format_span(body)
    return _strip_spans(body, [sp]) if sp else body


def parse_rfc1662_section42_escapes(body: str, sec: str) -> List[Tuple[str, Dict[str, str]]]:
    sp = _find_escape_examples_span(body)
    if not sp:
        return []
    s, e = sp
    chunk, _ = _span(body, s, e)
    if len(chunk) < 20:
        return []
    meta: Dict[str, str] = {
        "chunk_type": "constants",
        "hierarchical_field": "Control Escape encodings (0x7e, 0x7d, XON/XOFF)",
    }
    return [(chunk, meta)]


def strip_rfc1662_section42_escapes(body: str, sec: str) -> str:
    sp = _find_escape_examples_span(body)
    return _strip_spans(body, [sp]) if sp else body


def parse_rfc1662_section71_accm(body: str, sec: str) -> List[Tuple[str, Dict[str, str]]]:
    sp = _find_accm_option_span(body)
    if not sp:
        return []
    s, e = sp
    chunk, _ = _span(body, s, e)
    if len(chunk) < 40:
        return []
    meta: Dict[str, str] = {
        "chunk_type": "struct_definition",
        "hierarchical_field": "Async-Control-Character-Map (ACCM) option",
    }
    return [(chunk, meta)]


def strip_rfc1662_section71_accm(body: str, sec: str) -> str:
    sp = _find_accm_option_span(body)
    return _strip_spans(body, [sp]) if sp else body


register_rfc_parser(
    RfcSectionParser(
        rfc_number="1662",
        section_pattern=r"^3\.1$",
        parse_fn=parse_rfc1662_section31_frame,
        strip_fn=strip_rfc1662_section31_frame,
        priority=8,
    )
)
register_rfc_parser(
    RfcSectionParser(
        rfc_number="1662",
        section_pattern=r"^4\.2$",
        parse_fn=parse_rfc1662_section42_escapes,
        strip_fn=strip_rfc1662_section42_escapes,
        priority=8,
    )
)
register_rfc_parser(
    RfcSectionParser(
        rfc_number="1662",
        section_pattern=r"^7\.1$",
        parse_fn=parse_rfc1662_section71_accm,
        strip_fn=strip_rfc1662_section71_accm,
        priority=8,
    )
)
