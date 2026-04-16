"""
RFC-specific retrieval metadata and light heuristics.

Generic depagination / diagram logic lives in ``rfc_text_ops``; per-number facts and
topic tags live here so new RFCs (beyond 1661) can be registered without touching the core chunker.


Author: deviprasad
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Pattern, Tuple

_COMPILED_EXTRA: Dict[str, Tuple[Pattern[str], ...]] = {}


@dataclass(frozen=True)
class RfcProfile:
    """Stable facts about a standards-track RFC plaintext for chunk metadata and search."""

    number: str
    short_title: str
    stack_layer: str
    topics: Tuple[str, ...]
    extra_figure_line_patterns: Tuple[str, ...] = ()


def _compiled_for(profile: RfcProfile) -> Tuple[Pattern[str], ...]:
    key = profile.number
    if key not in _COMPILED_EXTRA:
        _COMPILED_EXTRA[key] = tuple(
            re.compile(p, re.MULTILINE) for p in profile.extra_figure_line_patterns
        )
    return _COMPILED_EXTRA[key]


# Curated profiles for high-value protocol texts (extend freely).
RFC_PROFILES: Dict[str, RfcProfile] = {
    "791": RfcProfile(
        number="791",
        short_title="IPv4",
        stack_layer="internet",
        topics=("ipv4", "datagram", "fragmentation", "ttl", "addressing", "icmp"),
        extra_figure_line_patterns=(
            r"^\s*\d+\s+\d+\s+\d+\s+\d+",
            r"^\s*0\s+1\s+2\s+3",
            r"\+-+\+",
        ),
    ),
    "768": RfcProfile(
        number="768",
        short_title="UDP",
        stack_layer="transport",
        topics=("udp", "datagram", "checksum", "multiplexing", "ports"),
        extra_figure_line_patterns=(
            r"^\s*0\s+7\s+8\s+15",
            r"Source\s+Port|Destination\s+Port",
            r"\+-+\+",
        ),
    ),
    "9293": RfcProfile(
        number="9293",
        short_title="TCP (consolidated)",
        stack_layer="transport",
        topics=("tcp", "mss", "ecn", "timestamps", "urgent", "control", "options"),
        extra_figure_line_patterns=(r"^\s*TCP\s+Header", r"\+-+\+", r"^\s*Kind\s+Length"),
    ),
    "1661": RfcProfile(
        number="1661",
        short_title="PPP LCP",
        stack_layer="link",
        topics=("ppp", "lcp", "negotiation", "ncp", "authentication", "link-control"),
        extra_figure_line_patterns=(
            r"^\s*Events\|",
            r"^\|\s+State\s*$",
            r"^\s*State\s+Transition",
        ),
    ),
    "1122": RfcProfile(
        number="1122",
        short_title="Host Requirements",
        stack_layer="host",
        topics=("must", "should", "host", "tcp", "udp", "ip", "icmp", "requirements"),
        extra_figure_line_patterns=(),
    ),
    "1662": RfcProfile(
        number="1662",
        short_title="PPP HDLC framing",
        stack_layer="link",
        topics=("ppp", "hdlc", "framing", "escape", "fcs", "async"),
        extra_figure_line_patterns=(r"\+-+\+", r"Flag|Address|Control"),
    ),
    "1332": RfcProfile(
        number="1332",
        short_title="IPCP",
        stack_layer="link",
        topics=("ppp", "ipcp", "ip", "configuration", "options"),
        extra_figure_line_patterns=(r"\+-+\+", r"Configuration\s+Option"),
    ),
    "826": RfcProfile(
        number="826",
        short_title="ARP",
        stack_layer="internet",
        topics=("arp", "ethernet", "hardware", "protocol", "opcode"),
        extra_figure_line_patterns=(r"\d+\s+bit", r"EtherType|opcode"),
    ),
    "792": RfcProfile(
        number="792",
        short_title="ICMP",
        stack_layer="internet",
        topics=("icmp", "echo", "unreachable", "redirect", "time", "parameter"),
        extra_figure_line_patterns=(r"\+-+\+", r"Type\s+Code"),
    ),
    "2131": RfcProfile(
        number="2131",
        short_title="DHCP",
        stack_layer="application",
        topics=("dhcp", "lease", "discover", "offer", "request", "ack", "bootp"),
        extra_figure_line_patterns=(r"\+-+\+", r"Opcode|Message\s+type"),
    ),
    "1035": RfcProfile(
        number="1035",
        short_title="DNS",
        stack_layer="application",
        topics=("dns", "query", "response", "rr", "name", "compression"),
        extra_figure_line_patterns=(r"\+-+\+", r"QR|OPCODE|RCODE"),
    ),
    "8200": RfcProfile(
        number="8200",
        short_title="IPv6",
        stack_layer="internet",
        topics=("ipv6", "extension", "hop", "routing", "fragment", "next header"),
        extra_figure_line_patterns=(r"\+-+\+", r"Version|Traffic\s+Class|Flow\s+Label"),
    ),
    "4443": RfcProfile(
        number="4443",
        short_title="ICMPv6",
        stack_layer="internet",
        topics=("icmpv6", "neighbor", "echo", "unreachable", "parameter"),
        extra_figure_line_patterns=(r"\+-+\+", r"Type\s+Code"),
    ),
    "4861": RfcProfile(
        number="4861",
        short_title="NDP",
        stack_layer="internet",
        topics=("ndp", "ra", "rs", "na", "ns", "redirect", "neighbor", "icmpv6"),
        extra_figure_line_patterns=(r"\+-+\+", r"Option\s+Type"),
    ),
    "4862": RfcProfile(
        number="4862",
        short_title="SLAAC",
        stack_layer="internet",
        topics=("slaac", "autoconf", "ndp", "router", "prefix", "dad"),
        extra_figure_line_patterns=(r"\+-+\+", r"state|State"),
    ),
}


def get_rfc_profile(rfc_number: str) -> Optional[RfcProfile]:
    if not rfc_number:
        return None
    return RFC_PROFILES.get(str(rfc_number).strip())


def profile_extra_diagram_line(line: str, profile: Optional[RfcProfile]) -> bool:
    if not profile or not profile.extra_figure_line_patterns:
        return False
    for rx in _compiled_for(profile):
        if rx.search(line):
            return True
    return False


def merge_profile_chunk_metadata(
    meta: Dict[str, str],
    profile: Optional[RfcProfile],
    *,
    section_number: str,
    section_title: str,
) -> Dict[str, str]:
    """Attach stable topic tags for hybrid / metadata filtering (Chroma ``where``)."""
    if not profile:
        return meta
    tags = ",".join(profile.topics)
    out = {**meta}
    out.setdefault("rfc_topics", tags)
    out.setdefault("rfc_stack_layer", profile.stack_layer)
    out.setdefault("rfc_profile_title", profile.short_title)
    if section_number:
        out.setdefault("section_number", section_number)
    if section_title:
        out.setdefault("section_title", section_title)
    return out
