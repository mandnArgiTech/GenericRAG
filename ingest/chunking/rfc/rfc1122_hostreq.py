"""RFC 1122 Host Requirements — normative rules use generic ``implementation_rule`` parser.

Reserved plugin slot (no section match): extend here for chapter-specific tables.

Author: deviprasad
"""
from __future__ import annotations

from ingest.chunking.rfc.rfc_registry import RfcSectionParser, register_rfc_parser

register_rfc_parser(
    RfcSectionParser(
        rfc_number="1122",
        section_pattern=r"a^",
        parse_fn=lambda body, sec: [],
        strip_fn=lambda body, sec: body,
        priority=0,
    )
)
