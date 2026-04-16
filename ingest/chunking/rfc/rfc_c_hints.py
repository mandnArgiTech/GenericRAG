"""C-oriented hints and cross-RFC dependency extraction from RFC plaintext.

Author: deviprasad
"""
from __future__ import annotations

import re
from typing import Any, Dict, List

_NET_ORDER = re.compile(
    r"\b(network\s+byte\s+order|big-?endian|most\s+significant\s+octet\s+first|MSB\s+first)\b",
    re.I,
)
_ALIGN = re.compile(
    r"\b(\d+)\s*[- ]?\s*bit\s+(word|boundary|aligned|alignment|integer)\b|"
    r"\b(multiple|on)\s+of\s+(\d+)\s*[- ]?\s*bytes?\b|"
    r"\b(32|16|64)\s*[- ]?\s*bit\s+(word|boundary|aligned|alignment)\b",
    re.I,
)
_RFC_REF = re.compile(r"\bRFC\s*(\d+)\b", re.I)
_SECTION_REF = re.compile(r"\bSection\s+(\d+(?:\.\d+)*)\b", re.I)


def extract_c_hints(text: str) -> Dict[str, Any]:
    """Return flags / strings useful for C struct layout (flat dict for metadata)."""
    requires = bool(_NET_ORDER.search(text))
    m = _ALIGN.search(text)
    warning = ""
    if m:
        warning = m.group(0).strip()[:500]
    return {
        "requires_network_byte_order": requires,
        "byte_alignment_warning": warning,
    }


def extract_rfc_dependencies(text: str) -> List[str]:
    """Unique RFC cross-references and section refs (short strings)."""
    seen: set[str] = set()
    out: List[str] = []
    for m in _RFC_REF.finditer(text):
        s = f"RFC {m.group(1)}"
        if s not in seen:
            seen.add(s)
            out.append(s)
    for m in _SECTION_REF.finditer(text):
        s = f"Section {m.group(1)}"
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out
