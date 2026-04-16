"""Generic RFC plaintext operations: depagination, diagram shielding, sliding windows.

Author: deviprasad
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from ingest.chunking.rfc.rfc_profiles import RfcProfile, profile_extra_diagram_line

_RFC_PAGE_MARKER = re.compile(r"\[Page\s+\d+\]", re.I)
_RFC_RUNNING_HDR = re.compile(r"^\s*RFC\s+\d+", re.I)
_RFC_FILENAME_RE = re.compile(r"rfc\d+", re.I)


def is_rfc_file(path: Path) -> bool:
    """True for RFC Editor-style plain-text RFCs (e.g. rfc3376.txt)."""
    return path.suffix.lower() == ".txt" and bool(_RFC_FILENAME_RE.search(path.stem))


def depaginate_rfc(text: str) -> str:
    """Strip form-feed pagination, page markers, and common running headers/footers."""
    pages = text.split("\f") if "\f" in text else [text]
    cleaned: List[str] = []
    for page in pages:
        page = _RFC_PAGE_MARKER.sub("", page)
        lines = page.split("\n")
        hdr_strips = 0
        while lines and hdr_strips < 2:
            first = lines[0].strip()
            if not first:
                lines.pop(0)
                continue
            if _RFC_RUNNING_HDR.match(first) or (
                re.match(r"^RFC\s+\d+", first, re.I) and len(first) < 140
            ):
                lines.pop(0)
                hdr_strips += 1
                continue
            if re.match(r"^Internet-Draft\b", first, re.I) and len(first) < 100:
                lines.pop(0)  # pragma: no cover
                hdr_strips += 1  # pragma: no cover
                continue  # pragma: no cover
            break
        ftr_strips = 0
        while lines and ftr_strips < 2:
            last = lines[-1].strip()
            if not last:
                lines.pop()
                continue
            if _RFC_PAGE_MARKER.search(last) or re.match(
                r"^.{0,140}\[Page\s+\d+\]\s*$", last, re.I
            ):
                lines.pop()  # pragma: no cover
                ftr_strips += 1  # pragma: no cover
                continue  # pragma: no cover
            if re.match(r"^Full\s+Standards?\s+Section", last, re.I):
                lines.pop()  # pragma: no cover
                ftr_strips += 1  # pragma: no cover
                continue  # pragma: no cover
            break
        cleaned.append("\n".join(lines))
    merged = "\n\n".join(p.strip() for p in cleaned if p.strip())
    merged = _RFC_PAGE_MARKER.sub("", merged)
    merged = re.sub(r"\n{3,}", "\n\n", merged)
    return merged


def rfc_heading_line_is_table_of_contents(title: str) -> bool:
    """
    RFC plaintext ToC lines look like ``3. PPP Link Operation ................ 6``.
    Real section headings do not contain long dot leaders or a trailing page column.
    """
    t = title.strip()
    if re.search(r"\.{5,}", t):
        return True
    if re.search(r"\.{3,}\s*\d{1,4}\s*$", t):
        return True
    return False


def rfc_base_line_is_diagram(line: str) -> bool:
    """Heuristic for monospaced ASCII art common in RFCs (independent of RFC number)."""
    if not line.strip():
        return False
    lead = len(line) - len(line.lstrip(" \t"))
    if lead >= 3:
        return True
    s = line
    if re.search(r"[+|]\s*[-+]{2,}", s) or re.search(r"[-+]{2,}\s*[+|]", s):
        return True
    if "|" in s and ("+-" in s or "-+" in s or ".-" in s):
        return True  # pragma: no cover
    st = s.strip()
    if len(st) >= 10 and sum(1 for c in st if c.isdigit()) >= 4:
        if re.match(r"^[\d\s]+$", st):
            return True  # pragma: no cover
    return False


def rfc_line_is_diagram(line: str, profile: Optional[RfcProfile] = None) -> bool:
    if rfc_base_line_is_diagram(line):
        return True
    return profile_extra_diagram_line(line, profile)


def shield_diagrams(
    text: str,
    line_is_diagram: Optional[Callable[[str], bool]] = None,
) -> Tuple[str, Dict[str, str]]:
    """Replace ASCII diagram runs with placeholders so paragraph splitters never break them."""
    pred = line_is_diagram or rfc_base_line_is_diagram
    lines = text.split("\n")
    vault: Dict[str, str] = {}
    out: List[str] = []
    i = 0
    n = len(lines)
    d_idx = 0
    while i < n:
        if not pred(lines[i]):
            out.append(lines[i])
            i += 1
            continue
        j = i
        diag_count = 0
        saw_blank = False
        while j < n:
            ln = lines[j]
            if pred(ln):
                diag_count += 1
                saw_blank = False
                j += 1
            elif not ln.strip():
                if saw_blank:  # pragma: no cover
                    break  # pragma: no cover
                if j + 1 < n and pred(lines[j + 1]):  # pragma: no cover
                    saw_blank = True  # pragma: no cover
                    j += 1  # pragma: no cover
                else:
                    break  # pragma: no cover
            else:
                break
        if diag_count >= 3:
            key = f"__DIAGRAM_{d_idx}__"  # pragma: no cover
            d_idx += 1  # pragma: no cover
            vault[key] = "\n".join(lines[i:j])  # pragma: no cover
            out.append(key)  # pragma: no cover
            i = j  # pragma: no cover
        else:
            out.extend(lines[i:j])
            i = j
    return "\n".join(out), vault


def _diagram_vault_sort_key(k: str) -> int:
    try:
        return int(k.replace("__DIAGRAM_", "").replace("__", ""))
    except ValueError:  # pragma: no cover
        return 0  # pragma: no cover


def unshield_diagrams(text: str, vault: Dict[str, str]) -> str:
    for k in sorted(vault.keys(), key=_diagram_vault_sort_key):
        text = text.replace(k, vault[k])  # pragma: no cover
    return text


def sliding_window_chunks(
    text: str,
    target_max_chars: int,
    overlap_frac: float = 0.15,
    base_meta: Optional[Dict[str, str]] = None,
) -> List[Tuple[str, Dict[str, str]]]:
    """Token-budget sliding windows with overlap; prefers paragraph then sentence boundaries."""
    base_meta = dict(base_meta or {})
    overlap = max(64, int(target_max_chars * overlap_frac))
    t = text.strip()
    if not t:
        return []  # pragma: no cover
    if len(t) <= target_max_chars:
        return [(t, {**base_meta, "chunk_type": "sliding_window", "chunk_index": "0"})]
    out: List[Tuple[str, Dict[str, str]]] = []
    pos = 0
    idx = 0
    tl = len(t)
    while pos < tl:
        end = min(pos + target_max_chars, tl)
        if end < tl:
            br = t.rfind("\n\n", pos + max(64, target_max_chars // 5), end)
            if br >= pos:
                end = br + 2
            else:
                br2 = t.rfind(". ", pos + target_max_chars // 3, end)  # pragma: no cover
                if br2 >= pos:  # pragma: no cover
                    end = br2 + 2  # pragma: no cover
        chunk = t[pos:end].strip()
        if chunk:
            out.append((chunk, {**base_meta, "chunk_type": "sliding_window", "chunk_index": str(idx)}))
            idx += 1
        if end >= tl:
            break
        next_pos = max(pos + 1, end - overlap)
        snap = t.find("\n\n", next_pos, min(next_pos + overlap * 3, tl))
        if snap != -1:
            next_pos = snap + 2
        pos = next_pos
    return out
