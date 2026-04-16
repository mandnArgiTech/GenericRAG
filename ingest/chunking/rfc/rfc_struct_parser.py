"""Generic RFC ASCII diagram / bit-ruler extraction for ``struct_definition`` chunks.

Author: deviprasad
"""
from __future__ import annotations

import re
from typing import Dict, List, Tuple

from ingest.chunking.rfc.rfc_text_ops import rfc_base_line_is_diagram

_MAX_FIELD_LABEL = 120
_FIELD_LIST_HEAD = re.compile(
    r"^\s*\d+\s*\.\s*(?:bit|bits|octet|octets|byte|bytes)\s*[.:]\s*(.+?)\s*$",
    re.I,
)


def infer_struct_field_label(chunk: str) -> str:
    """
    Best-effort primary field / row label for ``[Field: …]`` in hierarchical paths.

    Prefers RFC 826-style ``N.bit: description``; else first substantive ``| cell |`` in a box diagram.
    """
    if not chunk.strip():
        return ""
    for ln in chunk.split("\n"):
        m = _FIELD_LIST_HEAD.match(ln)
        if m:
            label = re.sub(r"\s+", " ", m.group(1).strip())
            while label.startswith("("):
                close = label.find(")")
                if close == -1:
                    break
                label = label[close + 1 :].strip()
            if len(label) > 2:
                return label[:_MAX_FIELD_LABEL]
    for ln in chunk.split("\n"):
        if "|" not in ln:
            continue
        st = ln.strip()
        if not st or st.startswith("+"):
            continue
        if re.fullmatch(r"[\-+|\s]+", re.sub(r"[^\s\-+|]", "", st)) and "+" in st:
            continue
        cells = [c.strip() for c in ln.split("|")]
        cells = [c for c in cells if c]
        for cell in cells:
            if len(cell) < 2:
                continue
            if re.fullmatch(r"[\d\s]+", cell):
                continue
            if set(cell) <= set("-_ "):
                continue
            clean = re.sub(r"\s+", " ", cell)
            return clean[:_MAX_FIELD_LABEL]
    return ""


def _line_byte_offsets(text: str) -> List[Tuple[int, int]]:
    """Inclusive start, exclusive end per line (splitlines without keepends)."""
    lines = text.split("\n")
    out: List[Tuple[int, int]] = []
    pos = 0
    for i, ln in enumerate(lines):
        start = pos
        end = start + len(ln)
        out.append((start, end))
        pos = end + (1 if i + 1 < len(lines) else 0)  # + newline except last
    return out


def _is_box_diagram_line(line: str) -> bool:
    s = line.rstrip("\n")
    if not s.strip():
        return False
    if rfc_base_line_is_diagram(s):
        return True
    st = s.strip()
    if ("+-" in st or "-+" in st) and "|" in st:
        return True
    if "|" in st and re.search(r"[-+]{3,}", st):
        return True
    return False


def parse_bit_ruler_diagrams(body: str) -> List[Tuple[str, Dict[str, str], Tuple[int, int]]]:
    """
    Find monospaced box diagrams (``+-+-+`` / ``| field |``) and following description lines.

    Returns list of (chunk_text, partial_meta, (start, end) span in ``body``).
    """
    if not body.strip():
        return []
    lines = body.split("\n")
    offs = _line_byte_offsets(body)
    n = len(lines)
    out: List[Tuple[str, Dict[str, str], Tuple[int, int]]] = []
    used: set[int] = set()

    i = 0
    while i < n:
        if i in used or not _is_box_diagram_line(lines[i]):
            i += 1
            continue
        j = i
        diagram_lines = 0
        while j < n and (_is_box_diagram_line(lines[j]) or (not lines[j].strip() and j + 1 < n and _is_box_diagram_line(lines[j + 1]))):
            if _is_box_diagram_line(lines[j]):
                diagram_lines += 1
            j += 1
        if diagram_lines < 2:
            i += 1
            continue
        k = j
        desc_count = 0
        while k < n and desc_count < 24:
            ln = lines[k]
            if not ln.strip():
                if desc_count > 0:
                    break
                k += 1
                continue
            if re.match(r"^\d+(?:\.\d+)*\.?\s+\S", ln):
                break
            if _is_box_diagram_line(ln):
                break
            desc_count += 1
            k += 1
        start = offs[i][0]
        end = offs[k - 1][1] if k > i else offs[j - 1][1]
        chunk = body[start:end].strip()
        if len(chunk) < 12:
            i += 1
            continue
        meta: Dict[str, str] = {"chunk_type": "struct_definition"}
        hf = infer_struct_field_label(chunk)
        if hf:
            meta["hierarchical_field"] = hf
        out.append((chunk, meta, (start, end)))
        for idx in range(i, k):
            used.add(idx)
        i = k
    return out


def parse_field_list_blocks(body: str) -> List[Tuple[str, Dict[str, str], Tuple[int, int]]]:
    """
    Detect ``N bit`` / ``N-octet`` field list blocks (e.g. ARP packet breakdown).
    """
    if not body.strip():
        return []
    lines = body.split("\n")
    offs = _line_byte_offsets(body)
    n = len(lines)
    out: List[Tuple[str, Dict[str, str], Tuple[int, int]]] = []
    # RFC 826 style: ``48.bit:`` / ``16.bit:``; also ``16 bit:`` / ``8 octets``.
    field_line = re.compile(
        r"^\s*(?P<nb>\d+)\s*\.\s*(?:bit|bits|octet|octets|byte|bytes)\s*[.:]\s*|"
        r"^\s*\d+\s+(?:bit|bits|octet|octets|byte|bytes)\s*[.:\-]\s*\(?",
        re.I,
    )
    i = 0
    while i < n:
        if not field_line.match(lines[i]):
            i += 1
            continue
        j = i
        while j < n and (
            field_line.match(lines[j])
            or (lines[j].strip() and (lines[j].startswith((" ", "\t")) or lines[j].lstrip().startswith("(")))
        ):
            j += 1
            if j - i > 80:
                break
        if j - i < 2:
            i += 1
            continue
        start = offs[i][0]
        end = offs[j - 1][1]
        chunk = body[start:end].strip()
        meta: Dict[str, str] = {"chunk_type": "struct_definition"}
        hf = infer_struct_field_label(chunk)
        if hf:
            meta["hierarchical_field"] = hf
        out.append((chunk, meta, (start, end)))
        i = j
    return out


def merge_spans(spans: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    if not spans:
        return []
    spans = sorted(spans)
    merged = [spans[0]]
    for s, e in spans[1:]:
        ps, pe = merged[-1]
        if s <= pe + 1:
            merged[-1] = (ps, max(pe, e))
        else:
            merged.append((s, e))
    return merged


def strip_spans(body: str, spans: List[Tuple[int, int]]) -> str:
    """Remove span ranges; collapse whitespace."""
    if not spans:
        return body
    t = body
    for s, e in reversed(merge_spans(spans)):
        s = max(0, min(s, len(t)))
        e = max(s, min(e, len(t)))
        t = t[:s].rstrip() + "\n\n" + t[e:].lstrip()
    return re.sub(r"\n{3,}", "\n\n", t).strip()


def parse_all_struct_regions(body: str) -> Tuple[List[Tuple[str, Dict[str, str]]], List[Tuple[int, int]]]:
    """Bit-ruler + field-list; returns (chunks, spans) for stripping."""
    seen: set[Tuple[int, int]] = set()
    chunks: List[Tuple[str, Dict[str, str]]] = []
    spans: List[Tuple[int, int]] = []
    for chunk, meta, span in parse_bit_ruler_diagrams(body) + parse_field_list_blocks(body):
        if span in seen:
            continue
        seen.add(span)
        chunks.append((chunk, meta))
        spans.append(span)
    return chunks, spans
