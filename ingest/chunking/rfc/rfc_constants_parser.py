"""Generic constant / opcode / table extraction for ``constants`` chunks.

Author: deviprasad
"""
from __future__ import annotations

import re
from typing import Dict, List, Tuple

from ingest.chunking.rfc.rfc_struct_parser import _line_byte_offsets, strip_spans


_HEX_LINE = re.compile(r"^\s*(?:0x)?[0-9A-Fa-f]{1,4}\b", re.MULTILINE)
_TABLE_HEADER = re.compile(
    r"^\s*(Kind|Type|Code|Value|Opcode|Message|Mnemonic|Number)\b.*\b(Length|Meaning|Description|Name)\b",
    re.I,
)


def parse_constant_tables(body: str) -> List[Tuple[str, Dict[str, str], Tuple[int, int]]]:
    """
    Heuristic blocks: lines with hex-like tokens, ``Kind ... Length`` tables,
    or numbered constant lists (``1`` ``FOO`` ...).
    """
    if not body.strip():
        return []
    lines = body.split("\n")
    offs = _line_byte_offsets(body)
    n = len(lines)
    out: List[Tuple[str, Dict[str, str], Tuple[int, int]]] = []
    i = 0
    while i < n:
        ln = lines[i]
        block_start = i
        # Table-ish header row
        if _TABLE_HEADER.search(ln) and i + 1 < n:
            j = i + 1
            while j < n and j - i < 40:
                if not lines[j].strip():
                    if j - i > 2:
                        break
                    j += 1
                    continue
                if re.match(r"^\d+(?:\.\d+)*\.?\s+[A-Z]", lines[j]):
                    break
                j += 1
            if j - block_start >= 3:
                start = offs[block_start][0]
                end = offs[j - 1][1]
                chunk = body[start:end].strip()
                if len(chunk) > 20:
                    out.append((chunk, {"chunk_type": "constants"}, (start, end)))
                i = j
                continue
        # Hex-heavy run
        hex_lines = 0
        j = i
        while j < n and hex_lines < 50:
            l2 = lines[j]
            if not l2.strip():
                if hex_lines >= 2:
                    break
                j += 1
                continue
            if _HEX_LINE.search(l2) or re.search(r"\b0x[0-9A-Fa-f]{2,}\b", l2):
                hex_lines += 1
                j += 1
                continue
            if hex_lines >= 2 and (len(l2) - len(l2.lstrip()) > 8 or l2.lstrip().startswith("|")):
                j += 1
                continue
            if hex_lines >= 2:
                break
            j += 1
            break
        if hex_lines >= 3 and j - i >= 3:
            start = offs[i][0]
            end = offs[j - 1][1]
            chunk = body[start:end].strip()
            out.append((chunk, {"chunk_type": "constants"}, (start, end)))
            i = j
            continue
        i += 1
    return out


def parse_all_constant_regions(body: str) -> Tuple[List[Tuple[str, Dict[str, str]]], List[Tuple[int, int]]]:
    chunks: List[Tuple[str, Dict[str, str]]] = []
    spans: List[Tuple[int, int]] = []
    for ch, meta, sp in parse_constant_tables(body):
        chunks.append((ch, meta))
        spans.append(sp)
    return chunks, spans


def strip_constant_spans(body: str, spans: List[Tuple[int, int]]) -> str:
    return strip_spans(body, spans)
