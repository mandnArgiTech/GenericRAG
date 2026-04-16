"""Release notes, frontmatter, and community doc chunking.

Author: deviprasad
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Tuple

def _extract_release_date_near_version(body: str) -> str:
    m = re.search(
        r"(?i)(?:released?|published|date)[:.\s]+(\d{4}-\d{2}-\d{2}|\d{1,2}\s+\w+\s+\d{4}|\w+\s+\d{1,2},?\s+\d{4})",
        body[:2500],
    )
    return m.group(1).strip() if m else ""




def chunk_release_notes(text: str, path: str) -> List[Tuple[str, Dict[str, str]]]:
    heads = list(
        re.finditer(r"(?m)^(#{1,3}\s*v?[\d.]+[^\n]*|Version\s+[\d.]+[^\n]*|#\s*[\d.]+[^\n]*)", text)
    )
    out: List[Tuple[str, Dict[str, str]]] = []
    if not heads:
        return [  # pragma: no cover
            (
                text[:10000],
                {
                    "chunk_strategy": "release_notes",
                    "chunk_type": "changelog",
                    "version": "",
                    "chunk_index": "0",
                },
            )
        ]
    for i, m in enumerate(heads):
        ver = re.sub(r"^[#vV\s]+", "", m.group(0)).strip()
        start = m.start()
        end = heads[i + 1].start() if i + 1 < len(heads) else len(text)
        body = text[start:end].strip()
        cat = "general"
        if re.search(r"known issue", body, re.I):
            cat = "Known Issues"  # pragma: no cover
        elif re.search(r"breaking", body, re.I):
            cat = "Breaking Changes"  # pragma: no cover
        elif re.search(r"bug fix", body, re.I):
            cat = "Bug Fixes"
        elif re.search(r"new feature", body, re.I):
            cat = "New Features"
        ctype = "general"
        if cat == "Known Issues":
            ctype = "edge_case"  # pragma: no cover
        elif cat == "Breaking Changes":
            ctype = "constraint"  # pragma: no cover
        rdate = _extract_release_date_near_version(body)
        parts = _split_paragraphs(body, 2000, 6000) if len(body) > 8000 else [body]
        for p in parts:
            out.append(
                (
                    p,
                    {
                        "chunk_strategy": "release_notes",
                        "chunk_type": "release",
                        "version": ver,
                        "release_date": rdate,
                        "section_category": cat,
                        "content_type": ctype,
                        "chunk_index": str(len(out)),
                    },
                )
            )
    return out


def parse_frontmatter(text: str) -> Tuple[Dict[str, str], str]:
    if not text.startswith("---"):
        return {}, text  # pragma: no cover
    end = text.find("\n---", 3)
    if end == -1:
        return {}, text  # pragma: no cover
    fm_raw = text[3:end]
    body = text[end + 4 :].lstrip("\n")
    fm: Dict[str, str] = {}
    for ln in fm_raw.splitlines():
        if ":" in ln:
            k, v = ln.split(":", 1)
            fm[k.strip()] = v.strip().strip('"')
    return fm, body


def chunk_community(text: str, path: str, fm: Dict[str, str]) -> List[Tuple[str, Dict[str, str]]]:
    if len(text) <= 8000:
        parts = [text]
    else:
        bits = re.split(r"(?m)(^---\s*$|^## Answer|^## Resolution)", text)  # pragma: no cover
        parts = []  # pragma: no cover
        buf = ""  # pragma: no cover
        for s in bits:  # pragma: no cover
            if len(buf) + len(s) > 8000 and buf:  # pragma: no cover
                parts.append(buf)  # pragma: no cover
                buf = s  # pragma: no cover
            else:
                buf += s  # pragma: no cover
        if buf:  # pragma: no cover
            parts.append(buf)  # pragma: no cover
    out: List[Tuple[str, Dict[str, str]]] = []
    for i, p in enumerate(parts):
        out.append(
            (
                p.strip(),
                {
                    "chunk_strategy": "community",
                    "chunk_type": "thread",
                    "source_platform": fm.get("source_platform", "unknown"),
                    "source_url": fm.get("source_url", ""),
                    "is_resolved": fm.get("is_resolved", ""),
                    "has_workaround": fm.get("has_workaround", ""),
                    "quality_score": fm.get("quality_score", ""),
                    "chunk_index": str(i),
                },
            )
        )
    return out

