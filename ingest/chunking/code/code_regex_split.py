"""Regex-based top-level splits for languages without dedicated parsers in this pipeline.

Author: deviprasad
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Tuple

from ingest.chunking.code.code_languages import generic_split

_REGEX_CODE_PATTERNS: Dict[str, List[str]] = {
    ".go": [r"^\s*func\s+", r"^\s*type\s+\w+\s+struct\b"],
    ".rs": [
        r"^\s*(?:pub\s+)?(?:unsafe\s+)?fn\s+",
        r"^\s*(?:pub\s+)?struct\s+",
        r"^\s*(?:pub\s+)?enum\s+",
        r"^\s*(?:pub\s+)?impl\b",
        r"^\s*(?:pub\s+)?trait\s+",
    ],
    ".rb": [r"^\s*class\s+", r"^\s*module\s+", r"^\s*def\s+"],
    ".kt": [
        r"^\s*(?:public\s+|private\s+|internal\s+|protected\s+)?(?:open\s+|abstract\s+|sealed\s+)?fun\s+",
        r"^\s*class\s+",
        r"^\s*object\s+",
        r"^\s*interface\s+",
    ],
    ".kts": [r"^\s*fun\s+", r"^\s*class\s+", r"^\s*object\s+"],
    ".swift": [r"^\s*func\s+", r"^\s*class\s+", r"^\s*struct\s+", r"^\s*enum\s+", r"^\s*protocol\s+"],
    ".scala": [r"^\s*def\s+", r"^\s*class\s+", r"^\s*object\s+", r"^\s*trait\s+"],
    ".php": [r"^\s*function\s+", r"^\s*class\s+"],
}


def _merge_small_regex_chunks(
    parts: List[str], min_chars: int = 200, max_chars: int = 12000
) -> List[str]:
    if not parts:
        return []
    merged: List[str] = []
    buf = parts[0]
    for p in parts[1:]:
        if len(buf) < min_chars and len(buf) + len(p) <= max_chars:
            buf = buf + "\n\n" + p
        else:
            merged.append(buf)
            buf = p
    merged.append(buf)
    return merged


def regex_code_split(content: str, path: Path, ext: str) -> List[Tuple[str, Dict[str, str]]]:
    """Split on language-typical top-level boundaries; fallback to generic_split."""
    patterns = _REGEX_CODE_PATTERNS.get(ext.lower())
    if not patterns:
        return generic_split(content, path, 1500)
    combined = "|".join(f"({p})" for p in patterns)
    try:
        rx = re.compile(combined, re.MULTILINE)
    except re.error:
        return generic_split(content, path, 1500)
    matches = list(rx.finditer(content))
    if not matches:
        return generic_split(content, path, 1500)
    starts = sorted({m.start() for m in matches})
    if starts[0] > 0:
        starts.insert(0, 0)
    raw_parts: List[str] = []
    for i, st in enumerate(starts):
        end = starts[i + 1] if i + 1 < len(starts) else len(content)
        seg = content[st:end].strip()
        if seg:
            raw_parts.append(seg)
    if not raw_parts:
        return generic_split(content, path, 1500)
    raw_parts = _merge_small_regex_chunks(raw_parts)
    return [
        (
            seg[:12000],
            {
                "chunk_strategy": "regex_code",
                "chunk_type": "fragment",
                "chunk_name": path.stem,
                "chunk_index": str(i),
            },
        )
        for i, seg in enumerate(raw_parts)
    ]
