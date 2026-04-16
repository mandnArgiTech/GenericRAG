"""File I/O helpers (encoding sniff, checksums, HTML stripping).

Author: deviprasad
"""
from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Optional, Tuple

from ingest.core.deps import BeautifulSoup

def read_file_bytes(path: Path) -> Tuple[Optional[str], str]:
    raw = path.read_bytes()
    for enc in ("utf-8", "latin-1", "cp1252", "iso-8859-1"):
        try:
            return raw.decode(enc), enc
        except UnicodeDecodeError:
            continue
    return None, "binary"  # pragma: no cover


def file_md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(65536), b""):
            h.update(block)
    return h.hexdigest()


def strip_html(text: str) -> str:
    if BeautifulSoup is not None:
        return BeautifulSoup(text, "html.parser").get_text("\n")
    return re.sub(r"<[^>]+>", " ", text)


def _safe_count(coll) -> int:
    try:
        return int(coll.count())
    except Exception:  # pragma: no cover
        try:  # pragma: no cover
            return int(coll._collection.count())  # type: ignore[attr-defined]  # pragma: no cover
        except Exception:  # pragma: no cover
            return 0  # pragma: no cover
