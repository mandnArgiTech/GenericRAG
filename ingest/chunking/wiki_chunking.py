"""Confluence / wiki page chunking.

Author: deviprasad
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

from ingest.core.file_utils import strip_html

def chunk_wiki_page(
    text: str, path: str, meta: Dict[str, str], embed_model: str = "nomic-embed-text"
) -> List[Tuple[str, Dict[str, str]]]:
    cleaned = strip_html(text)
    parts = chunk_markdown_domain(cleaned, path, embed_model=embed_model)
    out: List[Tuple[str, Dict[str, str]]] = []
    for t, m in parts:
        mm = {**m, "chunk_strategy": "wiki"}
        for k in ("page_title", "space", "labels", "author", "parent_page", "page_url", "last_modified"):
            mm[k] = meta.get(k, "")
        out.append((t, mm))
    return out

