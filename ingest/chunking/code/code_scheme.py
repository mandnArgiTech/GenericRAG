"""Scheme: tree-sitter ``define`` forms with regex fallback.

Author: deviprasad
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Tuple

from ingest.chunking.code.code_languages import generic_split
from ingest.chunking.code.code_tree_sitter import ts_comment_prefix
from ingest.core.deps import _ts_parser_for


def chunk_scheme(content: str, path: Path) -> List[Tuple[str, Dict[str, str]]]:
    try:
        parser = _ts_parser_for("scheme", "tree_sitter_scheme")
        if parser is not None:
            data = content.encode("utf-8", errors="replace")
            tree = parser.parse(data)
            out_ts: List[Tuple[str, Dict[str, str]]] = []

            def walk_scheme(n):
                if n.type == "list" and n.children:
                    first = n.children[0]
                    if first.type == "symbol" and content[first.start_byte : first.end_byte] == "define":
                        sym = n.children[1] if len(n.children) > 1 else None  # pragma: no cover
                        nm = (  # pragma: no cover
                            content[sym.start_byte : sym.end_byte].strip("()")
                            if sym
                            else path.stem
                        )
                        raw = content[n.start_byte : n.end_byte]  # pragma: no cover
                        cmt = ts_comment_prefix(content, n.start_byte, 2)  # pragma: no cover
                        body = (cmt + raw) if cmt else raw  # pragma: no cover
                        out_ts.append(  # pragma: no cover
                            (
                                body[:12000],
                                {
                                    "chunk_strategy": "scheme",
                                    "chunk_type": "define",
                                    "chunk_name": nm[:200],
                                    "chunk_index": str(len(out_ts)),
                                },
                            )
                        )
                for ch in n.children:
                    walk_scheme(ch)

            walk_scheme(tree.root_node)
            if out_ts:
                return out_ts  # pragma: no cover
    except Exception:  # pragma: no cover
        pass  # pragma: no cover
    forms = re.split(r"(?m)(?=^\s*\(define\b)", content)
    out: List[Tuple[str, Dict[str, str]]] = []
    for f in forms:
        f = f.strip()
        if not f.startswith("(define"):
            continue
        m = re.match(r"^\s*\(define\s+(\S+)", f)
        name = m.group(1).strip("()") if m else path.stem
        out.append(
            (
                f[:12000],
                {
                    "chunk_strategy": "scheme",
                    "chunk_type": "define",
                    "chunk_name": name,
                    "chunk_index": str(len(out)),
                },
            )
        )
    if not out:
        return generic_split(content, path, 2000)  # pragma: no cover
    return out
