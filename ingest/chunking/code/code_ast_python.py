"""Python AST–based top-level function/class chunks.

Author: deviprasad
"""
from __future__ import annotations

import ast
from pathlib import Path
from typing import Dict, List, Tuple

from ingest.chunking.code.code_languages import generic_split


def ast_chunk_python(path: Path, content: str) -> List[Tuple[str, Dict[str, str]]]:
    try:
        tree = ast.parse(content)
    except SyntaxError:
        return generic_split(content, path, 1500)
    chunks: List[Tuple[str, Dict[str, str]]] = []
    lines = content.splitlines()

    def slice_node(node: ast.AST) -> str:
        if hasattr(node, "lineno") and hasattr(node, "end_lineno"):
            start = max(0, node.lineno - 1)
            end = min(len(lines), node.end_lineno)
            return "\n".join(lines[start:end])
        seg = ast.get_source_segment(content, node)  # pragma: no cover
        return seg or ""  # pragma: no cover

    idx = 0
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            src = slice_node(node)
            if not src.strip():
                continue  # pragma: no cover
            nm = node.name
            ctype = "class" if isinstance(node, ast.ClassDef) else "function"
            chunks.append(
                (
                    src,
                    {
                        "chunk_strategy": "ast_python",
                        "chunk_type": ctype,
                        "chunk_name": nm,
                        "chunk_index": str(idx),
                    },
                )
            )
            idx += 1
    if not chunks:
        return generic_split(content, path, 1500)
    return chunks
