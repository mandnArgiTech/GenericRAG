"""Tree-sitter extraction for C/C++/Java (and shared comment prefix for Scheme).

Author: deviprasad
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

from ingest.core.deps import _ts_parser_for


def ts_comment_prefix(content: str, start_byte: int, max_lines: int = 2) -> str:
    """Prepend up to `max_lines` of // or /* */ comments immediately above node."""
    if start_byte <= 0:
        return ""
    prefix = content[:start_byte]
    lines = prefix.splitlines()
    if not lines:
        return ""  # pragma: no cover
    buf: List[str] = []
    i = len(lines) - 1
    lines_seen = 0
    while i >= 0 and lines_seen < max_lines:
        ln = lines[i].rstrip()
        stripped = ln.lstrip()
        if not stripped:
            i -= 1  # pragma: no cover
            continue  # pragma: no cover
        if stripped.startswith("//"):
            buf.append(ln)
            lines_seen += 1
            i -= 1
            continue
        if "*/" in stripped or stripped.startswith("/*") or stripped.startswith("*"):
            buf.append(ln)
            lines_seen += 1
            i -= 1
            continue
        break
    if not buf:
        return ""
    return "\n".join(reversed(buf)) + "\n"


def ts_extract_chunks(path: Path, content: str, grammar: str) -> Optional[List[Tuple[str, Dict[str, str]]]]:
    mod_map = {
        "c": "tree_sitter_c",
        "cpp": "tree_sitter_cpp",
        "java": "tree_sitter_java",
    }
    if grammar not in mod_map:
        return None
    parser = _ts_parser_for(grammar, mod_map[grammar])
    if parser is None:
        return None  # pragma: no cover
    data = content.encode("utf-8", errors="replace")
    tree = parser.parse(data)

    targets = {
        "c": {"function_definition", "struct_specifier", "enum_specifier"},
        "cpp": {"function_definition", "class_specifier", "struct_specifier", "enum_specifier"},
        "java": {"method_declaration", "class_declaration", "interface_declaration"},
    }[grammar]

    out: List[Tuple[str, Dict[str, str]]] = []

    def node_text(node) -> str:
        return content[node.start_byte : node.end_byte]

    def walk(node, classname: str = ""):
        t = node.type
        if t in targets:
            txt = node_text(node).strip()
            if not txt:
                return  # pragma: no cover
            cmt = ts_comment_prefix(content, node.start_byte, 2)
            if cmt:
                txt = cmt + txt  # pragma: no cover
            name = path.stem
            chunk_name = name
            if grammar == "cpp" and t == "function_definition":
                for ch in node.children:
                    if ch.type == "function_declarator":
                        for g in ch.children:
                            if g.type == "identifier":
                                chunk_name = content[g.start_byte : g.end_byte]
            if grammar == "java" and t == "method_declaration":
                for ch in node.children:
                    if ch.type == "identifier":
                        chunk_name = content[ch.start_byte : ch.end_byte]
                        break
            if classname and grammar in ("cpp", "java"):
                chunk_name = f"{classname}::{chunk_name}"
            out.append(
                (
                    txt[:12000],
                    {
                        "chunk_strategy": f"ast_{grammar}",
                        "chunk_type": t,
                        "chunk_name": chunk_name[:200],
                        "chunk_index": str(len(out)),
                    },
                )
            )
        if grammar == "cpp" and t == "class_specifier":
            cname = classname  # pragma: no cover
            for ch in node.children:  # pragma: no cover
                if ch.type == "type_identifier":  # pragma: no cover
                    cname = content[ch.start_byte : ch.end_byte]  # pragma: no cover
                    break  # pragma: no cover
            for ch in node.children:  # pragma: no cover
                walk(ch, cname or classname)  # pragma: no cover
        elif grammar == "java" and t in ("class_declaration", "interface_declaration"):
            cname = classname
            for ch in node.children:
                if ch.type == "identifier":
                    cname = content[ch.start_byte : ch.end_byte]
                    break
            for ch in node.children:
                walk(ch, cname or classname)
        else:
            for ch in node.children:
                walk(ch, classname)

    walk(tree.root_node)
    return out if out else None
