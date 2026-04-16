"""Build a PageIndex over structured markdown documentation chapters.

Each chapter file is a *page*. The tree mirrors the markdown header hierarchy.
When a code PageIndex is provided, doc→code cross-references are built by
matching mentioned function names and source filenames against the code tree.

Author: deviprasad
"""
from __future__ import annotations

import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from ingest.pageindex.code_index import save_index  # re-use same persistence format

logger = logging.getLogger("ingest.pageindex.docs")

MARKDOWN_EXTENSIONS = {".md", ".markdown"}

_HEADER_RE = re.compile(r"^(#{1,6})\s+(.+)$")
_FENCE_RE = re.compile(r"^```")

_IDENT_NOISE = {
    "THE", "AND", "FOR", "NOT", "BUT", "ARE", "THIS", "THAT", "WITH", "FROM",
    "HAVE", "HAS", "WAS", "WERE", "WILL", "CAN", "ALL", "ANY", "NOTE", "TODO",
    "FIXME", "WARNING", "ERROR", "CHAPTER", "SECTION", "FIGURE", "TABLE",
}


# ---------------------------------------------------------------------------
# Markdown parsing
# ---------------------------------------------------------------------------

def parse_markdown_structure(content: str, filepath: str) -> Dict[str, Any]:
    """Return section list + chapter title."""
    lines = content.split("\n")
    sections: List[Dict[str, Any]] = []
    in_fence = False

    for ln, raw in enumerate(lines, 1):
        stripped = raw.strip()
        if _FENCE_RE.match(stripped):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        m = _HEADER_RE.match(stripped)
        if m:
            sections.append(
                {
                    "title": m.group(2).strip(),
                    "level": len(m.group(1)),
                    "start_line": ln,
                }
            )

    for i, sec in enumerate(sections):
        start = sec["start_line"] - 1
        end = sections[i + 1]["start_line"] - 1 if i + 1 < len(sections) else len(lines)
        sec["end_line"] = end
        sec["text"] = "\n".join(lines[start:end]).strip()
        sec["line_count"] = end - start

    doc_title = os.path.splitext(os.path.basename(filepath))[0]
    for s in sections:
        if s["level"] == 1:
            doc_title = s["title"]
            break
    return {
        "title": doc_title,
        "filepath": filepath,
        "total_lines": len(lines),
        "sections": sections,
    }


def build_section_tree(sections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Nest flat header list into a tree based on header level (# vs ##)."""
    if not sections:
        return []
    stack: List[tuple] = []
    roots: List[Dict[str, Any]] = []
    counter = [1]
    for sec in sections:
        node = {
            "title": sec["title"],
            "node_id": str(counter[0]).zfill(4),
            "start_line": sec["start_line"],
            "end_line": sec["end_line"],
            "line_count": sec["line_count"],
            "nodes": [],
        }
        counter[0] += 1
        while stack and stack[-1][1] >= sec["level"]:
            stack.pop()
        if not stack:
            roots.append(node)
        else:
            stack[-1][0]["nodes"].append(node)
        stack.append((node, sec["level"]))
    return roots


# ---------------------------------------------------------------------------
# Code references
# ---------------------------------------------------------------------------

def extract_code_references(content: str) -> Dict[str, List[str]]:
    """Best-effort scan for source filenames / function / macro / struct mentions."""
    source_files = sorted(set(re.findall(r"\b[\w/]+\.[ch](?:pp|xx|c)?\b", content)))
    backtick_refs = re.findall(r"`([A-Za-z_][\w:]*)`", content)
    call_refs = re.findall(r"\b([A-Z][A-Za-z]+[a-z]\w*)\s*\(", content)
    macros = [m for m in re.findall(r"\b([A-Z][A-Z_]{2,})\b", content) if m not in _IDENT_NOISE]
    struct_refs = re.findall(r"\b(?:struct\s+)?([A-Z][a-z]+(?:instance|model|def|info|struct))\b", content)
    return {
        "source_files": source_files[:30],
        "functions": sorted(set(call_refs + [r for r in backtick_refs if r and r[0].isupper()]))[:30],
        "macros": sorted(set(macros))[:20],
        "structs": sorted(set(struct_refs))[:15],
    }


def _build_function_lookup(code_structure: Dict[str, Any]) -> Dict[str, str]:
    """Flatten the code PageIndex tree into ``function_name → filepath`` pairs."""
    lookup: Dict[str, str] = {}

    def _walk(nodes: List[Dict[str, Any]]) -> None:
        for node in nodes:
            funcs = node.get("functions") or []
            if isinstance(funcs, str):
                funcs = funcs.split()
            fp = node.get("filepath", "")
            if fp:
                for fn in funcs:
                    lookup.setdefault(fn, fp)
            if node.get("nodes"):
                _walk(node["nodes"])

    _walk(code_structure.get("structure", []))
    return lookup


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_doc_page_index(
    chapters_dir: str,
    *,
    code_index_dir: Optional[str] = None,
    use_llm_summaries: bool = False,
    summary_model: Optional[str] = None,
) -> Dict[str, Any]:
    """Build a doc PageIndex. Pages are chapter files; sections are nested under each chapter."""
    t0 = time.time()
    base = Path(chapters_dir).resolve()
    if not base.is_dir():
        raise ValueError(f"Not a directory: {chapters_dir}")

    md_files = sorted(
        (f for f in base.rglob("*") if f.is_file() and f.suffix.lower() in MARKDOWN_EXTENSIONS),
        key=lambda f: f.relative_to(base).as_posix(),
    )
    if not md_files:
        raise ValueError(f"No markdown files in {chapters_dir}")

    logger.info("found %s chapter files in %s", len(md_files), chapters_dir)

    # Optional: load code tree for cross-references.
    code_functions: Dict[str, str] = {}
    code_page_map: Dict[str, str] = {}
    if code_index_dir:
        struct_path = Path(code_index_dir) / "structure.json"
        map_path = Path(code_index_dir) / "page_map.json"
        if struct_path.is_file():
            with open(struct_path, "r", encoding="utf-8") as fh:
                code_functions = _build_function_lookup(json.load(fh))
        if map_path.is_file():
            with open(map_path, "r", encoding="utf-8") as fh:
                code_page_map = json.load(fh)
        logger.info(
            "loaded code cross-refs: %s functions, %s pages",
            len(code_functions),
            len(code_page_map),
        )

    chapters: List[Dict[str, Any]] = []
    pages: List[Dict[str, Any]] = []
    all_xrefs: Dict[str, Dict[str, str]] = {}

    for i, mdf in enumerate(md_files, 1):
        content = mdf.read_text(encoding="utf-8", errors="replace")
        parsed = parse_markdown_structure(content, str(mdf.relative_to(base)))
        section_tree = build_section_tree(parsed["sections"])
        refs = extract_code_references(content)

        xrefs: Dict[str, str] = {}
        for fn in refs["functions"]:
            if fn in code_functions:
                xrefs[fn] = code_functions[fn]
        for src in refs["source_files"]:
            for _, path in code_page_map.items():
                if src == os.path.basename(path) or path.endswith(src):
                    xrefs[src] = path
                    break

        section_titles = [s["title"] for s in parsed["sections"][:15]]
        summary_parts = []
        if section_titles:
            summary_parts.append("Covers: " + ", ".join(section_titles[:5]))
        if refs["functions"]:
            summary_parts.append("Refs functions: " + ", ".join(refs["functions"][:5]))
        if refs["source_files"]:
            summary_parts.append("Refs files: " + ", ".join(refs["source_files"][:3]))

        chapters.append(
            {
                "title": parsed["title"],
                "node_id": str(i).zfill(4),
                "start_index": i,
                "end_index": i,
                "filepath": parsed["filepath"],
                "line_count": parsed["total_lines"],
                "section_titles": section_titles,
                "code_references": refs,
                "cross_references": xrefs or None,
                "summary": ". ".join(summary_parts) or f"Chapter ({parsed['total_lines']} lines)",
                "nodes": section_tree,
            }
        )
        pages.append(
            {
                "page": i,
                "filepath": parsed["filepath"],
                "title": parsed["title"],
                "content": content,
            }
        )
        if xrefs:
            all_xrefs[parsed["filepath"]] = xrefs

        if i % 25 == 0:
            logger.info("processed %s/%s chapters", i, len(md_files))

    logger.info("doc PageIndex built in %.1fs", time.time() - t0)
    return {
        "index_kind": "docs",
        "doc_name": base.name,
        "doc_description": (
            f"Technical documentation: {len(md_files)} chapters, "
            f"cross-references {sum(len(v) for v in all_xrefs.values())} code locations."
        ),
        "total_pages": len(pages),
        "total_cross_references": sum(len(v) for v in all_xrefs.values()),
        "structure": chapters,
        "pages": pages,
        "cross_references": all_xrefs,
    }


def save_doc_index(index: Dict[str, Any], output_dir: str) -> None:
    """Persist doc PageIndex. Writes the same files as ``save_index`` plus ``cross_references.json``.

    ``cross_references`` is written to its own file only; it is excluded from
    ``structure.json`` to avoid bloating that file for large codebases.
    """
    # Strip cross_references before delegating so save_index only writes
    # structure.json / pages.json / page_map.json (no duplicate data).
    slim = {k: v for k, v in index.items() if k != "cross_references"}
    save_index(slim, output_dir)
    xref_path = Path(output_dir) / "cross_references.json"
    with open(xref_path, "w", encoding="utf-8") as fh:
        json.dump(index.get("cross_references", {}), fh, indent=2)
    logger.info("saved doc cross-references → %s", xref_path)
