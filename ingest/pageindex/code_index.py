"""Build a hierarchical PageIndex over C/H source trees.

Each source file is a *page*. The tree mirrors the directory structure,
with file metadata (functions, structs, includes) attached to leaf nodes.
Saved artifacts (``structure.json``, ``pages.json``, ``page_map.json``)
are consumed by :class:`ingest.pageindex.retrieve.PageIndexStore`.

Index-time summaries are optional and use Ollama via ``langchain_ollama``
to match the rest of the ingest pipeline.

Author: deviprasad
"""
from __future__ import annotations

import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

logger = logging.getLogger("ingest.pageindex.code")

SOURCE_EXTENSIONS = {".c", ".h", ".cpp", ".cc", ".hpp", ".hxx"}
MAX_FILE_SIZE = 500_000  # 500 KB — skip generated / mega headers
DEFAULT_IGNORED_DIRS = {
    ".git", ".svn", "build", "Build", "__pycache__", "node_modules",
    "dist", "out", ".vscode", ".idea", "cmake-build-debug",
}


# ---------------------------------------------------------------------------
# Metadata extraction
# ---------------------------------------------------------------------------

_FUNC_RE = re.compile(r"^[\w\s\*]+?\b(\w+)\s*\([^)]*\)\s*\{", re.MULTILINE)
_INCLUDE_RE = re.compile(r'#include\s+[<"]([^>"]+)[>"]')
_DEFINE_RE = re.compile(r"#define\s+(\w+)")
_STRUCT_RE = re.compile(r"(?:typedef\s+)?struct\s+(\w+)")
_ENUM_RE = re.compile(r"enum\s+(\w+)")
_SECTION_CMT_RE = re.compile(r"/\*+\s*(.+?)\s*\*+/", re.DOTALL)

_CONTROL_KEYWORDS = {"if", "while", "for", "switch", "else", "return", "sizeof"}


def extract_c_metadata(content: str) -> Dict[str, Any]:
    """Parse structural metadata out of a C/H file."""
    includes = _INCLUDE_RE.findall(content)
    defines = _DEFINE_RE.findall(content)
    functions = [f for f in _FUNC_RE.findall(content) if f not in _CONTROL_KEYWORDS]
    structs = _STRUCT_RE.findall(content)
    enums = _ENUM_RE.findall(content)
    section_comments = [
        c.strip() for c in _SECTION_CMT_RE.findall(content)
        if 5 < len(c.strip()) < 120
    ]
    return {
        "includes": includes,
        "defines": defines[:50],
        "functions": functions,
        "structs": structs,
        "enums": enums,
        "section_comments": section_comments[:20],
        "line_count": content.count("\n") + 1,
    }


# ---------------------------------------------------------------------------
# Tree scanning
# ---------------------------------------------------------------------------

def scan_source_tree(
    source_dir: str | os.PathLike,
    extensions: Optional[Iterable[str]] = None,
    ignored_dirs: Optional[Iterable[str]] = None,
) -> List[Dict[str, Any]]:
    """Walk ``source_dir`` and return a flat list of *page* dicts."""
    source_path = Path(source_dir).resolve()
    if not source_path.is_dir():
        raise ValueError(f"Not a directory: {source_dir}")

    exts = set(extensions or SOURCE_EXTENSIONS)
    ignore = set(ignored_dirs or DEFAULT_IGNORED_DIRS)

    pages: List[Dict[str, Any]] = []
    for root, dirs, files in os.walk(source_path):
        dirs[:] = [d for d in dirs if not d.startswith(".") and d not in ignore]
        for fname in sorted(files):
            fpath = os.path.join(root, fname)
            suf = os.path.splitext(fname)[1].lower()
            if suf not in exts:
                continue
            try:
                fsize = os.path.getsize(fpath)
            except OSError:
                continue
            if fsize == 0 or fsize > MAX_FILE_SIZE:
                continue
            try:
                with open(fpath, "r", encoding="utf-8", errors="replace") as fh:
                    content = fh.read()
            except OSError:
                continue
            rel = os.path.relpath(fpath, source_path)
            meta = extract_c_metadata(content)
            pages.append(
                {
                    "page_number": len(pages) + 1,
                    "filepath": rel,
                    "filename": fname,
                    "directory": os.path.dirname(rel),
                    "content": content,
                    "metadata": meta,
                    "line_count": meta["line_count"],
                    "file_size": fsize,
                }
            )
    return pages


# ---------------------------------------------------------------------------
# Tree building
# ---------------------------------------------------------------------------

def build_directory_tree(pages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Nest pages under their directory path → PageIndex-style tree."""
    counter = [1]

    def _nid() -> str:
        nid = str(counter[0]).zfill(4)
        counter[0] += 1
        return nid

    by_dir: Dict[str, List[Dict[str, Any]]] = {}
    for p in pages:
        by_dir.setdefault(p["directory"] or "(root)", []).append(p)

    dir_nodes: Dict[str, Dict[str, Any]] = {}
    for d in sorted(by_dir):
        file_nodes: List[Dict[str, Any]] = []
        for p in sorted(by_dir[d], key=lambda x: x["filename"]):
            md = p["metadata"]
            file_nodes.append(
                {
                    "title": p["filename"],
                    "node_id": _nid(),
                    "start_index": p["page_number"],
                    "end_index": p["page_number"],
                    "filepath": p["filepath"],
                    "line_count": p["line_count"],
                    "functions": md["functions"],
                    "structs": md["structs"],
                    "defines": md["defines"][:10],
                    "includes": md["includes"],
                }
            )
        start = min(p["page_number"] for p in by_dir[d])
        end = max(p["page_number"] for p in by_dir[d])
        dir_nodes[d] = {
            "title": d,
            "node_id": _nid(),
            "start_index": start,
            "end_index": end,
            "nodes": file_nodes,
        }

    # Nest sub-directories under parents
    consumed: set = set()
    for d in sorted(dir_nodes, key=lambda x: x.count("/"), reverse=True):
        parent = os.path.dirname(d)
        if parent and parent in dir_nodes and parent != d:
            dir_nodes[parent]["nodes"].append(dir_nodes[d])
            dir_nodes[parent]["start_index"] = min(
                dir_nodes[parent]["start_index"], dir_nodes[d]["start_index"]
            )
            dir_nodes[parent]["end_index"] = max(
                dir_nodes[parent]["end_index"], dir_nodes[d]["end_index"]
            )
            consumed.add(d)

    return [dir_nodes[d] for d in sorted(dir_nodes) if d not in consumed]


# ---------------------------------------------------------------------------
# Summaries (metadata-only; LLM variant lives in ``summaries.py`` if needed)
# ---------------------------------------------------------------------------

def add_metadata_summaries(tree: List[Dict[str, Any]], pages: List[Dict[str, Any]]) -> None:
    """Cheap, zero-LLM summaries derived from the extracted metadata."""
    page_map = {p["page_number"]: p for p in pages}

    def _walk(nodes: List[Dict[str, Any]]) -> None:
        for node in nodes:
            if "filepath" in node:
                p = page_map.get(node["start_index"])
                if not p:
                    continue
                md = p["metadata"]
                parts: List[str] = []
                if md["functions"]:
                    parts.append("Functions: " + ", ".join(md["functions"][:5]))
                if md["structs"]:
                    parts.append("Structs: " + ", ".join(md["structs"][:3]))
                if md["includes"]:
                    parts.append("Includes: " + ", ".join(md["includes"][:5]))
                node["summary"] = ". ".join(parts) if parts else f"{node['line_count']} lines"
            elif "nodes" in node:
                _walk(node["nodes"])
                files = sum(1 for c in node["nodes"] if "filepath" in c)
                subdirs = len(node["nodes"]) - files
                node["summary"] = f"{files} file(s), {subdirs} subdir(s)"

    _walk(tree)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_code_page_index(
    source_dir: str,
    *,
    use_llm_summaries: bool = False,
    summary_model: Optional[str] = None,
) -> Dict[str, Any]:
    """Build a PageIndex artefact in memory. Caller persists via :func:`save_index`."""
    t0 = time.time()
    pages = scan_source_tree(source_dir)
    logger.info(
        "scanned %s source files (%s lines)",
        len(pages),
        sum(p["line_count"] for p in pages),
    )
    tree = build_directory_tree(pages)

    if use_llm_summaries and summary_model:
        # Lazy import: keeps hard deps minimal for users who do not enable summaries.
        from ingest.pageindex.summaries import add_llm_summaries  # type: ignore
        add_llm_summaries(tree, pages, model=summary_model)
    else:
        add_metadata_summaries(tree, pages)

    page_list = [
        {"page": p["page_number"], "filepath": p["filepath"], "content": p["content"]}
        for p in pages
    ]
    logger.info("code PageIndex built in %.1fs", time.time() - t0)
    return {
        "index_kind": "code",
        "doc_name": os.path.basename(os.path.normpath(source_dir)),
        "doc_description": (
            f"Source tree with {len(pages)} files, "
            f"{sum(p['line_count'] for p in pages):,} total lines."
        ),
        "total_pages": len(pages),
        "structure": tree,
        "pages": page_list,
    }


def save_index(index: Dict[str, Any], output_dir: str) -> None:
    """Persist ``structure.json``, ``pages.json``, ``page_map.json`` to ``output_dir``."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    meta = {k: v for k, v in index.items() if k != "pages"}
    with open(out / "structure.json", "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2, ensure_ascii=False)

    with open(out / "pages.json", "w", encoding="utf-8") as fh:
        json.dump(index["pages"], fh, ensure_ascii=False)

    page_map = {p["page"]: p["filepath"] for p in index["pages"]}
    with open(out / "page_map.json", "w", encoding="utf-8") as fh:
        json.dump(page_map, fh, indent=2)

    logger.info(
        "saved PageIndex to %s (structure=%s KB, pages=%s KB)",
        out,
        (out / "structure.json").stat().st_size // 1024,
        (out / "pages.json").stat().st_size // 1024,
    )


def load_index(index_dir: str) -> Dict[str, Any]:
    """Load a saved PageIndex back into memory (structure + page contents)."""
    p = Path(index_dir)
    with open(p / "structure.json", "r", encoding="utf-8") as fh:
        data = json.load(fh)
    with open(p / "pages.json", "r", encoding="utf-8") as fh:
        data["pages"] = json.load(fh)
    return data
