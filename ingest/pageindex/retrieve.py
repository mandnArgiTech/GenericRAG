"""Retrieval-side helper used by ``mcp_server.py``.

Exposes the PageIndex 3-tool pattern (``get_document``, ``get_document_structure``,
``get_page_content``) plus cross-reference helpers for the doc index.

Author: deviprasad
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("ingest.pageindex.retrieve")


def _parse_pages(pages: str) -> List[int]:
    """Parse ``'5-7'``, ``'3,8'``, or ``'12'`` → sorted list of ints."""
    result: List[int] = []
    for part in pages.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-", 1)
            s, e = int(start.strip()), int(end.strip())
            if s > e:
                raise ValueError(f"Invalid range '{part}'")
            result.extend(range(s, e + 1))
        else:
            result.append(int(part))
    return sorted(set(result))


def _strip_fields(obj: Any, fields: set) -> Any:
    if isinstance(obj, dict):
        return {k: _strip_fields(v, fields) for k, v in obj.items() if k not in fields}
    if isinstance(obj, list):
        return [_strip_fields(x, fields) for x in obj]
    return obj


class PageIndexStore:
    """Lightweight reader around a saved PageIndex (``structure.json``, ``pages.json``).

    The class serves both code and doc indexes — their on-disk layouts are identical
    except for a single optional ``cross_references.json`` file written by the doc builder.
    """

    def __init__(self, index_dir: str):
        self.index_dir = Path(index_dir)
        if not (self.index_dir / "structure.json").is_file():
            raise FileNotFoundError(
                f"No PageIndex at {index_dir}. Run: "
                f"python -m ingest --mode pageindex-code --source <dir>"
            )
        with open(self.index_dir / "structure.json", "r", encoding="utf-8") as fh:
            meta = json.load(fh)
        self.kind: str = meta.get("index_kind", "code")
        self.doc_name: str = meta.get("doc_name", "unknown")
        self.doc_description: str = meta.get("doc_description", "")
        self.total_pages: int = meta.get("total_pages", 0)
        self.structure: List[Dict[str, Any]] = meta.get("structure", [])
        self.total_cross_references: int = meta.get("total_cross_references", 0)

        self._pages: Optional[Dict[int, Dict[str, Any]]] = None
        self._page_map: Optional[Dict[int, str]] = None
        self._cross_refs: Optional[Dict[str, Dict[str, str]]] = None

        map_path = self.index_dir / "page_map.json"
        if map_path.is_file():
            with open(map_path, "r", encoding="utf-8") as fh:
                self._page_map = {int(k): v for k, v in json.load(fh).items()}

    # --- lazy loaders -------------------------------------------------------

    def _ensure_pages(self) -> Dict[int, Dict[str, Any]]:
        if self._pages is None:
            with open(self.index_dir / "pages.json", "r", encoding="utf-8") as fh:
                self._pages = {p["page"]: p for p in json.load(fh)}
        return self._pages

    def _ensure_xrefs(self) -> Dict[str, Dict[str, str]]:
        if self._cross_refs is None:
            xref_path = self.index_dir / "cross_references.json"
            if xref_path.is_file():
                with open(xref_path, "r", encoding="utf-8") as fh:
                    self._cross_refs = json.load(fh)
            else:
                self._cross_refs = {}
        return self._cross_refs

    # --- tool-facing API ----------------------------------------------------

    def get_document(self) -> str:
        return json.dumps(
            {
                "doc_name": self.doc_name,
                "doc_description": self.doc_description,
                "total_pages": self.total_pages,
                "status": "completed",
                "kind": self.kind,
                "cross_references": self.total_cross_references or None,
            },
            indent=2,
        )

    def get_document_structure(self, include_metadata: bool = True) -> str:
        tree = self.structure
        if not include_metadata:
            tree = _strip_fields(
                tree,
                {"functions", "structs", "defines", "includes", "code_references"},
            )
        return json.dumps(tree, indent=2, ensure_ascii=False)

    def get_page_content(self, pages: str) -> str:
        try:
            page_nums = _parse_pages(pages)
        except (ValueError, AttributeError) as exc:
            return json.dumps({"error": f"Invalid pages={pages!r}: {exc}"})
        pm = self._ensure_pages()
        out: List[Dict[str, Any]] = []
        for pn in page_nums:
            page = pm.get(pn)
            if page:
                out.append(
                    {
                        "page": pn,
                        "filepath": page["filepath"],
                        "content": page["content"],
                    }
                )
            else:
                out.append(
                    {
                        "page": pn,
                        "error": f"Page {pn} not found (valid: 1-{self.total_pages})",
                    }
                )
        return json.dumps(out, ensure_ascii=False)

    def get_page_by_filepath(self, filepath: str) -> str:
        pm = self._ensure_pages()
        for pn, page in pm.items():
            if page["filepath"] == filepath:
                return json.dumps({"page": pn, **page}, ensure_ascii=False)
        matches = [
            {"page": pn, "filepath": pg["filepath"]}
            for pn, pg in pm.items()
            if filepath in pg["filepath"]
        ]
        if len(matches) == 1:
            pg = pm[matches[0]["page"]]
            return json.dumps({"page": matches[0]["page"], **pg}, ensure_ascii=False)
        if matches:
            return json.dumps({"error": f"Multiple matches for {filepath!r}", "suggestions": matches[:10]})
        return json.dumps({"error": f"File not found: {filepath}"})

    def list_pages(self) -> str:
        if self._page_map:
            return json.dumps(
                [{"page": k, "filepath": v} for k, v in sorted(self._page_map.items())],
                indent=2,
            )
        pm = self._ensure_pages()
        return json.dumps(
            [{"page": p["page"], "filepath": p["filepath"]} for p in sorted(pm.values(), key=lambda x: x["page"])],
            indent=2,
        )

    # --- doc-only helpers ---------------------------------------------------

    def find_function(self, function_name: str) -> str:
        """Scan tree metadata for a function definition. Useful on a code index."""
        hits: List[Dict[str, Any]] = []

        def _walk(nodes: List[Dict[str, Any]]) -> None:
            for node in nodes:
                funcs = node.get("functions") or []
                if isinstance(funcs, str):
                    funcs = funcs.split()
                if function_name in funcs:
                    hits.append(
                        {
                            "filepath": node.get("filepath", node.get("title")),
                            "page": node.get("start_index"),
                            "summary": node.get("summary", ""),
                            "all_functions": funcs[:20],
                        }
                    )
                if node.get("nodes"):
                    _walk(node["nodes"])

        _walk(self.structure)
        return json.dumps(
            {
                "function": function_name,
                "matches": hits,
                "count": len(hits),
            },
            ensure_ascii=False,
        )

    def find_chapters_for_function(self, function_name: str) -> str:
        """Doc-index only: return chapters whose ``code_references`` mention ``function_name``."""
        if self.kind != "docs":
            return json.dumps({"error": "find_chapters_for_function is only valid on a doc PageIndex"})
        xrefs_all = self._ensure_xrefs()
        out: List[Dict[str, Any]] = []
        for ch in self.structure:
            refs = (ch.get("code_references") or {}).get("functions") or []
            if function_name in refs:
                doc_xrefs = xrefs_all.get(ch.get("filepath", ""), {})
                out.append(
                    {
                        "chapter_page": ch.get("start_index"),
                        "title": ch.get("title"),
                        "summary": ch.get("summary", ""),
                        "code_file_link": doc_xrefs.get(function_name),
                    }
                )
        return json.dumps(
            {"function": function_name, "chapters": out, "count": len(out)},
            ensure_ascii=False,
        )
