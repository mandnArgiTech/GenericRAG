"""PageIndex — Vectorless, reasoning-based RAG layer for GenericRAG.

Adds a hierarchical tree index alongside the existing vector (Chroma) pipeline.
Two index types share the same retrieval surface:

- ``code``  : directory → file tree for ``.c``/``.h`` sources
- ``docs``  : chapter → section tree for structured markdown chapters

Pages (whole files) are the atomic retrieval unit. The LLM reasons over the
tree (``get_document_structure``) and fetches full pages on demand
(``get_page_content``). No embeddings, no chunking.

Layout: ``ingest.pageindex.code_index`` (C/H source tree),
``ingest.pageindex.doc_index`` (markdown chapters + code cross-refs),
``ingest.pageindex.retrieve`` (tool-facing helpers used by ``mcp_server``).

Author: deviprasad
"""
from __future__ import annotations

from ingest.pageindex.code_index import build_code_page_index, save_index, load_index
from ingest.pageindex.doc_index import build_doc_page_index
from ingest.pageindex.retrieve import PageIndexStore

__all__ = [
    "PageIndexStore",
    "build_code_page_index",
    "build_doc_page_index",
    "load_index",
    "save_index",
]
