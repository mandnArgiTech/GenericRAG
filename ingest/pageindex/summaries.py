"""Optional LLM-powered summarisation for PageIndex nodes.

Kept separate so the main ``ingest.pageindex`` package has *zero* LLM deps
when users skip summaries. This module uses ``langchain_ollama`` (already a
hard dep of GenericRAG), so no new requirements are introduced.

Author: deviprasad
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger("ingest.pageindex.summaries")


def _ollama_generate(model: str, prompt: str, system: Optional[str] = None) -> str:
    """Tiny completion wrapper — uses langchain_ollama.ChatOllama for portability."""
    try:
        from langchain_ollama import ChatOllama  # type: ignore
        from langchain_core.messages import HumanMessage, SystemMessage  # type: ignore
    except ImportError:
        logger.warning("langchain_ollama not available; skipping LLM summaries")
        return ""

    llm = ChatOllama(model=model, temperature=0.1, num_predict=150)
    msgs: List[Any] = []
    if system:
        msgs.append(SystemMessage(content=system))
    msgs.append(HumanMessage(content=prompt))
    try:
        resp = llm.invoke(msgs)
        return (resp.content or "").strip().split("\n", 1)[0][:300]
    except Exception as exc:  # pragma: no cover
        logger.warning("LLM summary failed: %s", exc)
        return ""


_FILE_SYSTEM_PROMPT = (
    "You summarize C source files in 1-2 sentences. Be technical and concise."
)
_DIR_SYSTEM_PROMPT = (
    "You summarize code directories in 1-2 sentences. Be technical and concise."
)


def _file_prompt(page: Dict[str, Any]) -> str:
    md = page["metadata"]
    lines_preview = "\n".join(page["content"].split("\n")[:60])
    parts = [f"File: {page['filepath']}"]
    if md["functions"]:
        parts.append("Functions: " + ", ".join(md["functions"][:8]))
    if md["structs"]:
        parts.append("Structs: " + ", ".join(md["structs"][:5]))
    if md["includes"]:
        parts.append("Includes: " + ", ".join(md["includes"][:6]))
    return (
        "Summarize this C source file in 1-2 sentences.\n"
        + "\n".join(parts)
        + "\n\nFirst 60 lines:\n"
        + lines_preview
        + "\n\nSummary:"
    )


def _dir_prompt(node: Dict[str, Any]) -> str:
    children = "\n".join(
        f"- {c.get('title')}: {(c.get('summary') or '')[:80]}"
        for c in (node.get("nodes") or [])[:15]
    )
    return (
        f"Summarize this source directory in 1-2 sentences.\nDirectory: {node['title']}\n"
        f"Contents:\n{children}\n\nSummary:"
    )


def add_llm_summaries(
    tree: List[Dict[str, Any]],
    pages: List[Dict[str, Any]],
    *,
    model: str,
    batch_log_every: int = 25,
) -> None:
    """Populate ``node['summary']`` for every node. Runs sequentially (SmolLM2 is fast on one GPU)."""
    page_map = {p["page_number"]: p for p in pages}

    file_nodes: List[Dict[str, Any]] = []

    def _collect(nodes: List[Dict[str, Any]]) -> None:
        for n in nodes:
            if "filepath" in n:
                file_nodes.append(n)
            if "nodes" in n:
                _collect(n["nodes"])

    _collect(tree)
    logger.info("summarising %s file nodes via %s", len(file_nodes), model)
    for i, fn in enumerate(file_nodes, 1):
        pg = page_map.get(fn["start_index"])
        if pg:
            fn["summary"] = _ollama_generate(model, _file_prompt(pg), _FILE_SYSTEM_PROMPT) or (
                f"{fn.get('line_count', '?')} lines"
            )
        if i % batch_log_every == 0:
            logger.info("  summarised %s/%s files", i, len(file_nodes))

    def _walk_dirs(nodes: List[Dict[str, Any]]) -> None:
        for n in nodes:
            if "filepath" in n:
                continue
            if "nodes" in n:
                _walk_dirs(n["nodes"])
            n["summary"] = _ollama_generate(model, _dir_prompt(n), _DIR_SYSTEM_PROMPT) or (
                f"{len(n.get('nodes') or [])} children"
            )

    _walk_dirs(tree)
