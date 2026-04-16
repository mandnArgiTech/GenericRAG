"""Language-aware and generic text splitting for source paths.

Author: deviprasad
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

from langchain_text_splitters import Language, RecursiveCharacterTextSplitter


def language_split(path: Path, content: str, lang: Language) -> List[Tuple[str, Dict[str, str]]]:
    splitter = RecursiveCharacterTextSplitter.from_language(
        language=lang, chunk_size=2000, chunk_overlap=200
    )
    docs = splitter.create_documents([content], metadatas=[{"path": str(path)}])
    return [
        (
            d.page_content,
            {
                "chunk_strategy": "language",
                "chunk_type": "fragment",
                "chunk_name": path.stem,
                "chunk_index": str(i),
            },
        )
        for i, d in enumerate(docs)
    ]


def generic_split(content: str, path: Path, size: int = 2000) -> List[Tuple[str, Dict[str, str]]]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=size, chunk_overlap=200)
    docs = splitter.create_documents([content], metadatas=[{"path": str(path)}])
    return [
        (
            d.page_content,
            {
                "chunk_strategy": "generic",
                "chunk_type": "fragment",
                "chunk_name": path.stem,
                "chunk_index": str(i),
            },
        )
        for i, d in enumerate(docs)
    ]


def sentence_window(text: str, path: Path) -> List[Tuple[str, Dict[str, str]]]:
    """Small chunks for retrieval; wider context_window in metadata for prompt expansion."""
    chunk_size, overlap = 300, 60
    splitter = RecursiveCharacterTextSplitter.from_language(
        language=Language.MARKDOWN, chunk_size=chunk_size, chunk_overlap=overlap
    )
    docs = splitter.create_documents([text])
    out: List[Tuple[str, Dict[str, str]]] = []
    for i, d in enumerate(docs):
        content = d.page_content
        needle = content[: min(80, len(content))] if content else ""
        idx = text.find(needle) if needle else -1
        if idx < 0:
            idx = 0  # pragma: no cover
        half = 600
        ctx_start = max(0, idx - half)
        ctx_end = min(len(text), idx + len(content) + half)
        context_window = text[ctx_start:ctx_end]
        if len(context_window) > 12000:
            context_window = context_window[:12000]  # pragma: no cover
        out.append(
            (
                content,
                {
                    "chunk_strategy": "sentence_window",
                    "chunk_type": "fragment",
                    "chunk_name": path.stem,
                    "chunk_index": str(i),
                    "context_window": context_window,
                },
            )
        )
    return out


def js_ts_lang(ext: str) -> Language:
    if ext in (".ts", ".tsx"):
        return getattr(Language, "TS", getattr(Language, "TYPESCRIPT", Language.HTML))
    return getattr(Language, "JS", getattr(Language, "JAVASCRIPT", Language.HTML))
