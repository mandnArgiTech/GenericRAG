"""Map paths and source types to chunk callables (orchestration entry point).

Author: deviprasad
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from langchain_text_splitters import Language

from ingest.chunking.code.code_ast_python import ast_chunk_python
from ingest.chunking.code.code_languages import generic_split, js_ts_lang, language_split, sentence_window
from ingest.chunking.code.code_regex_split import regex_code_split
from ingest.chunking.code.code_scheme import chunk_scheme
from ingest.chunking.code.code_tree_sitter import ts_extract_chunks
from ingest.core.constants import CONFIG_EXTS, STRATEGY_SIZE_LIMIT_MB
from ingest.chunking.docs_community import chunk_community, chunk_release_notes
from ingest.chunking.markdown_chunking import chunk_markdown_domain
from ingest.chunking.mib_chunking import chunk_mib
from ingest.chunking.rfc.rfc_chunking import chunk_rfc
from ingest.chunking.wiki_chunking import chunk_wiki_page

ChunkPairList = List[Tuple[str, Dict[str, str]]]
StrategyFn = Callable[..., ChunkPairList]


def choose_strategy_for_path(
    path: Path,
    source_type: str,
    mib_keep_deprecated: bool = False,
    embed_model: Optional[str] = None,
) -> Tuple[str, StrategyFn, int]:
    ext = path.suffix.lower()
    em = (embed_model or os.environ.get("EMBEDDING_MODEL", "nomic-embed-text") or "nomic-embed-text").strip()
    code_limit = STRATEGY_SIZE_LIMIT_MB["config"] if ext in CONFIG_EXTS else STRATEGY_SIZE_LIMIT_MB["code"]
    if source_type == "code":
        if ext == ".py":
            return "code", lambda p, c: ast_chunk_python(p, c), code_limit
        if ext == ".c":
            return (
                "code",
                lambda p, c: ts_extract_chunks(p, c, "c") or language_split(p, c, Language.C),
                code_limit,
            )
        if ext in (".cpp", ".cxx", ".cc", ".h", ".hpp", ".hxx"):
            return (
                "code",
                lambda p, c: ts_extract_chunks(p, c, "cpp") or language_split(p, c, Language.CPP),
                code_limit,
            )
        if ext == ".java":
            return (
                "code",
                lambda p, c: ts_extract_chunks(p, c, "java") or language_split(p, c, Language.JAVA),
                code_limit,
            )
        if ext == ".scm":
            return "code", lambda p, c: chunk_scheme(c, p), code_limit
        if ext in (".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs"):
            lg = js_ts_lang(ext)
            return "code", lambda p, c, lg=lg: language_split(p, c, lg), code_limit
        if ext in (".md", ".txt"):
            return "code", lambda p, c: sentence_window(c, p), code_limit  # pragma: no cover
        return "code", lambda p, c: regex_code_split(c, p, ext), code_limit
    if source_type in ("domain_doc", "theory"):
        lim = STRATEGY_SIZE_LIMIT_MB["theory" if source_type == "theory" else "domain_doc"]
        if ext in (".md", ".txt", ".rst"):
            return source_type, lambda p, c, _em=em: chunk_markdown_domain(c, str(p), embed_model=_em), lim
        return source_type, lambda p, c: generic_split(c, p, 2000), lim  # pragma: no cover
    if source_type == "rfc":
        return (
            "rfc",
            lambda p, c, _em=em: chunk_rfc(c, str(p), embed_model=_em),
            STRATEGY_SIZE_LIMIT_MB["rfc"],
        )
    if source_type == "mib":
        sk = not mib_keep_deprecated
        return "mib", lambda p, c, _sk=sk: chunk_mib(c, p, skip_deprecated=_sk), STRATEGY_SIZE_LIMIT_MB["mib"]
    if source_type == "release_notes":
        return "release_notes", lambda p, c: chunk_release_notes(c, str(p)), STRATEGY_SIZE_LIMIT_MB["release_notes"]
    if source_type == "community":
        return "community", lambda p, c: chunk_community(c, str(p), {}), STRATEGY_SIZE_LIMIT_MB["community"]
    if source_type == "wiki":
        return "wiki", lambda p, c: chunk_wiki_page(c, str(p), {}), STRATEGY_SIZE_LIMIT_MB["wiki"]
    return "default", lambda p, c: generic_split(c, p, 2000), STRATEGY_SIZE_LIMIT_MB["default"]
