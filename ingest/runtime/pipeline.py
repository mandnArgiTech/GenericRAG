"""CLI entrypoints and single-document ingest; batch orchestration lives in ``ingest_job``.

Author: deviprasad
"""
from __future__ import annotations

import argparse
import os
import signal
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeoutError
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import chromadb
from langchain_ollama import OllamaEmbeddings

from ingest.core.checkpoints import validate_embedding_dimension, write_ingestion_config
from ingest.core.constants import INGESTION_VERSION, SCRIPT_DIR
from ingest.runtime.ingest_job import ingest_run
from ingest.chunking.markdown_chunking import chunk_markdown_domain
from ingest.core.metadata_utils import (
    detect_content_type,
    empty_metadata,
    extract_concepts,
    finalize_metadata,
    format_concepts_field,
    iter_concept_ids,
    load_concept_registry,
    make_chunk_id,
    resolve_collection,
)
from ingest.chunking.rfc.rfc_chunking import _is_rfc_file, chunk_rfc
from ingest.core.state import _embed_lock, shutdown_event


def _handle_sig(*_args) -> None:
    shutdown_event.set()  # pragma: no cover


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Universal domain RAG ingestion")
    p.add_argument(
        "--mode",
        choices=[
            "code",
            "domain",
            "rfc",
            "rally",
            "customer",
            "mib",
            "wiki",
            "release-notes",
            "theory",
            "community",
            "status",
        ],
        default=os.environ.get("INGEST_MODE", "").strip() or None,
    )
    p.add_argument("--source", default=os.environ.get("SOURCE_FOLDER", "").strip() or None)
    p.add_argument("--domain", default=os.environ.get("INGEST_DOMAIN", "general"))
    p.add_argument("--collection", default=os.environ.get("CHROMA_COLLECTION", "").strip() or None)
    p.add_argument(
        "--db-path",
        default=os.environ.get("DB_PATH", "").strip()
        or str(SCRIPT_DIR / "Studio-Portable-RAG" / "VectorDB"),
    )
    p.add_argument("--rally-project", default=os.environ.get("RALLY_PROJECT", "").strip() or None)
    p.add_argument("--rally-filter", default=os.environ.get("RALLY_FILTER", "").strip() or None)
    p.add_argument("--confluence-space", default=os.environ.get("CONFLUENCE_SPACE", "").strip() or None)
    p.add_argument("--concept-registry", default=str(SCRIPT_DIR / "concept_registry.json"))
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--force", action="store_true")
    p.add_argument(
        "--recreate-collection",
        action="store_true",
        help=(
            "If the target Chroma collection's embedding dimension does not match the "
            "current model, delete that collection and its checkpoint entry, then ingest "
            "fresh (destructive for that collection only)."
        ),
    )
    p.add_argument(
        "--clean-stale",
        action="store_true",
        help="Delete Chroma chunks for sources removed from disk (checkpoint always pruned)",
    )
    p.add_argument(
        "--mib-keep-deprecated",
        action="store_true",
        help="Ingest deprecated/obsolete MIB objects (default: skip)",
    )
    p.add_argument(
        "--confluence-label",
        default=os.environ.get("CONFLUENCE_LABEL", "").strip() or None,
        help="Filter wiki pages by label (Confluence API)",
    )
    p.add_argument(
        "--git-diff",
        action="store_true",
        help="Only ingest files changed vs --git-diff-base (git); deletes removed paths from Chroma",
    )
    p.add_argument(
        "--git-diff-base",
        default=os.environ.get("GIT_DIFF_BASE", "").strip() or None,
        help="Git ref to diff against (default: last stored ingest ref or HEAD~1)",
    )
    p.add_argument("--verbose", action="store_true")
    return p


def _embed_documents_with_optional_timeout(
    embedder: OllamaEmbeddings, texts: List[str], timeout_sec: Optional[float]
) -> List[List[float]]:
    if not timeout_sec or timeout_sec <= 0:
        return embedder.embed_documents(texts)
    with ThreadPoolExecutor(max_workers=1) as pool:
        fut = pool.submit(embedder.embed_documents, texts)
        try:
            return fut.result(timeout=timeout_sec)
        except FuturesTimeoutError as exc:
            raise RuntimeError(f"Ollama embedding timed out after {timeout_sec}s") from exc


def feed_domain_document(
    filepath: str,
    domain: str,
    db_path: str,
    embed_model: str,
    concept_registry_path: Optional[str] = None,
    *,
    source_type: str = "auto",
    chroma_client: Optional[Any] = None,
    embedder: Optional[OllamaEmbeddings] = None,
    use_embed_lock: bool = True,
    embed_batch_timeout: Optional[float] = None,
    embed_lock_acquire_timeout: float = 300.0,
) -> Dict[str, Any]:
    """In-process single-file domain ingest (for MCP). Returns structured stats."""
    path = Path(filepath).resolve()
    text = path.read_text(encoding="utf-8", errors="replace")
    st = (source_type or "auto").strip().lower()
    if st not in ("auto", "rfc", "domain_doc"):
        st = "auto"
    if st == "rfc":
        use_rfc_chunker = True  # pragma: no cover
    elif st == "domain_doc":
        use_rfc_chunker = False
    else:
        use_rfc_chunker = _is_rfc_file(path)
    if use_rfc_chunker:
        parts = chunk_rfc(text, str(path), embed_model=embed_model)
        effective_source_type = "rfc"
        coll_mode = "rfc"
    else:
        parts = chunk_markdown_domain(text, str(path), embed_model=embed_model)
        effective_source_type = "domain_doc"
        coll_mode = "domain"
    reg = load_concept_registry(
        Path(concept_registry_path) if concept_registry_path else SCRIPT_DIR / "concept_registry.json"
    )
    dom = domain or "nms"
    concepts_all: set = set()
    sections: set = set()
    abs_src = str(path)
    coll_name = resolve_collection(coll_mode, dom, None)
    dbp = Path(db_path).resolve()
    dbp.mkdir(parents=True, exist_ok=True)
    client = chroma_client or chromadb.PersistentClient(path=str(dbp))
    coll = client.get_or_create_collection(name=coll_name)
    try:
        coll.delete(where={"source": abs_src})
    except Exception:  # pragma: no cover
        pass  # pragma: no cover
    embedder = embedder or OllamaEmbeddings(model=embed_model)
    dim_err = validate_embedding_dimension(coll, embedder, coll_name, embed_model)
    if dim_err:
        raise RuntimeError(dim_err)
    ingestion_ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    work: List[Tuple[str, str, Dict[str, str]]] = []
    for i, (chunk_text, partial) in enumerate(parts):
        sections.add(str(partial.get("section") or partial.get("section_number") or ""))
        ctype = partial.get("content_type") or detect_content_type(chunk_text)
        raw_c = partial.get("concepts")
        if raw_c is not None and str(raw_c).strip() != "":
            rc = str(raw_c).strip()  # pragma: no cover
            concepts = rc if rc.startswith("|") else format_concepts_field(iter_concept_ids(rc))  # pragma: no cover
        else:
            concepts = extract_concepts(chunk_text, dom, reg)
        for c in iter_concept_ids(concepts):
            concepts_all.add(c)  # pragma: no cover
        base = empty_metadata()
        base.update(
            {
                "source": abs_src,
                "source_type": effective_source_type,
                "domain": dom,
                "repository": "feed",
                "relative_path": path.name,
                "extension": path.suffix.lower(),
                "file_size_kb": str(round(path.stat().st_size / 1024.0, 3)),
                "last_modified": datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).strftime(
                    "%Y-%m-%dT%H:%M:%SZ"
                ),
                "ingestion_date": ingestion_ts,
                "ingestion_version": INGESTION_VERSION,
                "content_type": ctype,
                "concepts": concepts,
            }
        )
        meta = finalize_metadata({**base, **partial})
        try:
            cidx = int(meta.get("chunk_index") or i)
        except ValueError:
            cidx = i
        cid = make_chunk_id(abs_src, cidx, chunk_text)
        work.append((cid, chunk_text, meta))
    bs = 8
    for j in range(0, len(work), bs):
        batch = work[j : j + bs]
        ids = [x[0] for x in batch]
        texts = [x[1] for x in batch]
        metas = [x[2] for x in batch]

        def _do_embed() -> List[List[float]]:
            return _embed_documents_with_optional_timeout(embedder, texts, embed_batch_timeout)

        if use_embed_lock:
            acquired = _embed_lock.acquire(timeout=embed_lock_acquire_timeout)
            if not acquired:
                raise RuntimeError(
                    "Could not acquire embedding lock within "
                    f"{embed_lock_acquire_timeout}s; another ingest may be stuck."
                )
            try:
                embs = _do_embed()
            finally:
                _embed_lock.release()
        else:
            embs = _do_embed()
        coll.upsert(ids=ids, documents=texts, metadatas=metas, embeddings=embs)
    write_ingestion_config(dbp, embed_model)
    return {
        "chunk_count": len(parts),
        "sections_found": len([s for s in sections if s]),
        "concepts_found": sorted(concepts_all),
        "collection": coll_name,
        "source": abs_src,
    }


def main(argv: Optional[List[str]] = None) -> int:
    signal.signal(signal.SIGINT, _handle_sig)
    signal.signal(signal.SIGTERM, _handle_sig)
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if not args.mode:
        if args.source or os.environ.get("SOURCE_FOLDER"):
            args.mode = "code"
            args.domain = args.domain or "general"
        else:
            parser.error("--mode is required unless SOURCE_FOLDER is set (legacy code ingest)")
    if args.mode != "status":
        if args.mode not in ("rally", "wiki") and not args.source and not os.environ.get("SOURCE_FOLDER"):
            parser.error("--source required for this mode (or set SOURCE_FOLDER)")  # pragma: no cover
    return ingest_run(args)
