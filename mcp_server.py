"""
mcp_server.py - MCP server for Universal Domain RAG (multi-collection Chroma).

Transport: JSON-RPC 2.0 over stdio. File-only logging (MCP_LOG / mcp_server.log).

Tools:
- search_knowledge   -- semantic search across routed collections
- search_codebase    -- backward compat; delegates to search_knowledge(code)
- search_concepts  -- metadata filter on concept tags
- feed_domain_doc  -- ingest markdown or RFC .txt into {domain}_domain or rfc collection
- list_repositories, get_db_stats, reconnect

Author: deviprasad
"""

from __future__ import annotations

import asyncio
import atexit
import json
import logging
import os
import signal
import subprocess
import sys
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Tuple

import chromadb
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from mcp.server.fastmcp import FastMCP

from domain_feeder import feed_domain_document
from hybrid_search import (
    HYBRID_AVAILABLE,
    get_bm25_index,
    reciprocal_rank_fusion,
    search_bm25_ranked_ids,
    stable_doc_id,
)
from ingest import iter_concept_ids

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DB_PATH = os.environ.get("DB_PATH", "./vector_db")

DIM_TO_MODEL = {
    1024: "mxbai-embed-large",
    768: "nomic-embed-text",
}


def detect_embedding_model(db_path: str) -> str:
    env_val = os.environ.get("EMBEDDING_MODEL", "").strip()
    if env_val:
        return env_val
    config_path = os.path.join(db_path, "ingestion_config.json")
    if os.path.exists(config_path):
        try:
            with open(config_path, encoding="utf-8") as fh:
                model = json.load(fh).get("embedding_model", "")
            if model:
                return model
        except Exception:
            pass
    try:
        client = chromadb.PersistentClient(path=db_path)
        cols = client.list_collections()
        if cols:
            col = client.get_collection(cols[0].name)
            rows = col.get(limit=1, include=["embeddings"])
            embs = rows.get("embeddings")
            if embs is not None and len(embs) > 0 and len(embs[0]) > 0:
                dim = len(embs[0])
                if dim in DIM_TO_MODEL:
                    return DIM_TO_MODEL[dim]
    except Exception:
        pass
    return "nomic-embed-text"


EMBEDDING_MODEL = detect_embedding_model(DB_PATH)
OLLAMA_EXE = os.environ.get("OLLAMA_EXE", "ollama/ollama.exe")
QUERY_TIMEOUT = int(os.environ.get("QUERY_TIMEOUT", "307"))
TOP_K_DEFAULT = int(os.environ.get("TOP_K", "5"))
MCP_POOL_WORKERS = max(1, int(os.environ.get("MCP_POOL_WORKERS", "4")))
MCP_MAX_CONCURRENT_FEEDS = max(1, int(os.environ.get("MCP_MAX_CONCURRENT_FEEDS", "2")))
MAX_K_RESULTS = max(1, int(os.environ.get("MCP_MAX_K", "25")))
RESULT_CHUNK_MAX_CHARS = max(512, int(os.environ.get("MCP_RESULT_CHUNK_MAX_CHARS", "4096")))
EMBED_BATCH_TIMEOUT = float(os.environ.get("MCP_EMBED_BATCH_TIMEOUT", "120"))
OLLAMA_HEARTBEAT_SEC = max(5, int(os.environ.get("MCP_OLLAMA_HEARTBEAT_SEC", "30")))
MCP_OLLAMA_HEARTBEAT_TIMEOUT = max(1, int(os.environ.get("MCP_OLLAMA_HEARTBEAT_TIMEOUT", "15")))
HYBRID_SEARCH = os.environ.get("HYBRID_SEARCH", "1").strip().lower() not in ("0", "false", "no")
RRF_K = float(os.environ.get("RRF_K", "60"))
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_LOG = os.path.join(SCRIPT_DIR, "mcp_server.log")
LOG_FILE = os.environ.get("MCP_LOG", "") or DEFAULT_LOG

# ---------------------------------------------------------------------------
# Logging — file only
# ---------------------------------------------------------------------------

log_path = os.path.abspath(LOG_FILE)
os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
log_handler = logging.FileHandler(log_path, encoding="utf-8")
log_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
logger = logging.getLogger("mcp_server")
logger.setLevel(logging.INFO)
logger.addHandler(log_handler)
logger.propagate = False

LANG_TAG = {
    ".py": "python",
    ".java": "java",
    ".js": "javascript",
    ".ts": "typescript",
    ".tsx": "tsx",
    ".jsx": "jsx",
    ".cs": "csharp",
    ".cpp": "cpp",
    ".c": "c",
    ".go": "go",
    ".rs": "rust",
    ".rb": "ruby",
    ".kt": "kotlin",
    ".scala": "scala",
    ".sql": "sql",
    ".sh": "bash",
    ".ps1": "powershell",
    ".yml": "yaml",
    ".yaml": "yaml",
    ".json": "json",
    ".xml": "xml",
    ".html": "html",
    ".md": "markdown",
    ".proto": "protobuf",
    ".properties": "properties",
}

collections_map: Dict[str, Chroma] = {}
ollama_proc: Optional[subprocess.Popen] = None

_collections_lock = threading.RLock()
_shared_chroma_client: Optional[chromadb.PersistentClient] = None
_shared_embedder: Optional[OllamaEmbeddings] = None
_mcp_executor: Optional[ThreadPoolExecutor] = None
_shutdown_event = threading.Event()
_feed_semaphore: Optional[asyncio.Semaphore] = None
_ollama_health_lock = threading.Lock()
_ollama_healthy = True


def _set_ollama_healthy(ok: bool) -> None:
    global _ollama_healthy
    with _ollama_health_lock:
        _ollama_healthy = ok


def ollama_is_healthy() -> bool:
    with _ollama_health_lock:
        return _ollama_healthy


def _collections_snapshot() -> Dict[str, Chroma]:
    with _collections_lock:
        return dict(collections_map)


def _collections_assign(new_map: Dict[str, Chroma]) -> None:
    global collections_map
    with _collections_lock:
        collections_map = new_map


def init_mcp_executor() -> ThreadPoolExecutor:
    global _mcp_executor
    if _mcp_executor is None:
        _mcp_executor = ThreadPoolExecutor(
            max_workers=MCP_POOL_WORKERS,
            thread_name_prefix="mcp",
        )
    return _mcp_executor


def _heartbeat_worker() -> None:
    while not _shutdown_event.is_set():
        if _shutdown_event.wait(OLLAMA_HEARTBEAT_SEC):
            break
        try:
            urllib.request.urlopen(
                "http://127.0.0.1:11434/api/tags",
                timeout=MCP_OLLAMA_HEARTBEAT_TIMEOUT,
            )
            _set_ollama_healthy(True)
        except Exception:
            _set_ollama_healthy(False)
            logger.warning("Ollama heartbeat failed; attempting recovery")
            try:
                start_ollama()
                _set_ollama_healthy(True)
            except Exception as exc:
                logger.error("Ollama recovery after heartbeat failed: %s", exc)


def _start_heartbeat_thread() -> None:
    t = threading.Thread(
        target=_heartbeat_worker,
        name="mcp-ollama-heartbeat",
        daemon=True,
    )
    t.start()


def _safe_count_chroma(vs: Chroma) -> int:
    try:
        return int(vs._collection.count())  # type: ignore[attr-defined]
    except Exception as exc:
        logger.warning("count via _collection failed: %s", exc)
        try:
            c = _shared_chroma_client or chromadb.PersistentClient(path=DB_PATH)
            col = c.get_collection(vs._collection.name)  # type: ignore[attr-defined]
            return int(col.count())
        except Exception as exc2:
            logger.warning("fallback count failed: %s", exc2)
            return 0


def discover_collections(
    db_path: str,
    embeddings: OllamaEmbeddings,
    chroma_client: Optional[chromadb.PersistentClient] = None,
) -> Dict[str, Chroma]:
    out: Dict[str, Chroma] = {}
    try:
        client = chroma_client or chromadb.PersistentClient(path=db_path)
        for info in client.list_collections():
            name = info.name
            out[name] = Chroma(
                collection_name=name,
                persist_directory=db_path,
                embedding_function=embeddings,
            )
    except Exception as exc:
        logger.error("discover_collections: %s", exc)
    return out


def _ollama_is_ready() -> bool:
    try:
        urllib.request.urlopen("http://localhost:11434", timeout=2)
        return True
    except Exception:
        return False


def start_ollama() -> None:
    global ollama_proc
    if _ollama_is_ready():
        logger.info("Reusing existing Ollama instance on :11434")
        _set_ollama_healthy(True)
        return
    if not os.path.exists(OLLAMA_EXE):
        raise RuntimeError(f"Ollama executable not found at '{OLLAMA_EXE}'.")
    logger.info("Starting Ollama: %s", OLLAMA_EXE)
    ollama_proc = subprocess.Popen(
        [OLLAMA_EXE, "serve"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    for _ in range(30):
        if _ollama_is_ready():
            logger.info("Ollama is ready")
            _set_ollama_healthy(True)
            return
        time.sleep(1)
    try:
        ollama_proc.terminate()
    except Exception:
        pass
    ollama_proc = None
    raise RuntimeError("Ollama did not become ready within 30 seconds.")


def stop_ollama() -> None:
    global ollama_proc
    if ollama_proc is None:
        return
    if ollama_proc.poll() is not None:
        ollama_proc = None
        return
    logger.info("Stopping Ollama (started by this MCP server)")
    try:
        ollama_proc.terminate()
        ollama_proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        ollama_proc.kill()
    except Exception as exc:
        logger.warning("Error stopping Ollama: %s", exc)
    finally:
        ollama_proc = None


def connect_chroma_with_retry() -> Dict[str, Chroma]:
    embeddings = _shared_embedder or OllamaEmbeddings(model=EMBEDDING_MODEL)
    chroma_client = _shared_chroma_client
    last_exc: Optional[Exception] = None
    for attempt in range(3):
        try:
            return discover_collections(DB_PATH, embeddings, chroma_client)
        except Exception as exc:
            last_exc = exc
            logger.warning("Chroma connect attempt %d failed: %s", attempt + 1, exc)
            time.sleep(5)
    logger.error("ChromaDB connection failed after retries: %s", last_exc)
    return {}


def startup() -> None:
    global _shared_embedder, _shared_chroma_client
    init_mcp_executor()
    logger.info("MCP server starting -- model: %s; db: %s", EMBEDDING_MODEL, DB_PATH)
    try:
        start_ollama()
    except Exception as exc:
        logger.error("Ollama startup failed: %s", exc)
        _set_ollama_healthy(False)
    _shared_embedder = OllamaEmbeddings(model=EMBEDDING_MODEL)
    _shared_chroma_client = chromadb.PersistentClient(path=DB_PATH)
    new_map = connect_chroma_with_retry()
    _collections_assign(new_map)
    cmap = _collections_snapshot()
    total = sum(_safe_count_chroma(v) for v in cmap.values())
    logger.info("ChromaDB -- %d collections, ~%d total chunks", len(cmap), total)
    _start_heartbeat_thread()


def shutdown(*args: Any) -> None:
    _shutdown_event.set()
    stop_ollama()
    global _mcp_executor
    if _mcp_executor is not None:
        try:
            try:
                _mcp_executor.shutdown(wait=False, cancel_futures=True)
            except TypeError:
                _mcp_executor.shutdown(wait=False)
        except Exception as exc:
            logger.warning("ThreadPoolExecutor shutdown: %s", exc)
        _mcp_executor = None


def _signal_handler(signum: int, frame: Any) -> None:
    logger.info("Signal %s received, shutting down MCP server resources", signum)
    shutdown()
    sys.exit(0 if signum == signal.SIGINT else 128 + signum)


atexit.register(shutdown)


def _infer_source_type(meta: Dict[str, Any]) -> str:
    st = (meta.get("source_type") or "").strip().lower()
    if st:
        return st
    cname = (meta.get("chunk_strategy") or "").lower()
    if "rfc" in cname:
        return "rfc"
    if "mib" in cname:
        return "mib"
    if "rally" in cname or meta.get("rally_id"):
        return "rally"
    if meta.get("ticket_id"):
        return "customer"
    if "wiki" in cname:
        return "wiki"
    if "community" in cname:
        return "community"
    if "release" in cname:
        return "release_notes"
    return "code"


def _fence_for(content: str) -> str:
    return "~~~" if "```" in content else "```"


def _truncate_chunk(text: str) -> str:
    mx = RESULT_CHUNK_MAX_CHARS
    if len(text) <= mx:
        return text
    return text[: mx - 50] + "\n\n... (truncated for MCP response size) ..."


def format_result(doc: Any, score: Optional[float], source_type: str) -> str:
    meta = doc.metadata or {}
    header: List[str] = []
    repo = meta.get("repository", "") or ""
    src = meta.get("relative_path", meta.get("source", "")) or ""
    cname = meta.get("chunk_name", "") or ""
    ctype = meta.get("chunk_type", "") or ""
    if score is not None:
        header.append(f"**Distance:** {score:.4f}")
    header.append(f"**source_type:** {source_type}")
    content = _truncate_chunk(doc.page_content.strip())
    ext = (meta.get("extension") or "").lower()

    if source_type == "code":
        if repo:
            header.append(f"**Repo:** {repo}")
        header.append(f"**File:** {repo + '/' + src if repo else src}")
        if cname:
            header.append(f"**Component:** {cname} ({ctype})")
        dep = (meta.get("dependencies") or "").strip()
        if dep:
            dshow = dep if len(dep) <= 500 else dep[:500] + "…"
            header.append(f"**dependencies:** {dshow}")
        lang = LANG_TAG.get(ext, "")
        fence = _fence_for(content)
        return "### Code\n" + "\n".join(header) + f"\n\n{fence}{lang}\n{content}\n{fence}"

    if source_type in ("domain_doc", "theory", "wiki"):
        sec = meta.get("section", meta.get("doc_title", "Domain Knowledge"))
        return f"### {sec}\n*Source: {meta.get('source', '')}*\n\n{content}"

    if source_type == "rfc":
        rfc = meta.get("rfc_number", "")
        sec = meta.get("section_number", "")
        st = meta.get("section_title", "")
        head = f"RFC {rfc} §{sec}"
        if st:
            head += f": {st}"
        return f"### {head}\n\n{content}"

    if source_type in ("rally", "customer"):
        tid = meta.get("rally_id", "") or meta.get("ticket_id", "")
        title = meta.get("chunk_name", "") or meta.get("Name", "")
        res = meta.get("has_resolution", "") == "true"
        line = f"[{tid}] {title}" if title else f"[{tid}]"
        if res:
            line += " — Resolution: (see body)"
        return f"### {line}\n\n{content}"

    if source_type == "mib":
        oid = meta.get("object_name", "")
        path = meta.get("oid_path", "")
        return f"### OID: {oid} ({path}) — Description\n\n{content}"

    if source_type == "community":
        plat = meta.get("source_platform", "unknown")
        return f"### Source: [{plat}] — {meta.get('source_url', '')}\n\n{content}"

    return "### Result\n" + "\n".join(header) + f"\n\n{content}"


def _domain_filter(names: List[str], domain: str) -> List[str]:
    d = domain.strip().lower()
    if not d or d == "general":
        return names
    return [n for n in names if d in n.lower()]


def _hybrid_candidate_cap(k: int, env_var: str) -> int:
    raw = os.environ.get(env_var, "").strip()
    if raw.isdigit():
        return max(1, int(raw))
    return max(40, k * 4)


def _select_collection_names(cmap: Dict[str, Chroma], search_type: str, domain: str) -> List[str]:
    names = list(cmap.keys())
    names = _domain_filter(names, domain)
    st = search_type.lower().strip()
    if st == "auto" or not st:
        return names
    if st == "code":
        return [n for n in names if n.endswith("_code")]
    if st == "domain":
        return [n for n in names if "_domain" in n or n == "theory"]
    if st == "troubleshoot":
        return [
            n
            for n in names
            if any(x in n for x in ("_domain", "community", "_customer", "_internal"))
        ]
    if st == "reference":
        return [n for n in names if n == "rfc" or n == "theory" or "_mib" in n or n.endswith("_releases")]
    return names


def _sync_multi_search(
    query: str,
    k: int,
    search_type: str,
    domain: str,
    repo_filter: str,
    cmap: Dict[str, Chroma],
) -> List[Tuple[Any, Optional[float], str]]:
    if not cmap:
        raise RuntimeError("ChromaDB has no collections.")
    targets = _select_collection_names(cmap, search_type, domain)
    per = max(1, k // max(1, len(targets)) if targets else k)
    use_hybrid = HYBRID_SEARCH and HYBRID_AVAILABLE
    if HYBRID_SEARCH and not HYBRID_AVAILABLE:
        logger.warning(
            "HYBRID_SEARCH is on but rank-bm25 is not installed; using dense-only. "
            "pip install rank-bm25"
        )

    dense_cap = _hybrid_candidate_cap(k, "HYBRID_DENSE_CANDIDATES")
    bm25_cap = _hybrid_candidate_cap(k, "HYBRID_BM25_CANDIDATES")
    db_abs = os.path.abspath(DB_PATH)

    if not use_hybrid:
        merged: List[Tuple[Any, Optional[float], str, str]] = []
        for name in targets:
            vs = cmap[name]
            flt: Optional[Dict[str, str]] = None
            if repo_filter.strip():
                flt = {"repository": repo_filter.strip()}
            try:
                try:
                    pairs = vs.similarity_search_with_score(query, k=per, filter=flt)
                except TypeError:
                    pairs = vs.similarity_search_with_score(query, k=per)
            except Exception as exc:
                logger.warning("Skipping collection %s: %s", name, exc)
                continue
            for doc, score in pairs:
                meta = doc.metadata or {}
                st = _infer_source_type(meta)
                if search_type.lower() == "troubleshoot":
                    ct = (meta.get("content_type") or "").lower()
                    if ct not in ("edge_case", "workaround", "bug_report"):
                        continue
                merged.append((doc, score, st, name))
        merged.sort(key=lambda x: x[1] if x[1] is not None else 1e9)
        out: List[Tuple[Any, Optional[float], str]] = []
        for doc, score, st, _cname in merged[:k]:
            out.append((doc, score, st))
        return out

    fused: List[Tuple[Any, Optional[float], str, float]] = []
    for name in targets:
        vs = cmap[name]
        flt: Optional[Dict[str, str]] = None
        if repo_filter.strip():
            flt = {"repository": repo_filter.strip()}
        try:
            try:
                pairs = vs.similarity_search_with_score(query, k=dense_cap, filter=flt)
            except TypeError:
                pairs = vs.similarity_search_with_score(query, k=dense_cap)
        except Exception as exc:
            logger.warning("Skipping collection %s: %s", name, exc)
            continue

        dense_ids: List[str] = []
        dense_map: Dict[str, Tuple[Any, float]] = {}
        for doc, score in pairs:
            sid = stable_doc_id(name, doc.metadata or {}, doc.page_content)
            dense_ids.append(sid)
            dense_map[sid] = (doc, score)

        bm25_ids: List[str] = []
        col = getattr(vs, "_collection", None)
        if col is not None:
            idx = get_bm25_index(db_abs, name)
            if idx.ensure_loaded(col):
                bm25_ids = search_bm25_ranked_ids(idx, query, bm25_cap, repo_filter)
        rank_lists = [dense_ids, bm25_ids] if bm25_ids else [dense_ids]
        rrf_scores = reciprocal_rank_fusion(rank_lists, k=RRF_K)
        if not rrf_scores:
            continue

        bm25_rank = {sid: r for r, sid in enumerate(bm25_ids)}
        dense_rank = {sid: r for r, sid in enumerate(dense_ids)}
        sorted_sids = sorted(
            rrf_scores.keys(),
            key=lambda sid: (
                -rrf_scores[sid],
                bm25_rank.get(sid, 10**9),
                dense_rank.get(sid, 10**9),
            ),
        )
        idx_ref = get_bm25_index(db_abs, name) if col is not None else None
        for sid in sorted_sids:
            if sid in dense_map:
                doc, _dscore = dense_map[sid]
            elif idx_ref is not None and sid in idx_ref.id_to_doc:
                text, meta = idx_ref.id_to_doc[sid]
                doc = Document(page_content=text, metadata=meta)
            else:
                continue
            meta = doc.metadata or {}
            st = _infer_source_type(meta)
            if search_type.lower() == "troubleshoot":
                ct = (meta.get("content_type") or "").lower()
                if ct not in ("edge_case", "workaround", "bug_report"):
                    continue
            fused.append((doc, None, st, rrf_scores[sid]))

    fused.sort(key=lambda x: x[3], reverse=True)
    return [(doc, score, st) for doc, score, st, _ in fused[:k]]


def _depend_stems_from_results(
    results: List[Tuple[Any, Optional[float], str]],
) -> List[str]:
    from pathlib import Path as _P

    stems: set = set()
    for doc, _, _ in results:
        meta = getattr(doc, "metadata", None) or {}
        rel = str(meta.get("relative_path") or "").strip()
        if rel:
            stem = _P(rel).stem
            if stem:
                stems.add(stem)
        cn = str(meta.get("chunk_name") or "").strip()
        if cn:
            stems.add(cn)
    return sorted(stems)


def _dependencies_where_comma_token(stem: str) -> Dict[str, Any]:
    """Match *stem* as a whole entry in comma-separated ``dependencies`` metadata."""
    s = (stem or "").strip()
    if not s:
        return {"dependencies": {"$eq": "__empty_dependency_stem__"}}
    return {
        "$or": [
            {"dependencies": {"$eq": s}},
            {"dependencies": {"$contains": f"{s}, "}},
            {"dependencies": {"$contains": f", {s}, "}},
            {"dependencies": {"$contains": f", {s}"}},
        ]
    }


def _sync_fetch_dependents(
    primary: List[Tuple[Any, Optional[float], str]],
    search_type: str,
    domain: str,
    repo_filter: str,
    cmap: Dict[str, Chroma],
    max_hits: int,
) -> List[Tuple[Any, Optional[float], str]]:
    """Chunks whose *dependencies* metadata references symbols from primary hits."""
    targets = _select_collection_names(cmap, search_type, domain)
    lookups = _depend_stems_from_results(primary)
    if not lookups or not targets:
        return []
    seen: set = set()
    out: List[Tuple[Any, Optional[float], str]] = []
    rf = repo_filter.strip()
    for stem in lookups:
        if not stem:
            continue
        where_dep = _dependencies_where_comma_token(stem)
        for name in targets:
            vs = cmap[name]
            col = getattr(vs, "_collection", None)
            if col is None:
                continue
            try:
                res = col.get(
                    where=where_dep,
                    limit=40,
                    include=["documents", "metadatas", "ids"],
                )
            except Exception as exc:
                logger.warning("dependents get failed on %s: %s", name, exc)
                continue
            ids_list = res.get("ids") or []
            docs = res.get("documents") or []
            metas = res.get("metadatas") or []
            for i, did in enumerate(ids_list):
                if did in seen:
                    continue
                meta = metas[i] if i < len(metas) else {}
                if rf and str((meta or {}).get("repository", "")).strip() != rf:
                    continue
                seen.add(did)
                text = docs[i] if i < len(docs) else ""
                doc = Document(page_content=text or "", metadata=dict(meta or {}))
                st = _infer_source_type(meta or {})
                out.append((doc, None, st))
                if len(out) >= max_hits:
                    return out
    return out


async def _run_search(
    query: str,
    k: int,
    search_type: str,
    domain: str,
    repo: str,
    include_dependents: bool = False,
) -> str:
    cmap = _collections_snapshot()
    loop = asyncio.get_running_loop()
    ex = init_mcp_executor()
    fut = loop.run_in_executor(
        ex,
        _sync_multi_search,
        query,
        k,
        search_type,
        domain,
        repo,
        cmap,
    )
    results = await asyncio.wait_for(fut, timeout=QUERY_TIMEOUT)
    if not results:
        return "No matching chunks found."
    parts = [format_result(d, s, st) for d, s, st in results]
    text = "\n\n---\n\n".join(parts)
    if include_dependents:

        def _dep() -> List[Tuple[Any, Optional[float], str]]:
            return _sync_fetch_dependents(
                results,
                search_type,
                domain,
                repo,
                cmap,
                max(1, min(20, k * 2)),
            )

        dep = await asyncio.wait_for(
            loop.run_in_executor(ex, _dep),
            timeout=QUERY_TIMEOUT,
        )
        if dep:
            text += "\n\n## Dependent files (import metadata)\n\n"
            text += "\n\n---\n\n".join(format_result(d, s, st) for d, s, st in dep)
    return text


mcp = FastMCP(
    "codebase-rag",
    instructions=(
        "Universal Domain RAG: search_knowledge across code, docs, RFCs, MIBs, tickets. "
        "Use search_concepts for tagged topics. feed_domain_doc to add markdown or RFC .txt."
    ),
)


@mcp.tool()
async def reconnect() -> str:
    """Reconnect to ChromaDB after connection issues (retries built-in)."""
    logger.info("reconnect requested")
    loop = asyncio.get_running_loop()
    ex = init_mcp_executor()

    def _do() -> Tuple[int, int]:
        new_map = connect_chroma_with_retry()
        _collections_assign(new_map)
        n = len(new_map)
        t = sum(_safe_count_chroma(v) for v in new_map.values())
        return n, t

    n, t = await loop.run_in_executor(ex, _do)
    return f"Reconnected: {n} collections, ~{t} chunks."


@mcp.tool()
async def search_knowledge(
    query: str,
    k: int = TOP_K_DEFAULT,
    search_type: str = "auto",
    domain: str = "",
    repo: str = "",
    include_dependents: bool = False,
) -> str:
    """Semantic search across indexed collections (multi-domain RAG)."""
    logger.info(
        "search_knowledge q=%r k=%s type=%s domain=%s include_dependents=%s",
        query,
        k,
        search_type,
        domain,
        include_dependents,
    )
    try:
        try:
            ki = int(k)
        except (TypeError, ValueError):
            ki = TOP_K_DEFAULT
        k = max(1, min(ki, MAX_K_RESULTS))
        return await _run_search(
            query, k, search_type, domain, repo, include_dependents=include_dependents
        )
    except asyncio.TimeoutError:
        return f"Error: query timed out after {QUERY_TIMEOUT}s"
    except Exception as exc:
        logger.exception("search_knowledge failed")
        return f"Error: {exc}"


@mcp.tool()
async def search_codebase(query: str, k: int = TOP_K_DEFAULT, repo: str = "") -> str:
    """Backward-compatible code search; same as search_knowledge with search_type=code."""
    return await search_knowledge(
        query, k=k, search_type="code", domain="", repo=repo, include_dependents=False
    )


@mcp.tool()
async def search_concepts(concept: str, domain: str = "") -> str:
    """Find chunks tagged with a concept (metadata 'concepts' contains id)."""
    logger.info("search_concepts concept=%r domain=%s", concept, domain)
    cmap = _collections_snapshot()
    if not cmap:
        return "Error: no collections."
    concept = concept.strip()
    if not concept:
        return "Error: empty concept."
    safe_concept = concept.replace("|", "")
    if not safe_concept:
        return "Error: invalid concept."
    needle = f"|{safe_concept}|"

    def sync_search() -> str:
        by_st: Dict[str, List[str]] = {}
        for cname, vs in cmap.items():
            if domain and domain.lower() not in cname.lower():
                continue
            try:
                col = vs._collection  # type: ignore[attr-defined]
                res = col.get(
                    where={"concepts": {"$contains": needle}},
                    limit=80,
                    include=["documents", "metadatas", "ids"],
                )
                if not (res.get("ids") or []):
                    res = col.get(
                        where={"concepts": {"$contains": safe_concept}},
                        limit=80,
                        include=["documents", "metadatas", "ids"],
                    )
                ids_list = res.get("ids") or []
                docs = res.get("documents") or []
                metas = res.get("metadatas") or []
                pairs: List[Tuple[str, Dict[str, Any]]] = []
                for i, did in enumerate(ids_list):
                    text = docs[i] if i < len(docs) else ""
                    meta = metas[i] if i < len(metas) else {}
                    pairs.append((text, meta or {}))
            except Exception as exc:
                logger.warning("concept query failed on %s: %s", cname, exc)
                continue
            if not pairs:
                continue
            for text, meta in pairs:
                st = _infer_source_type(meta)
                fake = type("D", (), {"page_content": text, "metadata": meta})()
                formatted = format_result(fake, None, st)
                by_st.setdefault(st, []).append(f"*({cname})* {formatted}")
        if not by_st:
            return f"No chunks tagged with concept '{concept}'."
        parts = []
        for st in sorted(by_st.keys()):
            parts.append(f"## source_type: **{st}**")
            parts.extend(by_st[st])
        return "\n\n".join(parts)

    loop = asyncio.get_running_loop()
    ex = init_mcp_executor()
    try:
        return await asyncio.wait_for(
            loop.run_in_executor(ex, sync_search),
            timeout=QUERY_TIMEOUT,
        )
    except asyncio.TimeoutError:
        return f"Error: search_concepts timed out after {QUERY_TIMEOUT}s"


@mcp.tool()
async def feed_domain_doc(
    filepath: str, domain: str = "nms", source_type: str = "auto"
) -> str:
    """Ingest a markdown or RFC .txt file into the appropriate collection.

    RFC .txt files named like rfc<number>.txt are auto-detected and chunked with
    the RFC pipeline (section-aware, diagram-safe) into the ``rfc`` collection.
    Use source_type ``rfc`` or ``domain_doc`` to force chunking mode.

    Returns chunk/section/concept counts as JSON.
    """
    path = os.path.abspath(filepath)
    if not os.path.isfile(path):
        return f"Error: file not found: {path}"
    logger.info(
        "feed_domain_doc path=%s domain=%s source_type=%s", path, domain, source_type
    )
    global _feed_semaphore
    if _feed_semaphore is None:
        _feed_semaphore = asyncio.Semaphore(MCP_MAX_CONCURRENT_FEEDS)
    if not ollama_is_healthy():
        return (
            "Error: Ollama is not healthy (heartbeat). "
            "Check mcp_server.log and that Ollama is running on :11434."
        )
    try:
        async with _feed_semaphore:
            loop = asyncio.get_running_loop()
            ex = init_mcp_executor()

            def _run() -> Dict[str, Any]:
                return feed_domain_document(
                    filepath=path,
                    domain=domain,
                    db_path=DB_PATH,
                    embed_model=EMBEDDING_MODEL,
                    source_type=source_type,
                    chroma_client=_shared_chroma_client,
                    embedder=_shared_embedder,
                    use_embed_lock=False,
                    embed_batch_timeout=EMBED_BATCH_TIMEOUT,
                )

            stats = await asyncio.wait_for(
                loop.run_in_executor(ex, _run),
                timeout=float(QUERY_TIMEOUT),
            )
        return json.dumps(stats, indent=2)
    except asyncio.TimeoutError:
        return f"Error: feed_domain_doc timed out after {QUERY_TIMEOUT}s"
    except Exception as exc:
        logger.exception("feed_domain_doc")
        return f"Error: {exc}"


def _sample_source_type_counts(vs: Chroma, cap: int = 1500) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    try:
        n = _safe_count_chroma(vs)
        batch = vs._collection.get(  # type: ignore[attr-defined]
            limit=min(cap, max(1, n)), include=["metadatas"]
        )
        for m in batch.get("metadatas") or []:
            if not m:
                continue
            st = _infer_source_type(m)
            counts[st] = counts.get(st, 0) + 1
    except Exception:
        pass
    return counts


@mcp.tool()
async def list_repositories() -> str:
    """Collection-aware manifest, chunk counts, and sampled RFC/ticket/MIB-style breakdowns."""
    logger.info("list_repositories")
    manifest_path = os.path.join(DB_PATH, "repos_manifest.json")
    lines: List[str] = ["**Indexed knowledge (by collection)**\n"]
    try:
        if os.path.isfile(manifest_path):
            with open(manifest_path, encoding="utf-8") as fh:
                man = json.load(fh)
            bc = man.get("by_collection") or {}
            if isinstance(bc, dict):
                for cname, data in sorted(bc.items()):
                    if isinstance(data, dict):
                        lines.append(f"- **{cname}**: repos/files — {data}")
                    else:
                        lines.append(f"- **{cname}**: {data}")
    except Exception as exc:
        lines.append(f"(manifest read error: {exc})")
    cmap = _collections_snapshot()
    if cmap:
        lines.append("\n**Live chunk counts (sampled source_type mix per collection):**")
        for name in sorted(cmap.keys()):
            vs = cmap[name]
            cnt = _safe_count_chroma(vs)
            stc = _sample_source_type_counts(vs)
            rfc_n = stc.get("rfc", 0)
            tix = stc.get("rally", 0) + stc.get("customer", 0)
            mib_n = stc.get("mib", 0)
            extra = f" — sample: RFC-like ~{rfc_n}, tickets ~{tix}, MIB-like ~{mib_n} (cap 1500 metadatas)"
            lines.append(f"- **{name}**: {cnt:,} chunks{extra}")
    else:
        lines.append("\nNo Chroma collections connected.")
    return "\n".join(lines)


@mcp.tool()
async def get_db_stats() -> str:
    """Per-collection stats and quick concept/content_type sampling."""
    logger.info("get_db_stats")
    cmap = _collections_snapshot()
    if not cmap:
        return "Error: ChromaDB not connected."
    lines = ["**VectorDB (per collection)**\n", f"- DB path: {os.path.abspath(DB_PATH)}"]
    lines.append(f"- Embedding model: {EMBEDDING_MODEL}")
    concept_hits: Dict[str, int] = {}
    ctype_hits: Dict[str, int] = {}
    for name in sorted(cmap.keys()):
        vs = cmap[name]
        n = _safe_count_chroma(vs)
        lines.append(f"- **{name}**: {n:,} chunks")
        try:
            batch = vs._collection.get(limit=min(500, max(1, n)), include=["metadatas"])  # type: ignore[attr-defined]
            for m in batch.get("metadatas") or []:
                if not m:
                    continue
                cs = str(m.get("concepts", ""))
                for p in iter_concept_ids(cs):
                    concept_hits[p] = concept_hits.get(p, 0) + 1
                ct = str(m.get("content_type", "general"))
                ctype_hits[ct] = ctype_hits.get(ct, 0) + 1
        except Exception:
            pass
    if concept_hits:
        top = sorted(concept_hits.items(), key=lambda x: -x[1])[:20]
        lines.append("\n**Sample concept coverage:** " + ", ".join(f"{k}({v})" for k, v in top))
    if ctype_hits:
        lines.append("**Content types (sample):** " + ", ".join(f"{k}={v}" for k, v in sorted(ctype_hits.items())))
    return "\n".join(lines)


if __name__ == "__main__":
    if hasattr(signal, "SIGINT"):
        signal.signal(signal.SIGINT, _signal_handler)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, _signal_handler)
    startup()
    try:
        mcp.run(transport="stdio")
    finally:
        shutdown()
