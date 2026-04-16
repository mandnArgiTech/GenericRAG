#!/usr/bin/env python3
"""
query.py — Terminal AI agent: hybrid RAG search (BM25 + dense + RRF), optional LLM chat,
stateful REPL, and rich terminal output. Standalone; does not import mcp_server.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import sys
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import readline
except ImportError:  # pragma: no cover
    readline = None  # type: ignore[assignment]

import chromadb
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings

from hybrid_search import (
    HYBRID_AVAILABLE,
    get_bm25_index,
    reciprocal_rank_fusion,
    search_bm25_ranked_ids,
    stable_doc_id,
)

try:
    import ollama as _ollama_mod

    OLLAMA_LIB_AVAILABLE = True
except ImportError:  # pragma: no cover
    _ollama_mod = None  # type: ignore[assignment]
    OLLAMA_LIB_AVAILABLE = False

try:
    from rich.console import Console
    from rich.live import Live
    from rich.markdown import Markdown

    RICH_AVAILABLE = True
except ImportError:  # pragma: no cover
    Console = None  # type: ignore[misc, assignment]
    Live = None  # type: ignore[misc, assignment]
    Markdown = None  # type: ignore[misc, assignment]
    RICH_AVAILABLE = False

# ---------------------------------------------------------------------------
# Constants (aligned with mcp_server.py)
# ---------------------------------------------------------------------------

DIM_TO_MODEL = {
    1024: "mxbai-embed-large",
    768: "nomic-embed-text",
}

TOP_K_DEFAULT = int(os.environ.get("TOP_K", "5"))
MAX_K = max(1, int(os.environ.get("MCP_MAX_K", "25")))
RESULT_CHUNK_MAX_CHARS = max(512, int(os.environ.get("MCP_RESULT_CHUNK_MAX_CHARS", "4096")))
DEFAULT_TIMEOUT = int(os.environ.get("QUERY_CLI_TIMEOUT", "120"))
HISTORY_FILE = Path.home() / ".rag_query_history"

HYBRID_SEARCH = os.environ.get("HYBRID_SEARCH", "1").strip().lower() not in ("0", "false", "no")
RRF_K = float(os.environ.get("RRF_K", "60"))
RAG_CONTEXT_MAX_CHARS = max(4096, int(os.environ.get("RAG_CONTEXT_MAX_CHARS", "32000")))

DEFAULT_SYSTEM_PROMPT = (
    "You are a Senior Engineering AI assistant. Answer the user's question using strictly "
    "the provided context. If the context does not contain enough information, say so "
    "clearly. Cite sources by file path or document name when possible. Be concise."
)

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

EXIT_OK = 0
EXIT_NO_RESULTS = 1
EXIT_ARG = 2
EXIT_INFRA = 3

log = logging.getLogger("query")


def _resolve_db_abs(db_path: str, cmap: Dict[str, Chroma]) -> str:
    if db_path.strip():
        return os.path.abspath(db_path)
    for vs in cmap.values():
        pd = getattr(vs, "_persist_directory", None) or getattr(vs, "persist_directory", None)
        if pd:
            return os.path.abspath(str(pd))
    return ""


def _hybrid_candidate_cap(k: int, env_var: str) -> int:
    raw = os.environ.get(env_var, "").strip()
    if raw.isdigit():
        return max(1, int(raw))
    return max(40, k * 4)


def _make_console(*, no_color: bool, file=None) -> Any:
    if not RICH_AVAILABLE or Console is None:
        return None
    return Console(no_color=no_color, file=file or sys.stdout)


def _print_rich(console: Any, text: str, *, use_markdown: bool = True) -> None:
    if console and RICH_AVAILABLE and Markdown is not None and use_markdown:
        console.print(Markdown(text))
    else:
        print(text)


def _status_spinner(console: Any, message: str) -> Any:
    if console and RICH_AVAILABLE:
        return console.status(message, spinner="dots")

    class _NoOp:
        def __enter__(self):
            return self

        def __exit__(self, *_a):
            return False

    return _NoOp()


def _stream_chunk_text(chunk: Any) -> str:
    if chunk is None:
        return ""
    msg = getattr(chunk, "message", None)
    if msg is None and isinstance(chunk, dict):
        msg = chunk.get("message")
    if msg is None:
        return ""
    if isinstance(msg, dict):
        return str(msg.get("content") or "")
    return str(getattr(msg, "content", None) or "")


def embedding_model_from_db_path(db_path: str) -> str:
    """Resolve embedding model from on-disk DB config / Chroma dims only (no EMBEDDING_MODEL env).

    Use for ingestion paths that must match an existing VectorDB; ``detect_embedding_model`` still
    honors the env override first for query-time behavior.
    """
    config_path = os.path.join(db_path, "ingestion_config.json")
    if os.path.exists(config_path):
        try:
            with open(config_path, encoding="utf-8") as fh:
                model = json.load(fh).get("embedding_model", "")
            if model:
                return str(model).strip()
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


def detect_embedding_model(db_path: str) -> str:
    env_val = os.environ.get("EMBEDDING_MODEL", "").strip()
    if env_val:
        return env_val
    return embedding_model_from_db_path(db_path)


def check_ollama(timeout: float = 3.0) -> bool:
    try:
        urllib.request.urlopen("http://127.0.0.1:11434/api/tags", timeout=timeout)
        return True
    except Exception:
        return False


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
        logging.getLogger("query").error("discover_collections: %s", exc)
    return out


def connect_chroma_with_retry(
    db_path: str,
    model: str,
) -> Tuple[chromadb.PersistentClient, OllamaEmbeddings, Dict[str, Chroma]]:
    import time

    log = logging.getLogger("query")
    last_exc: Optional[Exception] = None
    for attempt in range(3):
        try:
            client = chromadb.PersistentClient(path=db_path)
            client.list_collections()
            embedder = OllamaEmbeddings(model=model)
            cmap = discover_collections(db_path, embedder, client)
            return client, embedder, cmap
        except Exception as exc:
            last_exc = exc
            log.warning("Chroma connect attempt %d failed: %s", attempt + 1, exc)
            time.sleep(2)
    raise RuntimeError(f"ChromaDB connection failed: {last_exc}")


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
    return text[: mx - 50] + "\n\n... (truncated for response size) ..."


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


@dataclass
class SearchHit:
    content: str
    score: Optional[float]
    source_type: str
    metadata: Dict[str, Any]
    collection: Optional[str] = None


def _sync_multi_search(
    query: str,
    k: int,
    search_type: str,
    domain: str,
    repo_filter: str,
    cmap: Dict[str, Chroma],
    db_path: str = "",
) -> List[SearchHit]:
    if not cmap:
        raise RuntimeError("ChromaDB has no collections.")
    targets = _select_collection_names(cmap, search_type, domain)
    per = max(1, k // max(1, len(targets)) if targets else k)
    use_hybrid = HYBRID_SEARCH and HYBRID_AVAILABLE and bool(_resolve_db_abs(db_path, cmap))
    if HYBRID_SEARCH and not HYBRID_AVAILABLE:
        log.warning(
            "HYBRID_SEARCH is on but rank-bm25 is not installed; using dense-only. pip install rank-bm25"
        )

    def _hit(doc: Any, score: Optional[float], st: str, cname: str) -> SearchHit:
        return SearchHit(
            content=doc.page_content or "",
            score=float(score) if score is not None else None,
            source_type=st,
            metadata=dict(doc.metadata or {}),
            collection=cname,
        )

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
                log.warning("Skipping collection %s: %s", name, exc)
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
        return [_hit(doc, score, st, cn) for doc, score, st, cn in merged[:k]]

    db_abs = _resolve_db_abs(db_path, cmap)
    dense_cap = _hybrid_candidate_cap(k, "HYBRID_DENSE_CANDIDATES")
    bm25_cap = _hybrid_candidate_cap(k, "HYBRID_BM25_CANDIDATES")
    fused: List[Tuple[Any, Optional[float], str, str, float]] = []

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
            log.warning("Skipping collection %s: %s", name, exc)
            continue

        dense_ids: List[str] = []
        dense_map: Dict[str, Tuple[Any, float]] = {}
        for doc, score in pairs:
            sid = stable_doc_id(name, doc.metadata or {}, doc.page_content)
            dense_ids.append(sid)
            dense_map[sid] = (doc, score)

        bm25_ids: List[str] = []
        col = getattr(vs, "_collection", None)
        if col is not None and db_abs:
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
            fused.append((doc, None, st, name, rrf_scores[sid]))

    fused.sort(key=lambda x: x[4], reverse=True)
    return [_hit(doc, score, st, cn) for doc, score, st, cn, _ in fused[:k]]


def _concept_parts(concepts_field: str) -> List[str]:
    s = (concepts_field or "").strip()
    if not s:
        return []
    if s.startswith("|"):
        return [x.strip() for x in s.strip("|").split("|") if x.strip()]
    return [x.strip() for x in s.split(",") if x.strip()]


def concept_search_hits(concept: str, domain: str, cmap: Dict[str, Chroma]) -> List[SearchHit]:
    concept = concept.strip()
    if not concept:
        return []
    safe_concept = concept.replace("|", "")
    if not safe_concept:
        return []
    needle = f"|{safe_concept}|"
    hits: List[SearchHit] = []
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
            for i, _did in enumerate(ids_list):
                text = docs[i] if i < len(docs) else ""
                meta = metas[i] if i < len(metas) else {}
                st = _infer_source_type(meta or {})
                hits.append(
                    SearchHit(
                        content=text or "",
                        score=None,
                        source_type=st,
                        metadata=dict(meta or {}),
                        collection=cname,
                    )
                )
        except Exception as exc:
            logging.getLogger("query").warning("concept query failed on %s: %s", cname, exc)
    return hits


def _safe_count(coll: Any) -> int:
    try:
        return int(coll.count())
    except Exception:
        return 0


def run_status(db_path: str) -> str:
    from collections import Counter

    lines: List[str] = []
    lines.append("=" * 64)
    lines.append("            DOMAIN RAG — KNOWLEDGE BASE STATUS")
    lines.append("=" * 64)
    client = chromadb.PersistentClient(path=db_path)
    cols = client.list_collections()
    concept_counts: Counter[str] = Counter()
    rows = []
    for c in sorted(cols, key=lambda x: x.name):
        coll = client.get_collection(c.name)
        n = _safe_count(coll)
        if n == 0:
            rows.append((c.name, 0, 0, ""))
            continue
        sample = coll.get(include=["metadatas"], limit=min(n, 8000))
        metas = sample.get("metadatas") or []
        sources = {str(m.get("source", "")) for m in metas if m}
        for m in metas:
            if not m:
                continue
            cs = m.get("concepts", "")
            if cs:
                for part in _concept_parts(str(cs)):
                    concept_counts[part] += 1
        dates = [str(m.get("ingestion_date", "")) for m in metas if m and m.get("ingestion_date")]
        last_ing = max(dates) if dates else ""
        rows.append((c.name, n, len(sources), last_ing))
    hdr = f"{'Collection':<22} {'Chunks':>8} {'Sources':>8} {'Last Ingested':<22}"
    lines.append(hdr)
    lines.append("-" * 64)
    for name, n, sc, li in rows:
        lines.append(f"{name:<22} {n:>8,} {sc:>8} {li:<22}")
    lines.append("=" * 64)
    top = concept_counts.most_common(15)
    if top:
        lines.append("Top concepts: " + ", ".join(f"{k}({v})" for k, v in top))
    return "\n".join(lines)


def run_with_timeout(seconds: int, fn, *args, **kwargs):
    if seconds <= 0:
        return fn(*args, **kwargs)
    if not hasattr(signal, "SIGALRM"):
        return fn(*args, **kwargs)

    def handler(_signum, _frame):
        raise TimeoutError(f"Query timed out after {seconds}s")

    old = signal.signal(signal.SIGALRM, handler)
    try:
        signal.alarm(max(1, int(seconds)))
        return fn(*args, **kwargs)
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)


def format_markdown(hits: List[SearchHit], query: str) -> str:
    if not hits:
        return "No matching chunks found."
    fake_docs = []
    for h in hits:
        doc = type("D", (), {"page_content": h.content, "metadata": h.metadata})()
        fake_docs.append(format_result(doc, h.score, h.source_type))
    return "\n\n---\n\n".join(fake_docs)


def format_concept_markdown(hits: List[SearchHit], concept: str) -> str:
    if not hits:
        return f"No chunks tagged with concept '{concept}'."
    by_st: Dict[str, List[str]] = {}
    for h in hits:
        doc = type("D", (), {"page_content": h.content, "metadata": h.metadata})()
        formatted = format_result(doc, None, h.source_type)
        cname = h.collection or ""
        by_st.setdefault(h.source_type, []).append(f"*({cname})* {formatted}")
    parts = []
    for st in sorted(by_st.keys()):
        parts.append(f"## source_type: **{st}**")
        parts.extend(by_st[st])
    return "\n\n".join(parts)


def format_json_output(
    query: str, hits: List[SearchHit], mode: str, answer: str = ""
) -> str:
    payload: Dict[str, Any] = {
        "query": query,
        "mode": mode,
        "results": [asdict(h) for h in hits],
    }
    if answer:
        payload["answer"] = answer
    return json.dumps(payload, indent=2)


def format_plain(hits: List[SearchHit]) -> str:
    if not hits:
        return ""
    blocks = []
    for h in hits:
        head = f"[{h.source_type}]"
        if h.collection:
            head += f" ({h.collection})"
        if h.score is not None:
            head += f" score={h.score:.4f}"
        src = h.metadata.get("relative_path") or h.metadata.get("source", "")
        blocks.append(f"{head}\n{src}\n{h.content[:RESULT_CHUNK_MAX_CHARS]}")
    return "\n\n---\n\n".join(blocks)


def _build_context_blocks(hits: List[SearchHit]) -> str:
    blocks: List[str] = []
    total = 0
    for i, h in enumerate(hits, 1):
        src = h.metadata.get("relative_path") or h.metadata.get("source", "?")
        block = f"[Source {i}: {src} ({h.source_type})]\n{h.content}"
        if total + len(block) > RAG_CONTEXT_MAX_CHARS:
            blocks.append(f"\n[... {len(hits) - i + 1} more chunks omitted due to context limit]")
            break
        blocks.append(block)
        total += len(block)
    return "\n\n---\n\n".join(blocks)


def _collect_llm_answer(
    user_query: str,
    hits: List[SearchHit],
    llm_model: str,
    system_prompt: str,
    history_messages: Optional[List[Dict[str, str]]] = None,
) -> str:
    if not OLLAMA_LIB_AVAILABLE or _ollama_mod is None:
        print("Warning: ollama package not installed; skipping LLM answer.", file=sys.stderr)
        return ""
    context = _build_context_blocks(hits)
    messages: List[Dict[str, str]] = [{"role": "system", "content": system_prompt}]
    if history_messages:
        messages.extend(history_messages)
    messages.append(
        {
            "role": "user",
            "content": f"Context:\n\n{context}\n\n---\n\nQuestion: {user_query}",
        }
    )
    try:
        resp = _ollama_mod.chat(model=llm_model, messages=messages, stream=False)
        msg = getattr(resp, "message", None) or (resp.get("message") if isinstance(resp, dict) else None)
        if isinstance(msg, dict):
            return str(msg.get("content") or "")
        return str(getattr(msg, "content", None) or "")
    except Exception as exc:
        err = str(exc).lower()
        if "not found" in err or "pull" in err:
            print(
                f"Error: model {llm_model!r} not found. Run: ollama pull {llm_model}",
                file=sys.stderr,
            )
        else:
            print(f"Error: LLM call failed: {exc}", file=sys.stderr)
        return ""


def _stream_llm_answer(
    user_query: str,
    hits: List[SearchHit],
    llm_model: str,
    system_prompt: str,
    console: Any,
    history_messages: Optional[List[Dict[str, str]]] = None,
) -> str:
    if not OLLAMA_LIB_AVAILABLE or _ollama_mod is None:
        print("Warning: ollama package not installed; skipping LLM answer.", file=sys.stderr)
        return ""
    context = _build_context_blocks(hits)
    messages: List[Dict[str, str]] = [{"role": "system", "content": system_prompt}]
    if history_messages:
        messages.extend(history_messages)
    messages.append(
        {
            "role": "user",
            "content": f"Context:\n\n{context}\n\n---\n\nQuestion: {user_query}",
        }
    )
    try:
        stream = _ollama_mod.chat(model=llm_model, messages=messages, stream=True)
    except Exception as exc:
        err = str(exc).lower()
        if "not found" in err or "pull" in err:
            print(
                f"Error: model {llm_model!r} not found. Run: ollama pull {llm_model}",
                file=sys.stderr,
            )
        else:
            print(f"Error: LLM call failed: {exc}", file=sys.stderr)
        return ""

    parts: List[str] = []
    try:
        if console and RICH_AVAILABLE and Live is not None and Markdown is not None:
            with Live(Markdown(""), console=console, refresh_per_second=12, transient=False) as live:
                for chunk in stream:
                    tok = _stream_chunk_text(chunk)
                    if tok:
                        parts.append(tok)
                        live.update(Markdown("".join(parts)))
        else:
            for chunk in stream:
                tok = _stream_chunk_text(chunk)
                if tok:
                    parts.append(tok)
                    print(tok, end="", flush=True)
            if parts:
                print()
    except KeyboardInterrupt:
        print("\n(generation interrupted)", file=sys.stderr)
    return "".join(parts)


class ConversationMemory:
    def __init__(self, max_turns: int = 5) -> None:
        self.max_turns = max(1, max_turns)
        self.turns: List[Dict[str, str]] = []

    def add_turn(self, raw_query: str, reformulated: str, answer_summary: str) -> None:
        self.turns.append(
            {
                "query": raw_query,
                "reformulated": reformulated,
                "answer_summary": answer_summary[:500],
            }
        )
        if len(self.turns) > self.max_turns:
            self.turns = self.turns[-self.max_turns :]

    def clear(self) -> None:
        self.turns.clear()

    def is_empty(self) -> bool:
        return not self.turns

    def recent_context_text(self, n: int = 2) -> str:
        if not self.turns:
            return ""
        lines: List[str] = []
        for t in self.turns[-n:]:
            q = t["reformulated"] or t["query"]
            lines.append(f"Q: {q}")
            if t.get("answer_summary"):
                lines.append(f"A: {t['answer_summary']}")
        return "\n".join(lines)

    def show(self) -> str:
        if not self.turns:
            return "(no conversation history)"
        lines: List[str] = []
        for i, t in enumerate(self.turns, 1):
            q = t["query"]
            r = t["reformulated"]
            extra = f"  → search: {r}" if r and r != q else ""
            lines.append(f"  {i}. {q}{extra}")
        return "\n".join(lines)

    def history_messages_for_llm(self) -> List[Dict[str, str]]:
        out: List[Dict[str, str]] = []
        for t in self.turns[-3:]:
            out.append({"role": "user", "content": t["reformulated"] or t["query"]})
            if t.get("answer_summary"):
                out.append({"role": "assistant", "content": t["answer_summary"]})
        return out


def reformulate_query(raw_query: str, memory: ConversationMemory, llm_model: str) -> str:
    if memory.is_empty():
        return raw_query
    if len(raw_query.split()) > 8:
        return raw_query
    if not OLLAMA_LIB_AVAILABLE or _ollama_mod is None:
        return raw_query
    ctx = memory.recent_context_text(2)
    prompt = (
        "Rewrite this follow-up into ONE standalone search query (under 20 words) "
        "that someone could type without prior context.\n\n"
        f"Prior turns:\n{ctx}\n\nFollow-up: {raw_query}\n\nStandalone query:"
    )
    try:
        resp = _ollama_mod.chat(
            model=llm_model,
            messages=[{"role": "user", "content": prompt}],
            stream=False,
            options={"num_predict": 80},
        )
        msg = getattr(resp, "message", None) or (
            resp.get("message") if isinstance(resp, dict) else None
        )
        text = ""
        if isinstance(msg, dict):
            text = str(msg.get("content") or "").strip()
        else:
            text = str(getattr(msg, "content", None) or "").strip()
        text = text.split("\n")[0].strip().strip('"').strip("'")
        if 3 < len(text) < 200:
            return text
    except Exception:
        pass
    return raw_query


class SessionState:
    def __init__(self, ns: argparse.Namespace) -> None:
        self.domain = ns.domain or ""
        self.repo = ns.repo or ""
        self.top_k = max(1, min(int(ns.top_k), MAX_K))
        self.search_type = ns.search_type
        self.out_format = ns.format
        self.mode = ns.mode  # semantic | concept | codebase
        self.timeout = int(ns.timeout)
        self.chat = bool(getattr(ns, "chat", False))
        env_llm = (os.environ.get("RAG_LLM_MODEL", "") or "").strip()
        self.llm_model = (getattr(ns, "llm_model", None) or env_llm or "llama3").strip()
        sp = (getattr(ns, "system_prompt", None) or "").strip()
        self.system_prompt = sp if sp else DEFAULT_SYSTEM_PROMPT
        self.history_depth = max(1, int(getattr(ns, "history_depth", 5)))

    def apply_set(self, key: str, value: str) -> str:
        k = key.lower().strip()
        v = value.strip()
        if k == "domain":
            self.domain = v
        elif k == "k" or k == "top_k":
            try:
                self.top_k = max(1, min(int(v), MAX_K))
            except ValueError:
                return "k must be an integer"
        elif k == "type" or k == "search-type":
            allowed = {"auto", "code", "domain", "troubleshoot", "reference"}
            if v not in allowed:
                return f"type must be one of: {', '.join(sorted(allowed))}"
            self.search_type = v
        elif k == "repo":
            self.repo = v
        elif k == "format":
            if v not in ("markdown", "json", "plain"):
                return "format must be markdown, json, or plain"
            self.out_format = v
        elif k == "mode":
            if v not in ("semantic", "concept", "codebase"):
                return "mode must be semantic, concept, or codebase (use /status for DB status)"
            self.mode = v
        elif k == "timeout":
            try:
                self.timeout = max(1, int(v))
            except ValueError:
                return "timeout must be an integer"
        elif k in ("history_depth", "history-depth"):
            try:
                self.history_depth = max(1, int(v))
            except ValueError:
                return "history_depth must be an integer"
        elif k == "chat":
            if v.lower() in ("1", "true", "yes", "on"):
                self.chat = True
            elif v.lower() in ("0", "false", "no", "off"):
                self.chat = False
            else:
                return "chat must be on or off"
        else:
            return f"Unknown key: {key}"
        return f"OK: {k} = {v!r}"

    def show(self) -> str:
        return (
            f"domain={self.domain!r} repo={self.repo!r} k={self.top_k} "
            f"search_type={self.search_type!r} format={self.out_format!r} "
            f"mode={self.mode!r} timeout={self.timeout} chat={self.chat} "
            f"llm_model={self.llm_model!r} history_depth={self.history_depth}"
        )


def _setup_readline() -> None:
    if readline is None:
        return
    try:
        if HISTORY_FILE.exists():
            readline.read_history_file(str(HISTORY_FILE))
    except OSError:
        pass
    readline.set_history_length(500)


def _save_history() -> None:
    if readline is None:
        return
    try:
        readline.write_history_file(str(HISTORY_FILE))
    except OSError:
        pass


def repl_loop(
    cmap: Dict[str, Chroma],
    db_path: str,
    ns: argparse.Namespace,
    chroma_client: chromadb.PersistentClient,
    embedder: OllamaEmbeddings,
) -> int:
    st = SessionState(ns)
    memory = ConversationMemory(st.history_depth)
    log = logging.getLogger("query")
    _setup_readline()
    print("Interactive RAG query. Type /help for commands, Ctrl+D to exit.")
    print(st.show())
    while True:
        try:
            line = input("rag> ").strip()
        except EOFError:
            print()
            _save_history()
            return EXIT_OK
        except KeyboardInterrupt:
            print("\n(Interrupted — empty line or /quit to exit)")
            continue
        if not line:
            continue
        if line in ("/quit", "/exit", "exit", "quit"):
            _save_history()
            return EXIT_OK
        if line == "/help":
            print(
                "Commands: /set, /show, /help, /status, /history, /clear, /quit\n"
                "  /set <key> <value>\n"
                "Keys: domain, k, type, repo, format, mode, timeout, chat, history_depth\n"
                "Anything else is treated as a search query."
            )
            continue
        if line == "/show":
            print(st.show())
            continue
        if line == "/status":
            print(run_status(db_path))
            continue
        if line == "/history":
            print(memory.show())
            continue
        if line == "/clear":
            memory.clear()
            print("Conversation memory cleared.")
            continue
        if line.startswith("/set "):
            rest = line[5:].strip()
            parts = rest.split(None, 1)
            if len(parts) < 2:
                print("Usage: /set <key> <value>")
                continue
            msg = st.apply_set(parts[0], parts[1])
            print(msg)
            key0 = parts[0].lower().strip()
            if msg.startswith("OK") and key0 in ("history_depth", "history-depth"):
                memory.max_turns = st.history_depth
            continue

        raw_query = line
        search_query = raw_query
        if st.chat and st.mode != "concept" and not memory.is_empty():
            search_query = reformulate_query(raw_query, memory, st.llm_model)
            if search_query != raw_query:
                print(f'(searching for: "{search_query}")')

        rich_ui = RICH_AVAILABLE and st.out_format == "markdown"
        console = _make_console(no_color=bool(ns.no_color)) if rich_ui else None

        effective_type = "code" if st.mode == "codebase" else st.search_type
        try:
            with _status_spinner(console, "Searching..."):
                if st.mode == "concept":
                    hits = run_with_timeout(
                        st.timeout,
                        concept_search_hits,
                        search_query,
                        st.domain,
                        cmap,
                    )
                else:
                    hits = run_with_timeout(
                        st.timeout,
                        _sync_multi_search,
                        search_query,
                        st.top_k,
                        effective_type,
                        st.domain,
                        st.repo,
                        cmap,
                        db_path,
                    )
        except (TimeoutError, ValueError) as exc:
            print(f"Error: {exc}", file=sys.stderr)
            continue
        except Exception as exc:
            log.exception("query failed")
            print(f"Error: {exc}", file=sys.stderr)
            continue
        if not hits:
            if st.mode == "concept":
                print(f"No chunks tagged with concept '{search_query}'.")
            else:
                print("No matching chunks found.")
            continue

        hist_msgs = memory.history_messages_for_llm() if st.chat else None

        if st.chat:
            if st.mode == "concept":
                if st.out_format == "json":
                    ans = _collect_llm_answer(
                        raw_query,
                        hits,
                        st.llm_model,
                        st.system_prompt,
                        hist_msgs,
                    )
                    print(format_json_output(raw_query, hits, "concept", answer=ans))
                elif st.out_format == "markdown":
                    ans = _stream_llm_answer(
                        raw_query,
                        hits,
                        st.llm_model,
                        st.system_prompt,
                        console,
                        hist_msgs,
                    )
                else:
                    ans = _stream_llm_answer(
                        raw_query,
                        hits,
                        st.llm_model,
                        st.system_prompt,
                        None,
                        hist_msgs,
                    )
                memory.add_turn(raw_query, search_query, ans or "(no answer)")
                continue

            if st.out_format == "json":
                ans = _collect_llm_answer(
                    raw_query,
                    hits,
                    st.llm_model,
                    st.system_prompt,
                    hist_msgs,
                )
                print(format_json_output(raw_query, hits, st.mode, answer=ans))
            elif st.out_format == "markdown":
                ans = _stream_llm_answer(
                    raw_query,
                    hits,
                    st.llm_model,
                    st.system_prompt,
                    console,
                    hist_msgs,
                )
            else:
                ans = _stream_llm_answer(
                    raw_query,
                    hits,
                    st.llm_model,
                    st.system_prompt,
                    None,
                    hist_msgs,
                )
            memory.add_turn(raw_query, search_query, ans or "(no answer)")
            continue

        if st.mode == "concept":
            if st.out_format == "json":
                print(format_json_output(raw_query, hits, "concept"))
            elif st.out_format == "markdown":
                _print_rich(console, format_concept_markdown(hits, raw_query))
            else:
                print(format_plain(hits))
        else:
            if st.out_format == "json":
                print(format_json_output(raw_query, hits, st.mode))
            elif st.out_format == "markdown":
                _print_rich(console, format_markdown(hits, raw_query))
            else:
                print(format_plain(hits))
    return EXIT_OK


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Query Universal Domain RAG (Chroma + Ollama).")
    p.add_argument("-q", "--query", default="", help="Search text (concept id in concept mode)")
    p.add_argument(
        "-m",
        "--mode",
        choices=("semantic", "concept", "codebase", "status"),
        default="semantic",
        help="Search mode (default: semantic)",
    )
    p.add_argument(
        "-t",
        "--search-type",
        choices=("auto", "code", "domain", "troubleshoot", "reference"),
        default="auto",
        help="Collection routing for semantic/codebase (default: auto)",
    )
    p.add_argument("-d", "--domain", default="", help="Filter collections by name substring")
    p.add_argument("-r", "--repo", default="", help="Filter by repository metadata")
    p.add_argument("-k", "--top-k", type=int, default=TOP_K_DEFAULT, help=f"Max results (1–{MAX_K})")
    p.add_argument(
        "-f",
        "--format",
        dest="format",
        choices=("markdown", "json", "plain"),
        default="markdown",
        help="Output format",
    )
    p.add_argument("-o", "--output", default="", help="Write output to file instead of stdout")
    p.add_argument(
        "-c",
        "--chat",
        action="store_true",
        help="After retrieval, generate an answer with the LLM (Ollama chat)",
    )
    p.add_argument(
        "--llm-model",
        default=(os.environ.get("RAG_LLM_MODEL", "") or "").strip() or "llama3",
        help="Ollama chat model for --chat (default: env RAG_LLM_MODEL or llama3)",
    )
    p.add_argument(
        "--system-prompt",
        default="",
        help="Override default RAG system prompt when using --chat",
    )
    p.add_argument(
        "--history-depth",
        type=int,
        default=5,
        help="REPL: max conversation turns to remember with --chat (default: 5)",
    )
    p.add_argument("--no-color", action="store_true", help="Disable color (rich/ANSI)")
    p.add_argument(
        "--db-path",
        default=os.environ.get("DB_PATH", "").strip()
        or str(Path(__file__).resolve().parent / "Studio-Portable-RAG" / "VectorDB"),
        help="Chroma persist directory",
    )
    p.add_argument("--model", default="", help="Embedding model (default: auto-detect)")
    p.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT, help="Per-query timeout (seconds)")
    p.add_argument("-i", "--interactive", action="store_true", help="Interactive REPL")
    p.add_argument("-v", "--verbose", action="store_true", help="Verbose logging on stderr")
    p.add_argument("--quiet", action="store_true", help="Suppress banners")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    ns = parse_args(argv)
    log = logging.getLogger("query")
    logging.basicConfig(
        level=logging.DEBUG if ns.verbose else logging.WARNING,
        format="%(levelname)s %(message)s",
        stream=sys.stderr,
    )

    db_path = str(Path(ns.db_path).resolve())
    if not Path(db_path).is_dir():
        if not ns.quiet:
            print(f"Error: DB path is not a directory: {db_path}", file=sys.stderr)
        return EXIT_INFRA

    need_query = not ns.interactive and ns.mode != "status"
    if need_query and not (ns.query or "").strip():
        if not ns.quiet:
            print("Error: --query is required unless --interactive or --mode status", file=sys.stderr)
        return EXIT_ARG

    if ns.mode == "status":
        text = run_status(db_path)
        if ns.output:
            Path(ns.output).write_text(text, encoding="utf-8")
        else:
            print(text)
        return EXIT_OK

    if not check_ollama():
        if not ns.quiet:
            print(
                "Error: Ollama not reachable at http://127.0.0.1:11434/api/tags. "
                "Start Ollama or use query.sh.",
                file=sys.stderr,
            )
        return EXIT_INFRA

    model = ns.model.strip() or detect_embedding_model(db_path)
    if not ns.quiet:
        print(f"DB: {db_path}\nModel: {model}", file=sys.stderr)

    try:
        chroma_client, embedder, cmap = connect_chroma_with_retry(db_path, model)
    except Exception as exc:
        if not ns.quiet:
            print(f"Error: {exc}", file=sys.stderr)
        return EXIT_INFRA

    if not cmap:
        if not ns.quiet:
            print("Error: no Chroma collections in DB.", file=sys.stderr)
        return EXIT_INFRA

    if ns.interactive:
        return repl_loop(cmap, db_path, ns, chroma_client, embedder)

    effective_type = "code" if ns.mode == "codebase" else ns.search_type
    try:
        k = max(1, min(int(ns.top_k), MAX_K))
    except (TypeError, ValueError):
        k = TOP_K_DEFAULT

    q = ns.query.strip()
    spin_console = _make_console(no_color=bool(ns.no_color)) if (
        RICH_AVAILABLE and ns.format == "markdown"
    ) else None
    display_console = _make_console(no_color=bool(ns.no_color)) if (
        RICH_AVAILABLE and ns.format == "markdown" and not ns.output
    ) else None

    try:
        with _status_spinner(spin_console, "Searching..."):
            if ns.mode == "concept":
                hits = run_with_timeout(
                    int(ns.timeout),
                    concept_search_hits,
                    q,
                    ns.domain,
                    cmap,
                )
            else:
                hits = run_with_timeout(
                    int(ns.timeout),
                    _sync_multi_search,
                    q,
                    k,
                    effective_type,
                    ns.domain,
                    ns.repo,
                    cmap,
                    db_path,
                )
    except TimeoutError as exc:
        if not ns.quiet:
            print(f"Error: {exc}", file=sys.stderr)
        return EXIT_INFRA
    except Exception as exc:
        log.exception("search failed")
        if not ns.quiet:
            print(f"Error: {exc}", file=sys.stderr)
        return EXIT_INFRA

    if not hits:
        msg = "No matching chunks found."
        if ns.output:
            Path(ns.output).write_text(msg + "\n", encoding="utf-8")
        elif not ns.quiet:
            print(msg)
        return EXIT_NO_RESULTS

    llm_model = (ns.llm_model or "").strip() or "llama3"
    system_prompt = (ns.system_prompt or "").strip() or DEFAULT_SYSTEM_PROMPT

    if ns.chat:
        if ns.mode == "concept":
            mode_label = "concept"
        else:
            mode_label = ns.mode
        if ns.format == "json":
            ans = _collect_llm_answer(q, hits, llm_model, system_prompt, None)
            text = format_json_output(q, hits, mode_label, answer=ans)
            if ns.output:
                Path(ns.output).write_text(text, encoding="utf-8")
            else:
                print(text)
            return EXIT_OK
        if ns.format == "plain":
            if ns.output:
                ans = _collect_llm_answer(q, hits, llm_model, system_prompt, None)
                Path(ns.output).write_text(ans + ("\n" if ans and not ans.endswith("\n") else ""), encoding="utf-8")
            else:
                _stream_llm_answer(q, hits, llm_model, system_prompt, None, None)
            return EXIT_OK
        # markdown
        if ns.output:
            ans = _collect_llm_answer(q, hits, llm_model, system_prompt, None)
            Path(ns.output).write_text(ans + ("\n" if ans and not ans.endswith("\n") else ""), encoding="utf-8")
        else:
            _stream_llm_answer(q, hits, llm_model, system_prompt, display_console, None)
        return EXIT_OK

    if ns.mode == "concept":
        text = (
            format_json_output(q, hits, "concept")
            if ns.format == "json"
            else format_concept_markdown(hits, q)
            if ns.format == "markdown"
            else format_plain(hits)
        )
    else:
        text = (
            format_json_output(q, hits, ns.mode)
            if ns.format == "json"
            else format_markdown(hits, q)
            if ns.format == "markdown"
            else format_plain(hits)
        )

    if ns.output:
        Path(ns.output).write_text(text, encoding="utf-8")
    elif ns.format == "markdown" and display_console:
        _print_rich(display_console, text)
    else:
        print(text)
    return EXIT_OK


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        sys.exit(130)
