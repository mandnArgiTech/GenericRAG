"""
BM25 sidecar index + Reciprocal Rank Fusion for Chroma dense search.

Used by mcp_server._sync_multi_search when HYBRID_SEARCH is enabled.
Requires: pip install rank-bm25

Author: deviprasad
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import pickle
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("hybrid_search")

try:
    from rank_bm25 import BM25Okapi

    HYBRID_AVAILABLE = True
except ImportError:
    BM25Okapi = None  # type: ignore[misc, assignment]
    HYBRID_AVAILABLE = False

STABLE_SEP = "\x1f"
_CHROMA_GET_BATCH = 512


def tokenize(text: str) -> List[str]:
    return (text or "").lower().split()


def stable_doc_id(collection: str, meta: Dict[str, Any], content: str) -> str:
    """Stable key for RRF dedupe: prefer source+chunk_index; else content hash."""
    src = str(meta.get("source") or "")
    idx = str(meta.get("chunk_index") if meta.get("chunk_index") is not None else "")
    if src and idx != "":
        return f"{collection}{STABLE_SEP}{src}{STABLE_SEP}{idx}"
    h = hashlib.sha256((content or "")[:400].encode("utf-8", errors="ignore")).hexdigest()[:20]
    return f"{collection}{STABLE_SEP}__h__{STABLE_SEP}{h}"


def reciprocal_rank_fusion(rank_lists: List[List[str]], k: float = 60.0) -> Dict[str, float]:
    scores: Dict[str, float] = {}
    for ids in rank_lists:
        if not ids:
            continue
        for rank, doc_id in enumerate(ids, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return scores


def bm25_cache_root(db_path: str) -> str:
    override = os.environ.get("BM25_CACHE_DIR", "").strip()
    if override:
        return override
    return os.path.join(db_path, "bm25_cache")


class CachedBM25Index:
    """Lazy BM25 index for one Chroma collection with pickle cache on disk."""

    _locks: Dict[str, threading.Lock] = {}
    _locks_guard = threading.Lock()

    @classmethod
    def _lock_for(cls, key: str) -> threading.Lock:
        with cls._locks_guard:
            if key not in cls._locks:
                cls._locks[key] = threading.Lock()
            return cls._locks[key]

    def __init__(self, collection_name: str, db_path: str) -> None:
        self.collection_name = collection_name
        self.db_path = db_path
        self.cache_dir = bm25_cache_root(db_path)
        self.bm25: Any = None
        self.ordered_ids: List[str] = []
        self.id_to_doc: Dict[str, Tuple[str, Dict[str, Any]]] = {}
        self._loaded_count = -1

    def _paths(self) -> Tuple[str, str]:
        safe = self.collection_name.replace("/", "_")
        base = os.path.join(self.cache_dir, safe)
        return base + ".pkl", base + ".meta.json"

    def ensure_loaded(self, chroma_collection: Any) -> bool:
        if not HYBRID_AVAILABLE:
            return False
        lock = self._lock_for(self.collection_name + "@" + self.db_path)
        with lock:
            try:
                count: int = int(chroma_collection.count())
            except Exception as exc:
                logger.warning("BM25: count failed for %s: %s", self.collection_name, exc)
                return False

            if count <= 0:
                self.bm25 = None
                self.ordered_ids = []
                self.id_to_doc = {}
                self._loaded_count = count
                logger.debug("BM25 skip empty collection %s", self.collection_name)
                return False

            if self.bm25 is not None and self._loaded_count == count:
                return True

            pkl_path, meta_path = self._paths()
            if os.path.isfile(pkl_path) and os.path.isfile(meta_path):
                try:
                    with open(meta_path, encoding="utf-8") as fh:
                        meta = json.load(fh)
                    if int(meta.get("count", -1)) == count and meta.get("name") == self.collection_name:
                        with open(pkl_path, "rb") as fh:
                            data = pickle.load(fh)
                        self.bm25 = data["bm25"]
                        self.ordered_ids = data["ordered_ids"]
                        self.id_to_doc = data["id_to_doc"]
                        self._loaded_count = count
                        logger.info(
                            "BM25 cache hit %s (%d docs)", self.collection_name, count
                        )
                        return True
                except Exception as exc:
                    logger.warning("BM25 cache load failed %s: %s", self.collection_name, exc)

            t0 = time.monotonic()
            os.makedirs(self.cache_dir, exist_ok=True)
            documents: List[str] = []
            metadatas: List[Optional[Dict[str, Any]]] = []
            offset = 0
            while True:
                batch = chroma_collection.get(
                    include=["documents", "metadatas"],
                    limit=_CHROMA_GET_BATCH,
                    offset=offset,
                )
                docs = batch.get("documents") or []
                metas = batch.get("metadatas") or []
                if not docs:
                    break
                documents.extend(docs)
                metadatas.extend(metas)
                offset += len(docs)
                if len(docs) < _CHROMA_GET_BATCH:
                    break

            tokenized_corpus: List[List[str]] = []
            ordered_ids: List[str] = []
            id_to_doc: Dict[str, Tuple[str, Dict[str, Any]]] = {}
            for text, meta in zip(documents, metadatas):
                m = dict(meta or {})
                sid = stable_doc_id(self.collection_name, m, text or "")
                tokenized_corpus.append(tokenize(text or ""))
                ordered_ids.append(sid)
                if sid not in id_to_doc:
                    id_to_doc[sid] = (text or "", m)

            if not tokenized_corpus:
                logger.warning(
                    "BM25: count=%d but no documents loaded for %s; skipping BM25",
                    count,
                    self.collection_name,
                )
                self.bm25 = None
                self.ordered_ids = []
                self.id_to_doc = {}
                self._loaded_count = count
                return False

            self.bm25 = BM25Okapi(tokenized_corpus)
            self.ordered_ids = ordered_ids
            self.id_to_doc = id_to_doc
            self._loaded_count = len(documents)

            try:
                payload = {
                    "bm25": self.bm25,
                    "ordered_ids": ordered_ids,
                    "id_to_doc": id_to_doc,
                }
                with open(pkl_path, "wb") as fh:
                    pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)
                with open(meta_path, "w", encoding="utf-8") as fh:
                    json.dump(
                        {"count": self._loaded_count, "name": self.collection_name},
                        fh,
                    )
            except Exception as exc:
                logger.warning("BM25 cache save failed %s: %s", self.collection_name, exc)

            elapsed = time.monotonic() - t0
            logger.info(
                "BM25 index built %s: %d docs in %.2fs",
                self.collection_name,
                self._loaded_count,
                elapsed,
            )
            return True


def search_bm25_ranked_ids(
    index: CachedBM25Index,
    query: str,
    top_n: int,
    repo_filter: str,
) -> List[str]:
    if not query.strip() or index.bm25 is None:
        return []
    q_tokens = tokenize(query)
    if not q_tokens:
        return []
    scores = index.bm25.get_scores(q_tokens)
    rf = repo_filter.strip()
    if rf:
        candidates: List[int] = []
        for i, sid in enumerate(index.ordered_ids):
            _text, meta = index.id_to_doc.get(sid, ("", {}))
            if str(meta.get("repository") or "") == rf:
                candidates.append(i)
        order = sorted(candidates, key=lambda i: scores[i], reverse=True)
    else:
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    out: List[str] = []
    for i in order:
        out.append(index.ordered_ids[i])
        if len(out) >= top_n:
            break
    return out


_index_singletons: Dict[str, CachedBM25Index] = {}
_singleton_guard = threading.Lock()


def get_bm25_index(db_path: str, collection_name: str) -> CachedBM25Index:
    key = db_path + "\x00" + collection_name
    with _singleton_guard:
        if key not in _index_singletons:
            _index_singletons[key] = CachedBM25Index(collection_name, db_path)
        return _index_singletons[key]
