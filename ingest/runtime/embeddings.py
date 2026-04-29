"""Ollama / HTTP embedding batches and worker helpers.

Author: deviprasad
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import queue
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from langchain_ollama import OllamaEmbeddings

from ingest.core.constants import EMBED_BACKOFF_SEC, MAX_RETRIES
from ingest.core.deps import aiohttp
from ingest.core.state import _embed_lock

logger = logging.getLogger("ingest")

def _embed_serialize_on() -> bool:
    return os.environ.get("EMBED_SERIALIZE", "0").strip().lower() in ("1", "true", "yes")


def _ollama_embed_url() -> str:
    base = os.environ.get("OLLAMA_HOST", "127.0.0.1:11434").strip()
    if base.startswith("http://") or base.startswith("https://"):
        return base.rstrip("/") + "/api/embed"
    return f"http://{base}/api/embed"


def _nvidia_total_vram_mb() -> Optional[int]:
    try:
        proc = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if proc.returncode != 0 or not proc.stdout.strip():
            return None
        lines = [x.strip() for x in proc.stdout.strip().splitlines() if x.strip()]
        if not lines:
            return None
        return int(float(lines[0]))
    except Exception:
        return None


def _host_total_ram_mb() -> Optional[int]:
    """Best-effort system RAM (MiB). Linux: /proc/meminfo; macOS: sysctl."""
    try:
        if sys.platform == "darwin":
            out = subprocess.run(
                ["sysctl", "-n", "hw.memsize"],
                capture_output=True,
                text=True,
                timeout=3,
                check=False,
            )
            if out.returncode != 0 or not out.stdout.strip():
                return None
            return int(int(out.stdout.strip()) // (1024 * 1024))
        p = Path("/proc/meminfo")
        if p.is_file():
            for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
                if line.startswith("MemTotal:"):
                    parts = line.split()
                    if len(parts) >= 2:
                        return int(parts[1]) // 1024  # kB -> MiB
    except Exception:
        return None
    return None


def resolve_embed_ingest_settings() -> Tuple[int, int, int]:
    """Return (batch_size, worker_threads, async_concurrency) with optional VRAM + RAM scaling."""
    cpu = os.cpu_count() or 2
    default_workers = min(4, max(2, cpu))
    batch = int(os.environ.get("EMBED_BATCH_SIZE", "16"))
    workers = int(os.environ.get("EMBED_WORKERS", str(default_workers)))
    conc = int(os.environ.get("EMBED_CONCURRENCY", str(max(2, min(8, workers * 2)))))
    batch_env_set = "EMBED_BATCH_SIZE" in os.environ
    conc_env_set = "EMBED_CONCURRENCY" in os.environ

    vram = _nvidia_total_vram_mb()
    if vram is not None:
        if vram >= 12000:
            batch = max(batch, 24)
            conc = max(conc, 6)
        elif vram >= 8000:
            batch = max(batch, 20)
            conc = max(conc, 4)
        logger.info(
            "Embedding autoscale: VRAM ~%d MiB -> batch=%d concurrency=%d workers=%d",
            vram,
            batch,
            conc,
            workers,
        )

    ram = _host_total_ram_mb()
    if ram is not None:
        if ram < 4096:
            batch = min(batch, 6)
            conc = min(conc, 2)
        elif ram < 8192:
            batch = min(batch, 12)
            conc = min(conc, 4)
        elif ram >= 32768 and not batch_env_set and not conc_env_set:
            batch = max(batch, 20)
            conc = max(conc, 8)
        logger.info(
            "Embedding autoscale: RAM ~%d MiB -> batch=%d concurrency=%d workers=%d",
            ram,
            batch,
            conc,
            workers,
        )

    return max(1, batch), max(1, workers), max(1, conc)


def http_embed_documents_batch(model: str, texts: List[str], timeout: float = 300.0) -> List[List[float]]:
    """Direct Ollama /api/embed (avoids shared LangChain client across threads)."""
    url = _ollama_embed_url()
    payload = json.dumps({"model": model, "input": texts}).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.load(resp)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace") if exc.fp else ""
        raise RuntimeError(f"Ollama embed HTTP {exc.code}: {body}") from exc
    embs = data.get("embeddings")
    if embs is not None:
        return embs
    one = data.get("embedding")
    if one is not None:
        return [one]
    raise RuntimeError(f"Unexpected Ollama embed response: {list(data.keys())}")


def embed_with_retry(embedder: OllamaEmbeddings, batch: List[str]) -> Optional[List[List[float]]]:
    def _try(b: List[str]) -> Optional[List[List[float]]]:
        for _ in range(MAX_RETRIES):
            try:
                if _embed_serialize_on():
                    with _embed_lock:
                        return embedder.embed_documents(b)
                return embedder.embed_documents(b)
            except Exception as exc:
                logger.warning("embed retry: %s", exc)
                time.sleep(EMBED_BACKOFF_SEC)
        if len(b) <= 1:
            return None
        mid = max(1, len(b) // 2)
        a = _try(b[:mid])
        b2 = _try(b[mid:])
        if a is None or b2 is None:
            return None
        return a + b2

    return _try(batch)


def embed_with_retry_http(model: str, batch: List[str]) -> Optional[List[List[float]]]:
    """HTTP embed with same retry/split semantics as embed_with_retry."""

    def _try(b: List[str]) -> Optional[List[List[float]]]:
        for _ in range(MAX_RETRIES):
            try:
                if _embed_serialize_on():
                    with _embed_lock:
                        return http_embed_documents_batch(model, b)
                return http_embed_documents_batch(model, b)
            except Exception as exc:
                logger.warning("embed http retry: %s", exc)
                time.sleep(EMBED_BACKOFF_SEC)
        if len(b) <= 1:
            return None
        mid = max(1, len(b) // 2)
        a = _try(b[:mid])
        b2 = _try(b[mid:])
        if a is None or b2 is None:
            return None
        return a + b2

    return _try(batch)


async def _async_http_embed_batch(
    session: Any, model: str, texts: List[str], timeout: float = 300.0
) -> List[List[float]]:
    if aiohttp is None:
        raise RuntimeError("aiohttp is not installed")
    url = _ollama_embed_url()
    to = aiohttp.ClientTimeout(total=timeout)
    async with session.post(url, json={"model": model, "input": texts}, timeout=to) as resp:
        resp.raise_for_status()
        data = await resp.json()
    embs = data.get("embeddings")
    if embs is not None:
        return embs
    one = data.get("embedding")
    if one is not None:
        return [one]
    raise RuntimeError(f"Unexpected Ollama embed response: {list(data.keys())}")


async def embed_with_retry_http_async(
    session: Any,
    model: str,
    batch: List[str],
    async_lock: Optional[asyncio.Lock],
) -> Optional[List[List[float]]]:
    async def _try(b: List[str]) -> Optional[List[List[float]]]:
        for _ in range(MAX_RETRIES):
            try:
                if async_lock is not None:
                    async with async_lock:
                        return await _async_http_embed_batch(session, model, b)
                return await _async_http_embed_batch(session, model, b)
            except Exception as exc:
                logger.warning("embed async retry: %s", exc)
                await asyncio.sleep(EMBED_BACKOFF_SEC)
        if len(b) <= 1:
            return None
        mid = max(1, len(b) // 2)
        a = await _try(b[:mid])
        b2 = await _try(b[mid:])
        if a is None or b2 is None:
            return None
        return a + b2

    return await _try(batch)


async def run_async_embedding_batches(
    batches: List[List[Tuple[str, str, Dict[str, str]]]],
    embed_model: str,
    concurrency: int,
) -> List[Optional[Tuple[List[str], List[str], List[Dict[str, str]], List[List[float]]]]]:
    """Concurrent aiohttp embedding; returns one result per input batch (order preserved)."""
    if aiohttp is None:
        return [None] * len(batches)
    sem = asyncio.Semaphore(concurrency)
    alock = asyncio.Lock() if _embed_serialize_on() else None

    async with aiohttp.ClientSession() as session:

        async def one(
            batch: List[Tuple[str, str, Dict[str, str]]],
        ) -> Optional[Tuple[List[str], List[str], List[Dict[str, str]], List[List[float]]]]:
            async with sem:
                ids, texts, metas = [], [], []
                for cid, text, meta in batch:
                    ids.append(cid)
                    texts.append(text)
                    metas.append(meta)
                vecs = await embed_with_retry_http_async(session, embed_model, texts, alock)
                if vecs is None:
                    return None
                return (ids, texts, metas, vecs)

        return list(await asyncio.gather(*[one(b) for b in batches]))


def embedding_worker(
    embed_model: str,
    worker_id: int,
    chunk_q: "queue.Queue[Optional[List[Tuple[str, str, Dict[str, str]]]]]",
    result_q: "queue.Queue[Optional[Tuple[List[str], List[str], List[Dict[str, str]], List[List[float]]]]]",
) -> None:
    embedder = OllamaEmbeddings(model=embed_model)
    use_http = os.environ.get("EMBED_HTTP", "1").strip().lower() not in ("0", "false", "no")
    while True:
        item = chunk_q.get()
        if item is None:
            chunk_q.task_done()
            break
        try:
            ids, texts, metas = [], [], []
            for cid, text, meta in item:
                ids.append(cid)
                texts.append(text)
                metas.append(meta)
            if use_http:
                vecs = embed_with_retry_http(embed_model, texts)
            else:
                vecs = embed_with_retry(embedder, texts)
            if vecs is None:
                srcs = [m.get("source", "") for m in metas[:5]]
                logger.error(
                    "embedding failed permanently worker=%d batch=%d sample_sources=%s",
                    worker_id,
                    len(texts),
                    srcs,
                )
                result_q.put(None)
            else:
                result_q.put((ids, texts, metas, vecs))
        except Exception as exc:  # pragma: no cover
            logger.exception("worker error: %s", exc)  # pragma: no cover
            result_q.put(None)  # pragma: no cover
        finally:
            chunk_q.task_done()

