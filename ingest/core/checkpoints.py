"""Chroma embedding dimension checks, checkpoints, manifests, ingestion config.

Author: deviprasad
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict

import chromadb
from langchain_ollama import OllamaEmbeddings

from ingest.core.constants import INGESTION_VERSION

logger = logging.getLogger("ingest")

def migrate_old_checkpoint(db_path: Path, collection_name: str) -> None:
    """One-time migration from legacy flat checkpoint file to per-collection ingest_checkpoint.json."""
    old_path = db_path / "ingestion_checkpoint.json"
    new_path = db_path / "ingest_checkpoint.json"
    if not old_path.exists() or new_path.exists():
        return
    try:
        with open(old_path, encoding="utf-8") as fh:
            old_data = json.load(fh)
        if not isinstance(old_data, dict):
            return
        cp_key = f"{collection_name}::checkpoint"
        new_data = {cp_key: json.dumps(old_data)}
        with open(new_path, "w", encoding="utf-8") as fh:
            json.dump(new_data, fh, indent=2)
        logger.info(
            "Migrated legacy checkpoint %s (%d entries) -> %s",
            old_path.name,
            len(old_data),
            new_path.name,
        )
    except Exception as exc:
        logger.warning("Checkpoint migration failed: %s", exc)


def load_checkpoint(db_path: Path) -> Dict[str, str]:
    p = db_path / "ingest_checkpoint.json"
    if not p.exists():
        return {}
    try:
        with open(p, encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return {}


def save_checkpoint(db_path: Path, data: Dict[str, str]) -> None:
    p = db_path / "ingest_checkpoint.json"
    with open(p, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)


def append_manifest(db_path: Path, record: Dict[str, Any]) -> None:
    p = db_path / "ingestion_history.jsonl"
    with open(p, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_ingestion_config(db_path: Path, model: str) -> None:
    cfg = {"embedding_model": model, "ingestion_version": INGESTION_VERSION}
    with open(db_path / "ingestion_config.json", "w", encoding="utf-8") as fh:
        json.dump(cfg, fh, indent=2)



def validate_embedding_dimension(
    coll: Any,
    embedder: OllamaEmbeddings,
    collection_name: str,
    embed_model: str,
) -> Optional[str]:
    """If the collection already has vectors, ensure *embed_model* matches their dimension.

    Returns a human-readable error string, or None when OK / when no check applies.
    """
    schema_dim: Optional[int] = None
    try:
        m = getattr(coll, "_model", None)
        if m is not None:
            raw = getattr(m, "dimension", None)
            if raw is not None and int(raw) > 0:
                schema_dim = int(raw)
    except Exception:  # pragma: no cover
        schema_dim = None  # pragma: no cover

    try:
        n = int(coll.count())
    except Exception:
        return None
    if n == 0:
        if schema_dim is not None:
            try:
                probe = embedder.embed_query("__dimension_probe__")
                probe_dim = len(probe)
            except Exception as exc:
                return f"Could not probe embedding model {embed_model!r}: {exc}"
            if schema_dim != probe_dim:
                return (
                    f"Embedding dimension mismatch for collection {collection_name!r}: "
                    f"collection schema expects {schema_dim} dimensions but model "
                    f"{embed_model!r} produces {probe_dim}. Re-ingest with the original "
                    f"model, or delete this collection first "
                    f"(e.g. nomic-embed-text → 768, mxbai-embed-large → 1024)."
                )
        return None
    try:
        rows = coll.get(limit=1, include=["embeddings"])
        embs = rows.get("embeddings") or []
        emb0 = embs[0] if embs else None
        if not embs or emb0 is None:
            return None
        existing_dim = len(emb0)
    except Exception:
        return None
    try:
        probe = embedder.embed_query("__dimension_probe__")
        probe_dim = len(probe)
    except Exception as exc:  # pragma: no cover
        return f"Could not probe embedding model {embed_model!r}: {exc}"  # pragma: no cover
    if existing_dim != probe_dim:
        return (
            f"Embedding dimension mismatch for collection {collection_name!r}: "
            f"existing index uses {existing_dim} dimensions but model {embed_model!r} "
            f"produces {probe_dim}. Re-ingest with the same model as the original index "
            f"(e.g. nomic-embed-text → 768, mxbai-embed-large → 1024), or delete this "
            f"collection / use a fresh VectorDB before switching embedding models."
        )
    return None


def update_repos_manifest(db_path: Path, collection_name: str, repo_counts: Dict[str, int]) -> None:
    path = db_path / "repos_manifest.json"
    data: Dict[str, Any] = {}
    if path.exists():
        try:
            with open(path, encoding="utf-8") as fh:
                data = json.load(fh)
        except Exception:
            data = {}
    if "by_collection" not in data or not isinstance(data["by_collection"], dict):
        data["by_collection"] = {}
    data["by_collection"][collection_name] = repo_counts
    if collection_name.endswith("_code"):
        for k, v in repo_counts.items():
            data[k] = v
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)

