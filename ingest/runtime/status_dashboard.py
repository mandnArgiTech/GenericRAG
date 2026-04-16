"""Chroma collection status printing.

Author: deviprasad
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path

import chromadb

from ingest.core.file_utils import _safe_count

def print_status_dashboard(db_path: Path) -> None:
    print("=" * 64)
    print("            DOMAIN RAG — KNOWLEDGE BASE STATUS")
    print("=" * 64)
    client = chromadb.PersistentClient(path=str(db_path))
    cols = client.list_collections()
    concept_counts: Counter[str] = Counter()
    rows = []
    for c in sorted(cols, key=lambda x: x.name):
        coll = client.get_collection(c.name)
        n = _safe_count(coll)
        if n == 0:
            rows.append((c.name, 0, 0, ""))  # pragma: no cover
            continue  # pragma: no cover
        sample = coll.get(include=["metadatas"], limit=min(n, 8000))
        metas = sample.get("metadatas") or []
        sources = {str(m.get("source", "")) for m in metas if m}
        for m in metas:
            if not m:
                continue  # pragma: no cover
            cs = m.get("concepts", "")
            if cs:
                for part in iter_concept_ids(str(cs)):
                    concept_counts[part] += 1
        dates = [str(m.get("ingestion_date", "")) for m in metas if m and m.get("ingestion_date")]
        last_ing = max(dates) if dates else ""
        rows.append((c.name, n, len(sources), last_ing))
    hdr = f"{'Collection':<22} {'Chunks':>8} {'Sources':>8} {'Last Ingested':<22}"
    print(hdr)
    print("-" * 64)
    for name, n, sc, li in rows:
        print(f"{name:<22} {n:>8,} {sc:>8} {li:<22}")
    print("=" * 64)
    top = concept_counts.most_common(15)
    if top:
        print("Top concepts:", ", ".join(f"{k}({v})" for k, v in top))

