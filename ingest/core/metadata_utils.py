"""Chunk IDs, collection names, metadata normalization, concept tagging.

Author: deviprasad
"""
from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from ingest.core.constants import CONTENT_TYPE_SIGNALS, METADATA_KEYS

logger = logging.getLogger("ingest")


def make_chunk_id(source: str, chunk_index: int, content: str) -> str:
    h = hashlib.sha256(f"{source}::{chunk_index}::{content}".encode("utf-8", errors="replace"))
    return h.hexdigest()[:20]


def resolve_collection(mode: str, domain: str, override: Optional[str]) -> str:
    if override:
        return override
    routes = {
        "code": "{domain}_code",
        "domain": "{domain}_domain",
        "rfc": "rfc",
        "rally": "{domain}_internal",
        "customer": "{domain}_customer",
        "mib": "{domain}_mib",
        "wiki": "{domain}_wiki",
        "release-notes": "{domain}_releases",
        "theory": "theory",
        "community": "community",
    }
    key = routes.get(mode, "{domain}_misc")
    return key.format(domain=domain)


def empty_metadata() -> Dict[str, str]:
    return {k: "" for k in METADATA_KEYS}


def finalize_metadata(meta: Dict[str, Any]) -> Dict[str, str]:
    out = empty_metadata()
    for k in METADATA_KEYS:
        if k in meta and meta[k] is not None and meta[k] != "":
            out[k] = str(meta[k])
    return out


def detect_content_type(text: str) -> str:
    tl = text.lower()
    scores = {
        ct: sum(1 for s in sigs if s in tl) for ct, sigs in CONTENT_TYPE_SIGNALS.items()
    }
    scores = {k: v for k, v in scores.items() if v > 0}
    return max(scores, key=scores.get) if scores else "general"


def load_concept_registry(path: Path) -> Dict[str, Dict[str, str]]:
    if path.exists():
        try:
            with open(path, encoding="utf-8") as fh:
                data = json.load(fh)
            if isinstance(data, dict):
                return {k: {str(a): str(b) for a, b in v.items()} for k, v in data.items() if isinstance(v, dict)}
        except Exception as exc:  # pragma: no cover
            logger.warning("Could not load concept registry: %s", exc)  # pragma: no cover
    # minimal defaults if missing
    return {
        "nms": {"vlan": "vlan", "snmp": "snmp_polling"},
        "occt": {"fillet": "fillet"},
        "spice": {"transient": "transient_analysis"},
        "kicad": {"footprint": "footprint"},
        "geda": {"netlist": "netlist"},
        "general": {},
    }


def extract_concepts(text: str, domain: str, registry: Dict[str, Dict[str, str]]) -> str:
    dom = domain if domain in registry else "general"
    table = registry.get(dom, {})
    if not table:
        return ""
    tl = text.lower()
    found = set()
    for keyword in sorted(table.keys(), key=len, reverse=True):
        if keyword.lower() in tl:
            found.add(table[keyword])
    return format_concepts_field(found) if found else ""


def iter_concept_ids(concepts_field: str) -> List[str]:
    """Split stored concepts metadata (pipe- or comma-delimited; supports legacy rows)."""
    s = (concepts_field or "").strip()
    if not s:
        return []
    if s.startswith("|"):
        return [x.strip() for x in s.strip("|").split("|") if x.strip()]
    return [x.strip() for x in s.split(",") if x.strip()]


def format_concepts_field(ids: Iterable[str]) -> str:
    """Pipe-delimited concept ids for Chroma $contains token search (|id| avoids substring false positives)."""
    unique = sorted({x.strip() for x in ids if x and str(x).strip()})
    return "|" + "|".join(unique) + "|" if unique else ""

