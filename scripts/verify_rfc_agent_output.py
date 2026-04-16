#!/usr/bin/env python3
"""Smoke-test: chunk all manifest RFCs, validate JSONL schema, struct counts.

Author: deviprasad
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Set

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from ingest.chunking.rfc.rfc_chunking import chunk_rfc

_KEYS: Set[str] = {
    "layer",
    "rfc_number",
    "protocol_name",
    "hierarchical_path",
    "chunk_type",
    "c_implementation_hints",
    "content",
    "dependencies",
}
_ALLOWED_CT = frozenset(
    {"struct_definition", "state_logic", "constants", "implementation_rule", "prose"}
)
_STRUCT_RFCS = frozenset({"791", "768", "826", "9293", "8200"})


def _validate_jsonl(path: Path) -> int:
    n = 0
    with path.open(encoding="utf-8") as f:
        for line in f:
            o: Dict[str, Any] = json.loads(line)
            missing = _KEYS - o.keys()
            if missing:
                raise SystemExit(f"line {n+1} missing keys {missing}")
            if o["chunk_type"] not in _ALLOWED_CT:
                raise SystemExit(f"line {n+1} bad chunk_type {o['chunk_type']!r}")
            hints = o["c_implementation_hints"]
            if not isinstance(hints, dict) or "requires_network_byte_order" not in hints:
                raise SystemExit(f"line {n+1} bad c_implementation_hints")
            n += 1
    return n


def _smoke_manifest_rfcs() -> None:
    man_path = _REPO / "tcp_ip_stack_rfcs" / "download_manifest.json"
    man = json.loads(man_path.read_text(encoding="utf-8"))
    base = Path(man.get("out_dir") or man_path.parent)
    for e in man.get("entries") or []:
        if e.get("status") not in ("ok", "skipped_exists"):
            continue
        rel = e.get("relative_path")
        if not rel:
            continue
        p = base / rel
        if not p.is_file():
            continue
        text = p.read_text(encoding="utf-8", errors="replace")
        chunks = chunk_rfc(text, str(p))
        assert len(chunks) >= 1, f"{p} produced no chunks"
        rfc = str(e.get("rfc_number", ""))
        types = Counter(m.get("chunk_type") for _, m in chunks)
        if rfc in _STRUCT_RFCS:
            assert types.get("struct_definition", 0) >= 1, f"RFC {rfc} expected struct_definition"


def main() -> int:
    _smoke_manifest_rfcs()
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as tmp:
        out = Path(tmp.name)
    try:
        subprocess.run(
            [sys.executable, str(_REPO / "scripts" / "parse_rfc_for_coder_agent.py"), "--out", str(out)],
            check=True,
            cwd=str(_REPO),
        )
        n = _validate_jsonl(out)
        print(f"verify ok: {n} jsonl rows, manifest smoke ok")
    finally:
        out.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
