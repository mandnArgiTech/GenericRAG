#!/usr/bin/env python3
"""Parse RFC plaintext from ``download_manifest.json`` into p1.txt-style JSONL chunks.

Author: deviprasad
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Repo root on PYTHONPATH when run as ``python scripts/parse_rfc_for_coder_agent.py``
_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from ingest.chunking.rfc.rfc_c_hints import extract_c_hints, extract_rfc_dependencies
from ingest.chunking.rfc.rfc_chunking import chunk_rfc

_CHUNK_TYPE_MAP = {
    "section": "prose",
    "sliding_window": "prose",
    "fsm_transition_rule": "state_logic",
}

_ALLOWED = frozenset(
    {"struct_definition", "state_logic", "constants", "implementation_rule", "prose"}
)


def _load_manifest(path: Path) -> Dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _find_manifest(cli_path: Optional[Path]) -> Path:
    if cli_path and cli_path.is_file():
        return cli_path
    for candidate in (
        _REPO / "download_manifest.json",
        _REPO / "tcp_ip_stack_rfcs" / "download_manifest.json",
    ):
        if candidate.is_file():
            return candidate
    raise FileNotFoundError("download_manifest.json not found; pass --manifest")


def _map_chunk_type(internal: str) -> str:
    t = _CHUNK_TYPE_MAP.get(internal, internal)
    if t not in _ALLOWED:
        return "prose"
    return t


def _hierarchical_path(
    *,
    layer_folder: str,
    protocol: str,
    rfc_number: str,
    section_number: str,
    section_title: str,
    field_label: str = "",
) -> str:
    parts = [
        f"[Layer: {layer_folder}]",
        f"[Protocol: {protocol} (RFC {rfc_number})]",
    ]
    if section_number or section_title:
        sec = section_number or ""
        if section_title:
            sec = f"{sec} {section_title}".strip()
        parts.append(f"[Section: {sec}]")
    fl = (field_label or "").strip()
    if fl:
        parts.append(f"[Field: {fl[:120]}]")
    return " -> ".join(parts)


def _record_to_p1_row(
    entry: Dict[str, Any],
    text: str,
    meta: Dict[str, str],
    base_dir: Path,
) -> Dict[str, Any]:
    rfc_num = str(entry.get("rfc_number", meta.get("rfc_number", "")))
    layer = entry.get("layer_folder", "")
    protocol_name = entry.get("protocol", "")
    hints = extract_c_hints(text)
    deps = extract_rfc_dependencies(text)
    chunk_type = _map_chunk_type(meta.get("chunk_type", "prose"))
    row_meta_deps = meta.get("rfc_dependencies", "")
    if row_meta_deps:
        for p in row_meta_deps.split(","):
            p = p.strip()
            if p and p not in deps:
                deps.append(p)
    hp = _hierarchical_path(
        layer_folder=layer,
        protocol=protocol_name,
        rfc_number=rfc_num,
        section_number=meta.get("section_number", ""),
        section_title=meta.get("section_title", ""),
        field_label=meta.get("hierarchical_field", ""),
    )
    # p1.txt: absolute hierarchy at the very beginning of every chunk's content.
    body = text.strip()
    content = f"{hp}\n\n{body}" if body else hp
    return {
        "layer": layer,
        "rfc_number": rfc_num,
        "protocol_name": protocol_name,
        "hierarchical_path": hp,
        "chunk_type": chunk_type,
        "c_implementation_hints": {
            "requires_network_byte_order": bool(hints["requires_network_byte_order"]),
            "byte_alignment_warning": hints.get("byte_alignment_warning") or "",
        },
        "content": content,
        "dependencies": deps,
    }


def _default_base_dir(manifest_path: Path, man: Dict[str, Any]) -> Path:
    od = man.get("out_dir")
    if od:
        return Path(str(od))
    return manifest_path.parent


def run(
    *,
    manifest_path: Path,
    base_dir: Optional[Path],
    out_path: Path,
    status_filter: Optional[str] = None,
) -> Tuple[int, int]:
    man = _load_manifest(manifest_path)
    base = base_dir or _default_base_dir(manifest_path, man)
    entries: List[Dict[str, Any]] = list(man.get("entries") or [])
    written = 0
    skipped = 0
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as out_f:
        for entry in entries:
            st = entry.get("status", "ok")
            if status_filter and st != status_filter:
                continue
            if st not in ("ok", "skipped_exists"):
                skipped += 1
                continue
            rel = entry.get("relative_path")
            if not rel:
                skipped += 1
                continue
            txt_path = base / rel
            if not txt_path.is_file():
                skipped += 1
                continue
            text = txt_path.read_text(encoding="utf-8", errors="replace")
            chunks = chunk_rfc(text, str(txt_path))
            for piece, meta in chunks:
                row = _record_to_p1_row(entry, piece, meta, base)
                out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
                written += 1
    return written, skipped


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, default=None, help="Path to download_manifest.json")
    p.add_argument(
        "--base-dir",
        type=Path,
        default=None,
        help="Directory containing RFC tree (default: manifest parent or tcp_ip_stack_rfcs)",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=_REPO / "rfc_chunks.jsonl",
        help="Output JSONL path",
    )
    args = p.parse_args(argv)
    manifest = _find_manifest(args.manifest)
    n, sk = run(manifest_path=manifest, base_dir=args.base_dir, out_path=args.out)
    print(f"wrote {n} lines to {args.out} (skipped entries: {sk})")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
