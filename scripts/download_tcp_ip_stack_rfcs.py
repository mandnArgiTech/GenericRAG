#!/usr/bin/env python3
"""
Download every RFC listed in ``tcp_ip_stack_rfc_guide.json`` into layer-named subfolders.

Reads ``tcp_ip_stack_guide.architecture_layers[*].layer`` for directory names and
``protocols[*].rfc`` (e.g. ``RFC 791``) for RFC numbers. Files are saved as
``rfc<number>.txt`` under each layer folder under ``--out-dir``.

Author: deviprasad
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.error
import urllib.request
from datetime import date, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple


RFC_RE = re.compile(r"RFC\s+(\d+)", re.I)
# Plaintext as published by RFC Editor (canonical for this repo's chunkers).
RFC_URL = "https://www.rfc-editor.org/rfc/rfc{n}.txt"


def _slug_layer(name: str) -> str:
    """Filesystem-safe folder name from JSON ``layer`` string."""
    s = name.strip().replace("&", "and")
    for ch in '\\/:*?"<>|':
        s = s.replace(ch, "_")
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "layer"


def _iter_rfc_jobs(data: Dict[str, Any]) -> List[Tuple[str, int, str, str]]:
    """
    Return list of (layer_folder, rfc_number, rfc_label, protocol) for each protocol entry.
    """
    root = data.get("tcp_ip_stack_guide") or {}
    layers = root.get("architecture_layers") or []
    jobs: List[Tuple[str, int, str, str]] = []
    for block in layers:
        layer_name = str(block.get("layer") or "unknown_layer")
        folder = _slug_layer(layer_name)
        for proto in block.get("protocols") or []:
            label = str(proto.get("rfc") or "").strip()
            m = RFC_RE.search(label)
            if not m:
                print(f"skip (no RFC number): {label!r} in layer {layer_name!r}", file=sys.stderr)
                continue
            num = int(m.group(1))
            protocol = str(proto.get("protocol") or "").strip()
            jobs.append((folder, num, label, protocol))
    return jobs


def _download(url: str, dest: Path, timeout: float) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "POC-tcp-ip-rfc-fetch/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 — fixed RFC Editor host
        body = resp.read()
    dest.write_bytes(body)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--json",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "tcp_ip_stack_rfc_guide.json",
        help="Path to tcp_ip_stack_rfc_guide.json",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "tcp_ip_stack_rfcs",
        help="Root directory for layer subfolders and rfc*.txt files",
    )
    p.add_argument("--timeout", type=float, default=60.0, help="HTTP timeout seconds per RFC")
    p.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if destination file already exists",
    )
    args = p.parse_args()

    if not args.json.is_file():
        print(f"JSON not found: {args.json}", file=sys.stderr)
        return 2

    data = json.loads(args.json.read_text(encoding="utf-8"))
    jobs = _iter_rfc_jobs(data)
    if not jobs:
        print("No RFC entries found in JSON.", file=sys.stderr)
        return 3

    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest: List[Dict[str, Any]] = []
    ok, fail = 0, 0

    for folder, num, label, protocol in jobs:
        rel = Path(folder) / f"rfc{num}.txt"
        dest = args.out_dir / rel
        url = RFC_URL.format(n=num)
        entry: Dict[str, Any] = {
            "layer_folder": folder,
            "rfc_number": num,
            "rfc_label": label,
            "protocol": protocol,
            "url": url,
            "relative_path": str(rel).replace("\\", "/"),
            "fetched_at_utc": date.today().isoformat(),
        }
        if dest.is_file() and not args.force:
            entry["status"] = "skipped_exists"
            manifest.append(entry)
            ok += 1
            print(f"skip exists {rel}")
            continue
        try:
            _download(url, dest, args.timeout)
            entry["status"] = "ok"
            entry["bytes"] = dest.stat().st_size
            manifest.append(entry)
            ok += 1
            print(f"ok {rel} ({entry['bytes']} bytes)")
        except (urllib.error.URLError, OSError) as exc:
            entry["status"] = "error"
            entry["error"] = str(exc)
            manifest.append(entry)
            fail += 1
            print(f"FAIL {rel} <- {url}\n  {exc}", file=sys.stderr)

    summary = {
        "generated_at_utc": date.today().isoformat(),
        "json_source": str(args.json.resolve()),
        "out_dir": str(args.out_dir.resolve()),
        "ok_or_skipped": ok,
        "failed": fail,
        "entries": manifest,
    }
    man_path = args.out_dir / "download_manifest.json"
    man_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nWrote manifest: {man_path}")
    return 0 if fail == 0 else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
