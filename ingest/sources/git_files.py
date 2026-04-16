"""Git-aware file walking and nested .gitignore matching.

Author: deviprasad
"""
from __future__ import annotations

import csv
import os
import re
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from ingest.core.constants import IGNORED_DIRS, IGNORED_EXTS
from ingest.core.deps import pathspec

def _respect_gitignore() -> bool:
    return os.environ.get("RESPECT_GITIGNORE", "1").strip().lower() not in ("0", "false", "no")


def _gitignore_parent_chain_dirs(rel_posix: str) -> List[str]:
    """Parent directories (posix paths from repo root) whose `.gitignore` may apply to *rel_posix*."""
    rel_posix = rel_posix.replace("\\", "/").strip("/")
    parts = [p for p in rel_posix.split("/") if p]
    if not parts:
        return [""]
    dirs: List[str] = [""]
    for i in range(len(parts) - 1):
        dirs.append("/".join(parts[: i + 1]))
    return dirs


def _relpath_under_gitignore_dir(dir_rel: str, full_rel_posix: str) -> str:
    if not dir_rel:
        return full_rel_posix
    prefix = dir_rel + "/"
    if not full_rel_posix.startswith(prefix):
        return full_rel_posix
    return full_rel_posix[len(prefix) :]


def _gitignore_file_path(root_dir: Path, dir_rel_posix: str) -> Path:
    if not dir_rel_posix:
        return root_dir / ".gitignore"
    return root_dir.joinpath(*dir_rel_posix.split("/")) / ".gitignore"


def _read_gitignore_spec_for_dir(root_dir: Path, dir_rel_posix: str, cache: Dict[str, Any]) -> Optional[Any]:
    if dir_rel_posix in cache:
        return cache[dir_rel_posix]
    if not _respect_gitignore() or pathspec is None:
        cache[dir_rel_posix] = None
        return None
    gi = _gitignore_file_path(root_dir, dir_rel_posix)
    if not gi.is_file():
        cache[dir_rel_posix] = None
        return None
    try:
        lines = gi.read_text(encoding="utf-8", errors="replace").splitlines()
        spec = pathspec.PathSpec.from_lines("gitwildmatch", lines)
        cache[dir_rel_posix] = spec
        return spec
    except Exception:
        cache[dir_rel_posix] = None
        return None


def _path_matches_any_nested_gitignore(
    root_dir: Path,
    rel_posix: str,
    cache: Dict[str, Any],
    *,
    is_dir: bool,
) -> bool:
    """True if any ancestor `.gitignore` excludes this path (Git-style, per-directory rules)."""
    if not _respect_gitignore() or pathspec is None:
        return False
    rel_norm = rel_posix.replace("\\", "/").strip("/")
    for drel in _gitignore_parent_chain_dirs(rel_norm):
        spec = _read_gitignore_spec_for_dir(root_dir, drel, cache)
        if spec is None:
            continue
        rel_for_spec = _relpath_under_gitignore_dir(drel, rel_norm)
        candidates = [rel_for_spec + "/", rel_for_spec] if is_dir else [rel_for_spec]
        for cand in candidates:
            if not cand:
                continue
            try:
                if spec.match_file(cand):
                    return True
            except Exception:
                continue
    return False


def _load_gitignore_spec(root: Path) -> Any:
    """Load only the repository root `.gitignore` (tests and simple callers)."""
    c: Dict[str, Any] = {}
    return _read_gitignore_spec_for_dir(root.resolve(), "", c)


def git_checkpoint_head_key(collection_name: str) -> str:
    return f"{collection_name}::git_head"


def _git_run(root: Path, *git_args: str, timeout: float = 120.0) -> Tuple[int, str, str]:
    try:
        proc = subprocess.run(
            ["git", "-C", str(root), *git_args],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        return proc.returncode, (proc.stdout or "").strip(), (proc.stderr or "").strip()
    except FileNotFoundError:
        return 127, "", "git not found"
    except subprocess.TimeoutExpired:
        return 124, "", "timeout"


def git_diff_file_sets(
    root: Path, base_ref: str
) -> Tuple[Optional[Set[str]], Optional[Set[str]], Optional[str]]:
    """Return (modified_paths, deleted_paths, head_sha) relative to *root*, or Nones if unusable."""
    code, _, _ = _git_run(root, "rev-parse", "--is-inside-work-tree")
    if code != 0:
        return None, None, None
    hc, head_out, _ = _git_run(root, "rev-parse", "HEAD")
    if hc != 0 or not head_out:
        return None, None, None
    head_sha = head_out.splitlines()[0].strip()

    def collect(diff_filter: str) -> Set[str]:
        rc, out, _ = _git_run(
            root,
            "diff",
            "--name-only",
            f"--diff-filter={diff_filter}",
            f"{base_ref}...HEAD",
        )
        if rc != 0:
            rc, out, _ = _git_run(
                root,
                "diff",
                "--name-only",
                f"--diff-filter={diff_filter}",
                base_ref,
                head_sha,
            )
        if rc != 0:
            return set()
        return {ln.strip().replace("\\", "/") for ln in out.splitlines() if ln.strip()}

    return collect("ACMR"), collect("D"), head_sha


def iter_files(
    root: Path,
    exts: Optional[set] = None,
    skip_dirs: Optional[set] = None,
    skip_exts: Optional[set] = None,
) -> List[Path]:
    root_dir = root.resolve()
    gi_cache: Dict[str, Any] = {}
    if root.is_file():
        p = root
        suf = p.suffix.lower()
        if skip_exts and suf in skip_exts:
            return []  # pragma: no cover
        if exts and suf not in exts:
            return []  # pragma: no cover
        return [p]
    out: List[Path] = []
    for dirpath, dirnames, files in os.walk(root_dir):
        if skip_dirs:
            dirnames[:] = [d for d in dirnames if d not in skip_dirs]
        if _respect_gitignore() and pathspec is not None:
            try:
                rel_parent = Path(dirpath).resolve().relative_to(root_dir).as_posix()
            except ValueError:
                rel_parent = ""
            dirnames[:] = [
                d
                for d in dirnames
                if not _path_matches_any_nested_gitignore(
                    root_dir,
                    f"{rel_parent}/{d}" if rel_parent else d,
                    gi_cache,
                    is_dir=True,
                )
            ]
        for fn in files:
            p = Path(dirpath) / fn
            suf = p.suffix.lower()
            if skip_exts and suf in skip_exts:
                continue
            if exts and suf not in exts:
                continue
            if _respect_gitignore() and pathspec is not None:
                try:
                    rel = p.resolve().relative_to(root_dir).as_posix()
                    if _path_matches_any_nested_gitignore(root_dir, rel, gi_cache, is_dir=False):
                        continue
                except Exception:
                    pass  # pragma: no cover
            out.append(p)
    return sorted(out)

