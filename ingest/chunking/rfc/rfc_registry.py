"""Plugin registry for per-RFC section parsers (self-registering modules).

Each ``rfcNNNN_*.py`` may call ``register_rfc_parser()`` at import time.
``rfc_chunking.discover_and_import()`` loads those modules once.

Author: deviprasad
"""
from __future__ import annotations

import importlib
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Tuple

_log = logging.getLogger(__name__)

# (section_body, section_number) -> list of (chunk_text, partial_metadata)
ParseFn = Callable[[str, str], List[Tuple[str, Dict[str, str]]]]
StripFn = Callable[[str, str], str]


@dataclass(frozen=True)
class RfcSectionParser:
    """One hook: when ``section_pattern`` matches ``section_number``, run parse then strip."""

    rfc_number: str
    section_pattern: str  # regex fullstring match on section_number, or "*" for all
    parse_fn: ParseFn
    strip_fn: StripFn
    priority: int = 0


_RFC_PARSERS: Dict[str, List[RfcSectionParser]] = {}
_discovered: bool = False


def register_rfc_parser(parser: RfcSectionParser) -> None:
    """Append a parser for ``parser.rfc_number`` (digits only, e.g. ``"1661"``)."""
    key = str(parser.rfc_number).strip()
    lst = _RFC_PARSERS.setdefault(key, [])
    sig = (parser.section_pattern, parser.priority, id(parser.parse_fn), id(parser.strip_fn))
    if any(
        (p.section_pattern, p.priority, id(p.parse_fn), id(p.strip_fn)) == sig for p in lst
    ):
        return
    lst.append(parser)
    lst.sort(key=lambda p: -p.priority)


def get_parsers_for_rfc(rfc_number: str) -> List[RfcSectionParser]:
    return list(_RFC_PARSERS.get(str(rfc_number).strip(), []))


def section_pattern_matches(pattern: str, section_number: str) -> bool:
    if pattern == "*":
        return True
    return bool(re.match(pattern, section_number))


def discover_and_import() -> None:
    """Import every ``rfc<digits>_*.py`` in this package so registrations run."""
    global _discovered
    if _discovered:
        return
    pkg_dir = Path(__file__).resolve().parent
    name_pat = re.compile(r"^rfc\d+_.+\.py$")
    for path in sorted(pkg_dir.glob("rfc*.py")):
        if not name_pat.match(path.name):
            continue
        mod = f"ingest.chunking.rfc.{path.stem}"
        try:
            importlib.import_module(mod)
        except Exception as exc:  # pragma: no cover
            _log.warning("rfc_registry: skip import %s: %s", mod, exc)
    _discovered = True


def reset_registry_for_tests() -> None:
    """Clear parsers and discovery flag (tests only)."""
    global _discovered
    _RFC_PARSERS.clear()
    _discovered = False
