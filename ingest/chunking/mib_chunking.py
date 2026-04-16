"""SNMP MIB text chunking.

Author: deviprasad
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

from ingest.core.constants import MIB_MODULE_CONCEPTS

def chunk_mib(text: str, path: Path, skip_deprecated: bool = True) -> List[Tuple[str, Dict[str, str]]]:
    mod_match = re.search(r"([\w-]+)\s+DEFINITIONS\s*::=", text)
    mib_module = mod_match.group(1) if mod_match else path.stem
    extra_concepts = MIB_MODULE_CONCEPTS.get(mib_module.upper(), "")
    blocks = re.split(
        r"(?=\n\s*[A-Za-z0-9-]+\s+(?:OBJECT-TYPE|NOTIFICATION-TYPE|MODULE-IDENTITY|TEXTUAL-CONVENTION)\b)",
        text,
    )
    out: List[Tuple[str, Dict[str, str]]] = []
    for block in blocks:
        block = block.strip()
        if not re.search(
            r"\b(OBJECT-TYPE|NOTIFICATION-TYPE|MODULE-IDENTITY|TEXTUAL-CONVENTION)\b", block
        ):
            continue
        name_m = re.match(
            r"^\s*([A-Za-z0-9-]+)\s+(OBJECT-TYPE|NOTIFICATION-TYPE|MODULE-IDENTITY|TEXTUAL-CONVENTION)",
            block,
        )
        if not name_m:
            continue  # pragma: no cover
        obj_name = name_m.group(1)
        kind = name_m.group(2)
        status_m = re.search(r"STATUS\s+(\w+)", block)
        status = status_m.group(1) if status_m else ""
        if skip_deprecated and status.lower() in ("deprecated", "obsolete"):
            continue
        syntax_m = re.search(
            r"SYNTAX\s+(.+?)(?:\n\s*(?:MAX-ACCESS|ACCESS|STATUS|DESCRIPTION|DISPLAY-HINT)\b)", block, re.S
        )
        syntax = syntax_m.group(1).strip() if syntax_m else ""
        maxacc = ""
        mmax = re.search(r"MAX-ACCESS\s+(\S+)", block)
        if mmax:
            maxacc = mmax.group(1)
        oid_m = re.search(r"::=\s*\{\s*([^\}]+)\}", block)
        oid_path = oid_m.group(1).strip() if oid_m else ""
        obj_type = "scalar"
        if kind == "NOTIFICATION-TYPE":
            obj_type = "notification"  # pragma: no cover
        elif kind == "MODULE-IDENTITY":
            obj_type = "identity"  # pragma: no cover
        elif kind == "TEXTUAL-CONVENTION":
            obj_type = "textual_convention"  # pragma: no cover
        elif "ENTRY" in obj_name.upper() and "TABLE" not in obj_name.upper():
            obj_type = "row"  # pragma: no cover
        elif "TABLE" in obj_name.upper():
            obj_type = "table"  # pragma: no cover
        elif "INDEX" in block or ("SEQUENCE" in block.upper() and "OF" in block.upper()):
            obj_type = "table"  # pragma: no cover
        elif "AUGMENTS" in block.upper() or "column" in block.lower():
            obj_type = "column"  # pragma: no cover
        out.append(
            (
                f"{obj_name}\n{block[:12000]}",
                {
                    "chunk_strategy": "mib",
                    "chunk_type": "mib_object",
                    "chunk_name": obj_name,
                    "mib_module": mib_module,
                    "object_name": obj_name,
                    "oid_path": oid_path,
                    "syntax": syntax[:500],
                    "max_access": maxacc,
                    "status": status,
                    "object_type": obj_type,
                    "section": mib_module,
                    "chunk_index": str(len(out)),
                },
            )
        )
    if extra_concepts and out:
        t0, m0 = out[0]
        prev = m0.get("concepts", "")
        tokens: Set[str] = set(iter_concept_ids(prev))
        tokens.update(x.strip() for x in extra_concepts.split(",") if x.strip())
        m0["concepts"] = format_concepts_field(tokens)
        out[0] = (t0, m0)
    if not out:
        out.append(
            (
                text[:8000],
                {
                    "chunk_strategy": "mib",
                    "chunk_type": "mib_file",
                    "mib_module": mib_module,
                    "chunk_name": mib_module,
                    "chunk_index": "0",
                },
            )
        )
    return out

