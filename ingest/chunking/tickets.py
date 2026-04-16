"""Rally and customer ticket chunking.

Author: deviprasad
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

from ingest.core.file_utils import strip_html

def _html_to_text(s: str) -> str:
    return strip_html(s)


def chunk_rally_ticket(obj: Dict[str, Any], source: str) -> List[Tuple[str, Dict[str, str]]]:
    title = str(obj.get("Name") or obj.get("title") or "")
    desc = _html_to_text(str(obj.get("Description") or obj.get("description") or ""))
    resolution = _html_to_text(str(obj.get("Resolution") or obj.get("resolution") or ""))
    discussion = _html_to_text(str(obj.get("Discussion") or obj.get("discussion") or ""))
    fid = str(obj.get("FormattedID") or obj.get("id") or source)
    meta_base = {
        "chunk_strategy": "rally_ticket",
        "chunk_type": "ticket",
        "rally_id": fid,
        "artifact_type": str(obj.get("_type") or obj.get("artifact_type") or "artifact"),
        "state": str(obj.get("State") or obj.get("state") or ""),
        "priority": str(obj.get("Priority") or obj.get("priority") or ""),
        "severity": str(obj.get("Severity") or obj.get("severity") or ""),
        "has_resolution": "true" if resolution.strip() else "false",
        "iteration": str(obj.get("Iteration") or ""),
        "release": str(obj.get("Release") or ""),
        "created_date": str(obj.get("CreationDate") or obj.get("created") or ""),
        "closed_date": str(obj.get("ClosedDate") or obj.get("closed") or ""),
        "tags": str(obj.get("Tags") or obj.get("tags") or ""),
    }
    part1 = f"{title}\n\n{desc}\n\nResolution:\n{resolution}".strip()
    blob = f"{part1}\n\nDiscussion:\n{discussion}".strip()
    if len(blob) <= 8000:
        return [(blob, {**meta_base, "chunk_name": title, "chunk_index": "0"})]
    chunks: List[Tuple[str, Dict[str, str]]] = [  # pragma: no cover
        (part1, {**meta_base, "chunk_name": title, "chunk_index": "0"})
    ]
    if discussion.strip():  # pragma: no cover
        for k, piece in enumerate(_split_paragraphs(discussion, 4000, 8000)):  # pragma: no cover
            chunks.append(  # pragma: no cover
                (
                    f"Discussion ({fid}) part {k+1}:\n{piece}",
                    {**meta_base, "chunk_name": f"{fid}-disc-{k+1}", "chunk_index": str(k + 1)},
                )
            )
    return chunks  # pragma: no cover


def chunk_customer_ticket(obj: Dict[str, Any], source: str) -> List[Tuple[str, Dict[str, str]]]:
    out: List[Tuple[str, Dict[str, str]]] = []
    for text, meta in chunk_rally_ticket(obj, source):
        t = sanitize_pii(text) if sanitize_pii else text
        tid = str(obj.get("ticket_id") or obj.get("FormattedID") or source)
        mm = {
            **meta,
            "chunk_strategy": "customer_ticket",
            "ticket_id": tid,
            "customer_type": "anonymized",
            "has_workaround": "true" if re.search(r"workaround|mitigation", t, re.I) else "false",
            "product_version": str(obj.get("product_version") or ""),
            "related_rally": str(obj.get("related_rally") or ""),
        }
        mm.pop("rally_id", None)
        out.append((t, mm))
    return out

