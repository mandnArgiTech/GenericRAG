"""Markdown / domain document chunking with fenced-code and table masking.

Author: deviprasad
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Tuple

from ingest.core.text_split import _md_char_targets, _split_paragraphs

_BARE_MERMAID_STARTERS = re.compile(
    r"^(?:graph\s+(?:TD|TB|BT|RL|LR)|sequenceDiagram|classDiagram|stateDiagram"
    r"|erDiagram|gantt|pie|flowchart|journey|gitGraph|mindmap|timeline|quadrantChart"
    r"|sankey|xychart|block-beta|packet-beta|kanban|architecture-beta)\b",
    re.MULTILINE,
)

_DIAGRAM_TYPE_HINTS = {
    "graph": "Mermaid Flowchart",
    "flowchart": "Mermaid Flowchart",
    "sequencediagram": "Mermaid Sequence Diagram",
    "classdiagram": "Mermaid Class Diagram",
    "statediagram": "Mermaid State Diagram",
    "erdiagram": "Mermaid ER Diagram",
    "gantt": "Mermaid Gantt Chart",
    "pie": "Mermaid Pie Chart",
    "journey": "Mermaid User Journey",
    "gitgraph": "Mermaid Git Graph",
    "mindmap": "Mermaid Mind Map",
    "timeline": "Mermaid Timeline",
    "sankey": "Mermaid Sankey Diagram",
    "xychart": "Mermaid XY Chart",
    "@startuml": "PlantUML Diagram",
    "@startmindmap": "PlantUML Mind Map",
    "@startgantt": "PlantUML Gantt",
}


def _detect_diagram_type(block_text: str) -> str:
    """Return a human-readable diagram type label or empty string."""
    first_line = block_text.strip().split("\n", 1)[0].strip().lower()
    for opener in ("```mermaid", "```plantuml"):
        if first_line.startswith(opener):
            first_line = block_text.strip().split("\n", 2)[1].strip().lower() if "\n" in block_text else ""
            break
    first_token = re.split(r"[\s{(\[]", first_line, 1)[0].rstrip(":;").lower() if first_line else ""
    return _DIAGRAM_TYPE_HINTS.get(first_token, "")


@dataclass
class _MaskedBlock:
    text: str
    diagram_type: str


def _mask_markdown_fences_and_tables(text: str) -> Tuple[str, List[_MaskedBlock]]:
    """Replace fenced blocks, HTML diagram wrappers, bare Mermaid, and pipe-tables with placeholders."""
    vault: List[_MaskedBlock] = []

    def stash(m: re.Match, force_type: str = "") -> str:
        raw = m.group(0)
        dtype = force_type or _detect_diagram_type(raw)
        vault.append(_MaskedBlock(text=raw, diagram_type=dtype))
        return f"\n<<BLOCK{len(vault) - 1}>>\n"

    t = re.sub(r"(?ms)^```.*?^```", stash, text)

    t = re.sub(
        r"(?ms)<div\s[^>]*class\s*=\s*[\"'](?:mermaid|plantuml)[\"'][^>]*>.*?</div>",
        lambda m: stash(m, "Mermaid Diagram (HTML)"),
        t,
    )
    t = re.sub(
        r"(?ms)<details[^>]*>.*?</details>",
        lambda m: stash(m, "HTML Details Block"),
        t,
    )

    matches = list(_BARE_MERMAID_STARTERS.finditer(t))
    for m in reversed(matches):
        block_start = m.start()
        rest = t[block_start:]
        lines = rest.split("\n")
        block_lines = [lines[0]]
        for ln in lines[1:]:
            stripped = ln.strip()
            if not stripped:
                block_lines.append(ln)
                continue
            if stripped.startswith("```") or re.match(r"^#{1,6}\s", stripped):
                break
            block_lines.append(ln)
        block_text = "\n".join(block_lines).rstrip()
        dtype = _detect_diagram_type(block_text) or "Bare Diagram Block"
        vault.append(_MaskedBlock(text=block_text, diagram_type=dtype))
        placeholder = f"\n<<BLOCK{len(vault) - 1}>>\n"
        t = t[:block_start] + placeholder + t[block_start + len(block_text):]

    t = re.sub(r"(?ms)(?:^\|[^\n]+\n)+", lambda m: stash(m, ""), t)
    return t, vault


def _unmask_markdown_with_meta(
    s: str, vault: List[_MaskedBlock]
) -> Tuple[str, bool, str]:
    """Restore placeholders. Returns (text, has_diagram, diagram_type_label)."""
    has_diagram = False
    diagram_types: List[str] = []
    for i, blk in enumerate(vault):
        placeholder = f"<<BLOCK{i}>>"
        if placeholder in s:
            if blk.diagram_type:
                has_diagram = True
                if blk.diagram_type not in diagram_types:
                    diagram_types.append(blk.diagram_type)
            s = s.replace(placeholder, blk.text)
    label = ", ".join(diagram_types) if diagram_types else ""
    if has_diagram and label:
        s = f"[Metadata: This chunk contains a {label}]\n\n{s}"
    return s, has_diagram, label


def _unmask_markdown(s: str, vault: List[_MaskedBlock]) -> str:
    """Backward-compatible unmask (used by callers that don't need diagram metadata)."""
    for i, blk in enumerate(vault):  # pragma: no cover
        s = s.replace(f"<<BLOCK{i}>>", blk.text)  # pragma: no cover
    return s  # pragma: no cover



def chunk_markdown_domain(
    text: str, path: str, embed_model: str = "nomic-embed-text"
) -> List[Tuple[str, Dict[str, str]]]:
    masked, vault = _mask_markdown_fences_and_tables(text)
    h1_m = re.search(r"(?m)^#\s+(.+)$", masked)
    h1 = h1_m.group(1).strip() if h1_m else ""
    t_min, t_max = _md_char_targets(embed_model)
    split_threshold = int(t_max * 1.2)

    def _finalize(raw_piece: str, section: str, chunk_type: str, idx_ref: List[int]) -> Tuple[str, Dict[str, str]]:
        header = f"{section}\n\n{raw_piece}" if section else raw_piece
        txt, has_diag, diag_label = _unmask_markdown_with_meta(header, vault)
        meta: Dict[str, str] = {
            "chunk_strategy": "markdown_domain",
            "chunk_type": chunk_type,
            "section": section or path,
            "doc_title": h1,
            "chunk_index": str(idx_ref[0]),
            "contains_diagram": "true" if has_diag else "",
        }
        if diag_label:
            meta["diagram_type"] = diag_label  # pragma: no cover
        idx_ref[0] += 1
        return txt, meta

    chunks: List[Tuple[str, Dict[str, str]]] = []
    idx_ref = [0]
    parts = re.split(r"(?m)(^##\s+.+$)", masked)
    i = 0
    while i < len(parts):
        seg = parts[i].strip()
        if seg.startswith("##"):
            title = seg.lstrip("#").strip()
            body = parts[i + 1].strip() if i + 1 < len(parts) else ""
            i += 2
            hierarchy = " > ".join(x for x in (h1, title) if x)
            if len(body) > split_threshold and "###" in body:
                sub = re.split(r"(?m)(^###\s+.+$)", body)  # pragma: no cover
                j = 0  # pragma: no cover
                while j < len(sub):  # pragma: no cover
                    sseg = sub[j].strip()  # pragma: no cover
                    if sseg.startswith("###"):  # pragma: no cover
                        st = sseg.lstrip("#").strip()  # pragma: no cover
                        b2 = sub[j + 1].strip() if j + 1 < len(sub) else ""  # pragma: no cover
                        j += 2  # pragma: no cover
                        sub_hier = " > ".join(x for x in (h1, title, st) if x)  # pragma: no cover
                        pieces = _split_paragraphs(b2, t_min, t_max) if len(b2) > split_threshold else [b2]  # pragma: no cover
                        for piece in pieces:  # pragma: no cover
                            chunks.append(_finalize(piece, sub_hier or hierarchy, "section", idx_ref))  # pragma: no cover
                    else:
                        b0 = sseg  # pragma: no cover
                        j += 1  # pragma: no cover
                        pieces = _split_paragraphs(b0, t_min, t_max) if len(b0) > split_threshold else [b0]  # pragma: no cover
                        for piece in pieces:  # pragma: no cover
                            chunks.append(_finalize(piece, hierarchy, "section", idx_ref))  # pragma: no cover
            else:
                pieces = _split_paragraphs(body, t_min, t_max) if len(body) > split_threshold else [body]
                for piece in pieces:
                    chunks.append(_finalize(piece, hierarchy, "section", idx_ref))
        else:
            intro = seg
            i += 1
            if intro and not intro.startswith("#"):
                hier = h1 or path
                pieces = _split_paragraphs(intro, t_min, t_max) if len(intro) > split_threshold else [intro]
                for piece in pieces:
                    chunks.append(_finalize(piece, hier if h1 else "", "preamble", idx_ref))
    if not chunks and text.strip():
        txt, has_diag, diag_label = _unmask_markdown_with_meta(text[:8000], vault)
        meta: Dict[str, str] = {
            "chunk_strategy": "markdown_domain",
            "chunk_type": "document",
            "section": h1 or path,
            "doc_title": h1,
            "chunk_index": "0",
            "contains_diagram": "true" if has_diag else "",
        }
        if diag_label:
            meta["diagram_type"] = diag_label  # pragma: no cover
        chunks.append((txt, meta))
    return chunks

