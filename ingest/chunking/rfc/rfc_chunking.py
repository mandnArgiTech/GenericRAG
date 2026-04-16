"""RFC plaintext chunking orchestration (generic text ops + registry-driven parsers).

Author: deviprasad
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

from ingest.chunking.rfc.rfc_constants_parser import parse_all_constant_regions
from ingest.chunking.rfc.rfc_impl_rule_parser import parse_all_impl_regions
from ingest.chunking.rfc.rfc_profiles import RfcProfile, get_rfc_profile, merge_profile_chunk_metadata
from ingest.chunking.rfc.rfc_registry import discover_and_import, get_parsers_for_rfc, section_pattern_matches
from ingest.chunking.rfc.rfc_struct_parser import parse_all_struct_regions, strip_spans
from ingest.chunking.rfc.rfc_text_ops import (
    depaginate_rfc,
    is_rfc_file,
    rfc_heading_line_is_table_of_contents,
    rfc_line_is_diagram,
    shield_diagrams,
    sliding_window_chunks,
    unshield_diagrams,
)
from ingest.core.text_split import _rfc_char_targets, _split_paragraphs

# Back-compat for ``from ingest.chunking.rfc.rfc_chunking import _is_rfc_file``
_is_rfc_file = is_rfc_file


def _spurious_section_number(sec_num: str, title: str) -> bool:
    r"""Reject lines like ``100 useless entries...`` matched as fake ``^\d+`` headings."""
    tit = (title or "").strip()
    # Bit-ruler lines (e.g. ``0    1    2    3``) match ``^\d+`` and become a fake section ``0``.
    if sec_num == "0" and tit and re.fullmatch(r"\d+(?:\s+\d+)*", tit):
        return True
    if re.fullmatch(r"\d+", sec_num or ""):
        try:
            if int(sec_num) > 50:
                return True
        except ValueError:
            pass
    return False


def _finalize_rfc_chunk_meta(
    meta: Dict[str, str],
    *,
    profile: Optional[RfcProfile],
    section_number: str,
    section_title: str,
) -> Dict[str, str]:
    return merge_profile_chunk_metadata(meta, profile, section_number=section_number, section_title=section_title)


def _contains_diagram_token(piece: str) -> str:
    return "true" if ("+-" in piece or "-+" in piece or "__DIAGRAM_" in piece) else "false"


def _meta_scalar(v: object) -> str:
    """Flatten metadata values for Chroma-style string dicts."""
    if isinstance(v, bool):
        return "true" if v else "false"
    return str(v)


def chunk_rfc(
    text: str, path: str, embed_model: str = "nomic-embed-text"
) -> List[Tuple[str, Dict[str, str]]]:
    # Section boundaries must be computed on depaginated *unshielded* text so indices match
    # the ASCII transition table (otherwise shield_diagrams collapses the matrix).
    text_raw = depaginate_rfc(text)
    if not text_raw.strip():
        return []
    t_min, t_max = _rfc_char_targets(embed_model)
    section_body_threshold = t_max * 2

    sec_start = None
    for _m in re.finditer(r"(?m)^(\d+(?:\.\d+)*)\.?\s+([^\n]+)", text_raw):
        if not rfc_heading_line_is_table_of_contents(_m.group(2)) and not _spurious_section_number(
            _m.group(1), _m.group(2)
        ):
            sec_start = _m
            break
    if sec_start and sec_start.start() > 0:
        trimmed = text_raw[sec_start.start() :]
        if len(trimmed.strip()) > 200:
            text_raw = trimmed

    rfc_no = ""
    m = re.search(r"RFC\s*(\d+)", path, re.I) or re.search(r"RFC\s*(\d+)", text_raw[:2000], re.I)
    if m:
        rfc_no = m.group(1)
    profile = get_rfc_profile(rfc_no)

    title = ""
    for ln in text_raw.splitlines()[:40]:
        if ln.strip() and not ln.strip().lower().startswith("request for comments"):
            title = ln.strip()
            if "Network Working Group" in title:
                continue
            break

    diagram_pred = lambda line, _p=profile: rfc_line_is_diagram(line, _p)

    sections = [
        m
        for m in re.finditer(r"(?m)^(\d+(?:\.\d+)*)\.?\s+([^\n]+)", text_raw)
        if not rfc_heading_line_is_table_of_contents(m.group(2))
        and not _spurious_section_number(m.group(1), m.group(2))
    ]
    out: List[Tuple[str, Dict[str, str]]] = []

    if not sections:
        body_work = text_raw
        sec_num = ""
        sec_title = ""
        for parser in get_parsers_for_rfc(rfc_no):
            if not section_pattern_matches(parser.section_pattern, sec_num):
                continue
            for sent, fsm_partial in parser.parse_fn(body_work, sec_num):
                row_meta: Dict[str, str] = {
                    "chunk_strategy": "rfc",
                    "rfc_number": rfc_no,
                    "rfc_title": title,
                    "section_number": sec_num,
                    "section_title": sec_title,
                    "chunk_index": str(len(out)),
                    "contains_diagram": "false",
                }
                for k, v in fsm_partial.items():
                    if v is None or v == "":
                        continue
                    row_meta[k] = _meta_scalar(v)
                row_meta.setdefault("chunk_type", "fsm_transition_rule")
                row_meta = _finalize_rfc_chunk_meta(
                    row_meta, profile=profile, section_number=sec_num, section_title=sec_title
                )
                out.append((sent, row_meta))
            body_work = parser.strip_fn(body_work, sec_num)

        struct_chunks, struct_spans = parse_all_struct_regions(body_work)
        body1 = strip_spans(body_work, struct_spans)
        const_chunks, const_spans = parse_all_constant_regions(body1)
        body2 = strip_spans(body1, const_spans)
        impl_chunks, impl_spans = parse_all_impl_regions(body2)
        body3 = strip_spans(body2, impl_spans)

        for ch, partial in struct_chunks + const_chunks + impl_chunks:
            row_meta = {
                "chunk_strategy": "rfc",
                "chunk_type": partial.get("chunk_type", "struct_definition"),
                "rfc_number": rfc_no,
                "rfc_title": title,
                "section_number": "",
                "section_title": "",
                "chunk_index": str(len(out)),
                "contains_diagram": _contains_diagram_token(ch),
            }
            hf = partial.get("hierarchical_field", "")
            if hf:
                row_meta["hierarchical_field"] = str(hf)
            row_meta = _finalize_rfc_chunk_meta(
                row_meta, profile=profile, section_number="", section_title=""
            )
            out.append((ch, row_meta))

        text_s, diagram_vault = shield_diagrams(body3, diagram_pred)
        base = {
            "chunk_strategy": "rfc",
            "rfc_number": rfc_no,
            "rfc_title": title,
            "section_number": "",
            "section_title": "",
        }

        def _meta_contains_diagram(piece: str) -> str:
            return "true" if any(k in piece for k in diagram_vault) else "false"

        for win_text, partial in sliding_window_chunks(text_s, t_max, 0.15, base):
            raw_piece = win_text
            final_t = unshield_diagrams(raw_piece, diagram_vault)
            meta = {
                **partial,
                "chunk_strategy": "rfc",
                "chunk_type": "prose",
                "rfc_number": rfc_no,
                "rfc_title": title,
                "section_number": "",
                "section_title": "",
                "contains_diagram": _meta_contains_diagram(raw_piece),
            }
            meta = _finalize_rfc_chunk_meta(
                meta, profile=profile, section_number="", section_title=""
            )
            out.append((final_t, meta))
        return out

    for i, msec in enumerate(sections):
        start = msec.start()
        end = sections[i + 1].start() if i + 1 < len(sections) else len(text_raw)
        sec_num = msec.group(1)
        sec_title = msec.group(2).strip()
        body_raw = text_raw[start:end].strip()

        # Each plugin sees text after prior plugins stripped (avoids duplicate chunks).
        body_work = body_raw
        for parser in get_parsers_for_rfc(rfc_no):
            if not section_pattern_matches(parser.section_pattern, sec_num):
                continue
            for sent, fsm_partial in parser.parse_fn(body_work, sec_num):
                row_meta: Dict[str, str] = {
                    "chunk_strategy": "rfc",
                    "rfc_number": rfc_no,
                    "rfc_title": title,
                    "section_number": sec_num,
                    "section_title": sec_title,
                    "chunk_index": str(len(out)),
                    "contains_diagram": "false",
                }
                for k, v in fsm_partial.items():
                    if v is None or v == "":
                        continue
                    row_meta[k] = _meta_scalar(v)
                row_meta.setdefault("chunk_type", "fsm_transition_rule")
                row_meta = _finalize_rfc_chunk_meta(
                    row_meta, profile=profile, section_number=sec_num, section_title=sec_title
                )
                out.append((sent, row_meta))
            body_work = parser.strip_fn(body_work, sec_num)
        body_for_split = body_work

        struct_chunks, struct_spans = parse_all_struct_regions(body_for_split)
        body_a = strip_spans(body_for_split, struct_spans)
        for ch, partial in struct_chunks:
            row_meta = {
                "chunk_strategy": "rfc",
                "chunk_type": partial.get("chunk_type", "struct_definition"),
                "rfc_number": rfc_no,
                "rfc_title": title,
                "section_number": sec_num,
                "section_title": sec_title,
                "chunk_index": str(len(out)),
                "contains_diagram": _contains_diagram_token(ch),
            }
            hf = partial.get("hierarchical_field", "")
            if hf:
                row_meta["hierarchical_field"] = str(hf)
            row_meta = _finalize_rfc_chunk_meta(
                row_meta, profile=profile, section_number=sec_num, section_title=sec_title
            )
            out.append((ch, row_meta))

        const_chunks, const_spans = parse_all_constant_regions(body_a)
        body_b = strip_spans(body_a, const_spans)
        for ch, partial in const_chunks:
            row_meta = {
                "chunk_strategy": "rfc",
                "chunk_type": partial.get("chunk_type", "constants"),
                "rfc_number": rfc_no,
                "rfc_title": title,
                "section_number": sec_num,
                "section_title": sec_title,
                "chunk_index": str(len(out)),
                "contains_diagram": _contains_diagram_token(ch),
            }
            row_meta = _finalize_rfc_chunk_meta(
                row_meta, profile=profile, section_number=sec_num, section_title=sec_title
            )
            out.append((ch, row_meta))

        impl_chunks, impl_spans = parse_all_impl_regions(body_b)
        body_c = strip_spans(body_b, impl_spans)
        for ch, partial in impl_chunks:
            row_meta = {
                "chunk_strategy": "rfc",
                "chunk_type": partial.get("chunk_type", "implementation_rule"),
                "rfc_number": rfc_no,
                "rfc_title": title,
                "section_number": sec_num,
                "section_title": sec_title,
                "chunk_index": str(len(out)),
                "contains_diagram": "false",
            }
            row_meta = _finalize_rfc_chunk_meta(
                row_meta, profile=profile, section_number=sec_num, section_title=sec_title
            )
            out.append((ch, row_meta))

        body_s, sec_vault = shield_diagrams(body_c, diagram_pred)

        def _meta_contains_diagram_local(piece: str) -> str:
            return "true" if any(k in piece for k in sec_vault) else "false"

        parts = (
            _split_paragraphs(body_s, t_min, t_max) if len(body_s) > section_body_threshold else [body_s]
        )
        for p in parts:
            raw_piece = p
            final_t = unshield_diagrams(raw_piece, sec_vault)
            if not final_t.strip():
                continue
            meta = {
                "chunk_strategy": "rfc",
                "chunk_type": "prose",
                "rfc_number": rfc_no,
                "rfc_title": title,
                "section_number": sec_num,
                "section_title": sec_title,
                "chunk_index": str(len(out)),
                "contains_diagram": _meta_contains_diagram_local(raw_piece),
            }
            meta = _finalize_rfc_chunk_meta(
                meta, profile=profile, section_number=sec_num, section_title=sec_title
            )
            out.append((final_t, meta))
    return out


discover_and_import()
