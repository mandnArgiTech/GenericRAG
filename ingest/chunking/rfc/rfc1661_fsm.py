"""RFC 1661 §4.1 LCP state transition table — structured chunks only (not generic RFC logic).

Author: deviprasad
"""
from __future__ import annotations

import re
from typing import Dict, List, Tuple

from ingest.chunking.rfc.rfc_registry import RfcSectionParser, register_rfc_parser
from ingest.chunking.rfc.rfc_text_ops import rfc_heading_line_is_table_of_contents

__all__ = (
    "parse_rfc1661_section41_fsm_chunks",
    "rfc1661_section41_strip_ascii_tables",
    "rfc_heading_line_is_table_of_contents",
    "rfc1661_lcp_state_num_to_key",
    "rfc1661_fsm_state_header_to_key",
    "rfc1661_fsm_cell_to_sentence",
)


def rfc1661_lcp_state_num_to_key(num: str) -> str:
    """Map RFC 1661 Section 4.1 numeric state index to a stable uppercase key."""
    table = {
        "0": "INITIAL",
        "1": "STARTING",
        "2": "CLOSED",
        "3": "STOPPED",
        "4": "CLOSING",
        "5": "STOPPING",
        "6": "REQ_SENT",
        "7": "ACK_RCVD",
        "8": "ACK_SENT",
        "9": "OPENED",
    }
    return table.get(num.strip(), f"STATE_{num.strip()}")


def rfc1661_fsm_state_header_to_key(name: str) -> str:
    """Normalize a column header like 'Req-Sent' or 'Initial' to metadata key form."""
    t = name.strip()
    if not t:
        return ""
    return re.sub(r"[^A-Za-z0-9]+", "_", t).strip("_").upper()


def rfc1661_section41_strip_ascii_tables(section_body: str) -> str:
    """
    Remove the monospaced transition-matrix block(s) from RFC 1661 Section 4.1 so
    paragraph chunking does not duplicate the structured FSM chunks.
    """
    end_anchor = "The states in which the Restart timer"
    j = section_body.find(end_anchor)
    if j < 0:
        return section_body
    m = re.search(r"(?m)^\|\s+State\s*$", section_body)
    if not m or m.start() >= j:
        return section_body
    start = m.start()
    return (section_body[:start].rstrip() + "\n\n" + section_body[j:].lstrip()).strip()


def rfc1661_fsm_cell_to_sentence(
    *,
    state_key: str,
    state_header: str,
    event_raw: str,
    event_key: str,
    cell: str,
) -> Tuple[str, Dict[str, str]]:
    c = cell.strip()
    if not c or c == "-":
        sent = (
            f"State Transition Rule: In state {state_key}, event {event_raw} is an illegal "
            "transition (marked '-') per RFC 1661 Section 4.1 State Transition Table."
        )
        return sent, {"fsm_illegal": "true"}
    footnote = ""
    actions: List[str] = []
    next_num = ""
    slash = c.rfind("/")
    if slash >= 0:
        left, right = c[:slash], c[slash + 1 :]
        rm = re.match(r"^(\d+)([a-z])?$", right.strip())
        if not rm:
            sent = (
                f"State Transition Rule: In state {state_key}, event {event_raw} has cell {c!r} "
                "in RFC 1661 Section 4.1 (unparsed)."
            )
            return sent, {"fsm_parse_error": "true"}
        next_num, fn = rm.group(1), rm.group(2) or ""
        footnote = fn
        actions = [a.strip() for a in left.split(",") if a.strip()]
    else:
        m = re.match(r"^(\d+)([a-z])?$", c)
        if m:
            next_num, fn = m.group(1), m.group(2) or ""
            footnote = fn
        else:
            sent = (
                f"State Transition Rule: In state {state_key}, event {event_raw} has cell {c!r} "
                "in RFC 1661 Section 4.1 (unparsed)."
            )
            return sent, {"fsm_parse_error": "true"}

    next_key = rfc1661_lcp_state_num_to_key(next_num)
    if actions:
        act_txt = ", ".join(actions)
        sent = (
            f"State Transition Rule: In state {state_key}, event {event_raw} triggers "
            f"action(s) {act_txt} and transitions to state {next_key}."
        )
    else:
        sent = (
            f"State Transition Rule: In state {state_key}, event {event_raw} triggers "
            f"a transition to state {next_key} (no explicit action listed in the table cell)."
        )
    if footnote:
        sent += f" (RFC table footnote suffix: {footnote})"
    return sent, {}


def parse_rfc1661_section41_fsm_chunks(section_body: str) -> List[Tuple[str, Dict[str, str]]]:
    """
    Parse RFC 1661 Section 4.1 "State Transition Table" ASCII matrix (events on rows,
    states on columns). Produces one English sentence per legal cell; illegal cells (-)
    are also emitted for completeness.
    """
    lines = section_body.splitlines()
    out: List[Tuple[str, Dict[str, str]]] = []
    i = 0
    n = len(lines)
    while i < n:
        ln = lines[i]
        if not ln.strip().startswith("Events|"):
            i += 1
            continue
        _, rhs = ln.split("|", 1)
        state_headers = rhs.split()
        if not state_headers:
            i += 1
            continue
        ncol = len(state_headers)
        state_keys = [rfc1661_fsm_state_header_to_key(h) for h in state_headers]
        i += 1
        while i < n:
            s = lines[i].strip()
            if not s:
                break
            if re.match(r"^\|(?:\s+\d+)+$", s):
                i += 1
                continue
            if re.match(r"^[-+|]+$", re.sub(r"[^-+|\s]", "", s)) and "+" in s and "-" in s:
                i += 1
                continue
            break
        while i < n:
            row = lines[i]
            st = row.strip()
            if not st:
                break
            if st.startswith("| State"):
                break
            if st.startswith("Events|"):
                break
            if "|" not in row:
                i += 1
                continue
            if re.match(r"^[-+|]+$", re.sub(r"[^-+|\s]", "", st)) and "+" in st:
                i += 1
                continue
            left, right = row.split("|", 1)
            event_raw = left.strip()
            if event_raw.lower() == "events" or not event_raw:
                i += 1
                continue
            event_key = re.sub(r"[^A-Za-z0-9]+", "_", event_raw).strip("_").upper()
            cells = right.strip().split()
            if len(cells) != ncol:
                if len(cells) < ncol:
                    cells.extend(["-"] * (ncol - len(cells)))
                else:
                    cells = cells[:ncol]
            for col_idx, cell in enumerate(cells):
                state_key = state_keys[col_idx]
                sentence, extra = rfc1661_fsm_cell_to_sentence(
                    state_key=state_key,
                    state_header=state_headers[col_idx],
                    event_raw=event_raw,
                    event_key=event_key,
                    cell=cell,
                )
                meta: Dict[str, str] = {
                    "chunk_type": "fsm_transition_rule",
                    "fsm_state": state_key,
                    "fsm_event": event_key,
                    "fsm_state_header": state_headers[col_idx],
                    "fsm_event_raw": event_raw,
                }
                meta.update({k: v for k, v in extra.items() if v})
                out.append((sentence, meta))
            i += 1
    return out


def _parse_1661_41(body: str, sec: str) -> List[Tuple[str, Dict[str, str]]]:
    return parse_rfc1661_section41_fsm_chunks(body)


def _strip_1661_41(body: str, sec: str) -> str:
    return rfc1661_section41_strip_ascii_tables(body)


register_rfc_parser(
    RfcSectionParser(
        rfc_number="1661",
        section_pattern=r"^4\.1$",
        parse_fn=_parse_1661_41,
        strip_fn=_strip_1661_41,
        priority=10,
    )
)
