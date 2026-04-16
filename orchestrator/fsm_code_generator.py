#!/usr/bin/env python3
"""
Chroma + Ollama PPP FSM **code generator** (C case synthesis from RAG chunks).

Formerly ``fsm_orchestrator.py``; CLI entry: ``python -m orchestrator.fsm_code_generator``.

Author: deviprasad
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import chromadb
import requests
from jinja2 import Environment, FileSystemLoader

try:
    from hybrid_search import reciprocal_rank_fusion

    _RRF_AVAILABLE = True
except ImportError:  # pragma: no cover
    reciprocal_rank_fusion = None  # type: ignore[misc, assignment]
    _RRF_AVAILABLE = False

log = logging.getLogger("fsm_code_generator")

# Jinja keys align with typedef enum names in ppp_fsm.c.jinja
STATE_ORDER: List[str] = [
    "PPP_STATE_INITIAL",
    "PPP_STATE_STARTING",
    "PPP_STATE_CLOSED",
    "PPP_STATE_STOPPED",
    "PPP_STATE_CLOSING",
    "PPP_STATE_STOPPING",
    "PPP_STATE_REQ_SENT",
    "PPP_STATE_ACK_RCVD",
    "PPP_STATE_ACK_SENT",
    "PPP_STATE_OPENED",
]

@dataclass(frozen=True)
class OrchestratorConfig:
    chroma_path: str
    collection_name: str
    ollama_host: str
    extractor_model: str
    generator_model: str
    embed_model: str
    http_timeout_s: float = 600.0


def _ollama_chat(
    host: str,
    model: str,
    system: str,
    user: str,
    *,
    temperature: float = 0.0,
    timeout: float = 600.0,
) -> str:
    url = f"{host.rstrip('/')}/api/chat"
    payload: Dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "stream": False,
        "options": {"temperature": temperature},
    }
    resp = requests.post(url, json=payload, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    msg = data.get("message") or {}
    content = msg.get("content")
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError(f"Ollama returned no message content: {data!r}")
    return content.strip()


def _ollama_embed(host: str, model: str, text: str, *, timeout: float = 120.0) -> List[float]:
    url = f"{host.rstrip('/')}/api/embeddings"
    resp = requests.post(
        url,
        json={"model": model, "prompt": text},
        timeout=timeout,
    )
    resp.raise_for_status()
    data = resp.json()
    emb = data.get("embedding")
    if not isinstance(emb, list):
        raise RuntimeError(f"Ollama embeddings missing 'embedding': {data!r}")
    return [float(x) for x in emb]


def _strip_code_fences(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", t)
        t = re.sub(r"\s*```$", "", t)
    return t.strip()


def _json_from_extractor_response(text: str) -> Dict[str, Any]:
    raw = _strip_code_fences(text)
    return json.loads(raw)


def retrieve_fsm_chunks_hybrid(
    collection: chromadb.Collection,
    *,
    fsm_state: str,
    embed_fn: Callable[[str], List[float]],
    dense_n: int = 24,
    metadata_cap: int = 512,
) -> List[str]:
    """
    Hybrid retrieval: dense vector query restricted by metadata, fused with a full
    metadata scan of the same filter via reciprocal rank fusion when available.
    """
    where: Dict[str, Any] = {
        "$and": [
            {"chunk_type": "fsm_transition_rule"},
            {"fsm_state": fsm_state},
        ]
    }
    q = (
        f"PPP LCP RFC 1661 finite-state machine transition rules for state {fsm_state}. "
        "Events, actions (tlu, scr, irc, …), and next states."
    )
    emb = embed_fn(q)
    dq = collection.query(
        query_embeddings=[emb],
        n_results=max(1, dense_n),
        where=where,
        include=["documents", "ids"],
    )
    dense_ids: List[str] = list(dq.get("ids") or [[]])[0]
    dense_docs: List[str] = list(dq.get("documents") or [[]])[0]

    got = collection.get(where=where, limit=max(1, metadata_cap), include=["documents", "ids"])
    meta_ids: List[str] = list(got.get("ids") or [])
    meta_docs: List[str] = list(got.get("documents") or [])

    if _RRF_AVAILABLE and reciprocal_rank_fusion is not None and (dense_ids or meta_ids):
        rrf = reciprocal_rank_fusion([dense_ids, meta_ids], k=60.0)
        ordered_ids = [i for i, _ in sorted(rrf.items(), key=lambda kv: kv[1], reverse=True)]
        id_to_doc: Dict[str, str] = {}
        for i, d in zip(dense_ids, dense_docs):
            id_to_doc[i] = d
        for i, d in zip(meta_ids, meta_docs):
            id_to_doc.setdefault(i, d)
        return [id_to_doc[i] for i in ordered_ids if i in id_to_doc]

    # Fallback: dense order, then append unseen metadata hits
    out: List[str] = []
    seen: set[str] = set()
    for d in dense_docs:
        if d and d not in seen:
            seen.add(d)
            out.append(d)
    for d in meta_docs:
        if d and d not in seen:
            seen.add(d)
            out.append(d)
    return out


EXTRACTOR_SYSTEM_PROMPT = (
    "You are a strict information extractor for RFC 1661 PPP LCP transition rules. "
    "Output ONLY a single JSON object, no markdown, no commentary. "
    "Schema: {\"current_state\": str, \"event\": str, \"action\": str, \"next_state\": str}. "
    "Use uppercase underscore state names (e.g. REQ_SENT, ACK_RCVD). "
    "For event use symbols like Up, Down, Open, Close, TO+, TO-, RCR+, RCR-, RCA, RCN, "
    "RTR, RTA, RUC, RXJ+, RXJ-, RXR. "
    "action must be a comma-separated list of RFC action abbreviations (e.g. irc,scr) or "
    "the empty string if none. "
    "next_state must be an uppercase underscore state name or ILLEGAL if the rule says the "
    "transition is illegal."
)


def extract_json_rule(
    chunk_text: str,
    cfg: OrchestratorConfig,
    *,
    max_retries: int = 3,
) -> Dict[str, str]:
    required = {"current_state", "event", "action", "next_state"}
    last_err: Optional[Exception] = None
    for attempt in range(1, max_retries + 1):
        user = (
            f"Transition rule text:\n{chunk_text}\n\n"
            "Return one JSON object matching the schema."
        )
        try:
            raw = _ollama_chat(
                cfg.ollama_host,
                cfg.extractor_model,
                EXTRACTOR_SYSTEM_PROMPT,
                user,
                temperature=0.0,
                timeout=cfg.http_timeout_s,
            )
            obj = _json_from_extractor_response(raw)
            if not isinstance(obj, dict):
                raise ValueError("JSON root must be an object")
            missing = required - obj.keys()
            if missing:
                raise ValueError(f"missing keys: {sorted(missing)}")
            out = {k: str(obj[k]).strip() for k in sorted(required)}
            if not out["current_state"] or not out["event"]:
                raise ValueError("current_state and event must be non-empty")
            return out
        except (json.JSONDecodeError, ValueError, requests.RequestException, RuntimeError) as exc:
            last_err = exc
            log.warning("extract_json_rule attempt %s/%s failed: %s", attempt, max_retries, exc)
    raise RuntimeError(f"extract_json_rule failed after {max_retries} attempts") from last_err


GENERATOR_CASE_SYSTEM_PROMPT = (
    "You are an embedded C engineer. Output ONLY bare-metal C code: one or more "
    "`case PPP_EVENT_*:` arms suitable for nesting inside "
    "`switch (event) { ... }` within `void ppp_handle_event(ppp_context_t *ctx, ppp_event_t event)`. "
    "Assume strict C11, `stdint.h` / `stdbool.h` already included. "
    "Use `ctx->state`, `ctx->restart_count`, etc. when needed. "
    "Each case must end with `break;`. "
    "Always assign `ctx->state` to the JSON `next_state` as the corresponding `PPP_STATE_*` "
    "enumerator after performing JSON `action` abbreviations as no-op comments if you "
    "have no helper functions yet. "
    "Map JSON event strings to enum constants: Up→PPP_EVENT_UP, Down→PPP_EVENT_DOWN, "
    "Open→PPP_EVENT_OPEN, Close→PPP_EVENT_CLOSE, TO+→PPP_EVENT_TO_PLUS, TO-→PPP_EVENT_TO_MINUS, "
    "RCR+→PPP_EVENT_RCR_PLUS, RCR-→PPP_EVENT_RCR_MINUS, RCA→PPP_EVENT_RCA, RCN→PPP_EVENT_RCN, "
    "RTR→PPP_EVENT_RTR, RTA→PPP_EVENT_RTA, RUC→PPP_EVENT_RUC, "
    "RXJ+→PPP_EVENT_RXJ_PLUS, RXJ-→PPP_EVENT_RXJ_MINUS, RXR→PPP_EVENT_RXR. "
    "Do not emit comments that restate the JSON. Indent with 16 spaces at the case label line."
)


def generate_c_case(
    json_rule: Dict[str, str],
    cfg: OrchestratorConfig,
) -> str:
    user = json.dumps(json_rule, ensure_ascii=False, sort_keys=True)
    out = _ollama_chat(
        cfg.ollama_host,
        cfg.generator_model,
        GENERATOR_CASE_SYSTEM_PROMPT,
        user,
        temperature=0.0,
        timeout=cfg.http_timeout_s,
    )
    return _strip_code_fences(out).strip() + "\n"


GENERATOR_FIX_SYSTEM_PROMPT = (
    "You are an embedded C engineer. This C code failed to compile with gcc -Wall -Wextra. "
    "Output ONLY the complete corrected translation unit (full .c file beginning with "
    "#include lines). No markdown fences, no explanation."
)


def _compile_c_file(c_path: Path, *, extra_flags: Sequence[str] = ()) -> Tuple[int, str, str]:
    cmd = ["gcc", "-c", str(c_path), "-Wall", "-Wextra", *extra_flags]
    proc = subprocess.run(
        cmd,
        cwd=str(c_path.parent),
        text=True,
        capture_output=True,
        check=False,
    )
    return proc.returncode, proc.stdout or "", proc.stderr or ""


def _repair_c_source_with_generator(
    failing_source: str,
    gcc_stderr: str,
    cfg: OrchestratorConfig,
    *,
    max_retries: int = 3,
) -> str:
    last_err: Optional[Exception] = None
    for attempt in range(1, max_retries + 1):
        user = (
            "This C code failed to compile. Here is GCC stderr:\n\n"
            f"{gcc_stderr}\n\n"
            "Here is the full failing source:\n\n"
            f"{failing_source}\n"
        )
        try:
            raw = _ollama_chat(
                cfg.ollama_host,
                cfg.generator_model,
                GENERATOR_FIX_SYSTEM_PROMPT,
                user,
                temperature=0.0,
                timeout=cfg.http_timeout_s,
            )
            fixed = _strip_code_fences(raw).strip() + "\n"
            if "#include" not in fixed or "ppp_handle_event" not in fixed:
                raise ValueError("model output missing expected C structure")
            return fixed
        except (ValueError, requests.RequestException, RuntimeError) as exc:
            last_err = exc
            log.warning("gcc repair attempt %s/%s failed: %s", attempt, max_retries, exc)
    raise RuntimeError(f"could not repair C after {max_retries} attempts") from last_err


def render_ppp_fsm_c(
    template_dir: Path,
    state_case_blocks: Dict[str, str],
    out_path: Path,
) -> None:
    env = Environment(
        loader=FileSystemLoader(str(template_dir)),
        autoescape=False,
        trim_blocks=True,
        lstrip_blocks=True,
    )
    tpl = env.get_template("ppp_fsm.c.jinja")
    text = tpl.render(state_order=STATE_ORDER, state_case_blocks=state_case_blocks)
    out_path.write_text(text, encoding="utf-8")


def run_pipeline(cfg: OrchestratorConfig, out_c: Path, template_path: Path) -> int:
    client = chromadb.PersistentClient(path=os.path.abspath(cfg.chroma_path))
    collection = client.get_collection(name=cfg.collection_name)

    def embed_fn(t: str) -> List[float]:
        return _ollama_embed(cfg.ollama_host, cfg.embed_model, t, timeout=120.0)

    state_case_blocks: Dict[str, str] = {k: "" for k in STATE_ORDER}

    for state_key in STATE_ORDER:
        fsm_meta_state = state_key.replace("PPP_STATE_", "", 1)
        chunks = retrieve_fsm_chunks_hybrid(
            collection,
            fsm_state=fsm_meta_state,
            embed_fn=embed_fn,
        )
        parts: List[str] = []
        for ch in chunks:
            try:
                rule = extract_json_rule(ch, cfg)
            except RuntimeError as exc:
                log.error("skipping chunk after extraction failure: %s", exc)
                continue
            if rule.get("next_state") == "ILLEGAL":
                continue
            try:
                parts.append(generate_c_case(rule, cfg))
            except (requests.RequestException, RuntimeError) as exc:
                log.error("generator failed for rule %s: %s", rule, exc)
        state_case_blocks[state_key] = "\n".join(parts).strip() + ("\n" if parts else "")

    render_ppp_fsm_c(template_path.parent, state_case_blocks, out_c)

    src = out_c.read_text(encoding="utf-8")
    for round_i in range(3):
        code, _out, err = _compile_c_file(out_c)
        if code == 0:
            log.info("gcc OK (%s)", out_c)
            return 0
        log.warning("gcc failed (validation round %s/3); invoking generator repair", round_i + 1)
        try:
            src = _repair_c_source_with_generator(src, err, cfg, max_retries=3)
            out_c.write_text(src, encoding="utf-8")
        except RuntimeError as exc:
            log.error("repair failed: %s", exc)
            return 1
    code, _out, err = _compile_c_file(out_c)
    if code != 0:
        log.error("gcc still failing after 3 repair rounds:\n%s", err)
    return code


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--chroma-path", default=os.environ.get("CHROMA_PATH", "./chroma_db"))
    p.add_argument("--collection", default=os.environ.get("CHROMA_COLLECTION", "default"))
    p.add_argument("--ollama-host", default=os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434"))
    p.add_argument("--extractor-model", default=os.environ.get("FSM_EXTRACT_MODEL", "qwen2.5:1.5b"))
    p.add_argument("--generator-model", default=os.environ.get("FSM_GENERATE_MODEL", "qwen2.5:14b"))
    p.add_argument("--embed-model", default=os.environ.get("OLLAMA_EMBED_MODEL", "nomic-embed-text"))
    p.add_argument(
        "--out-c",
        type=Path,
        default=Path(os.environ.get("PPP_FSM_OUT", "ppp_fsm.c")),
        help="Output C file path",
    )
    p.add_argument(
        "--template-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent,
        help="Directory containing ppp_fsm.c.jinja (repo root by default)",
    )
    p.add_argument("--timeout", type=float, default=float(os.environ.get("FSM_HTTP_TIMEOUT", "600")))
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_arg_parser().parse_args(list(argv) if argv is not None else None)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    cfg = OrchestratorConfig(
        chroma_path=args.chroma_path,
        collection_name=args.collection,
        ollama_host=args.ollama_host,
        extractor_model=args.extractor_model,
        generator_model=args.generator_model,
        embed_model=args.embed_model,
        http_timeout_s=args.timeout,
    )
    return run_pipeline(cfg, args.out_c.resolve(), args.template_dir.resolve())


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
