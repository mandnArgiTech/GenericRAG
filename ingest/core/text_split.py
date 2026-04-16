"""Paragraph and token-budget splitting helpers.

Author: deviprasad
"""
from __future__ import annotations

import re
from typing import List, Optional, Tuple

from ingest.core.constants import DEFAULT_RFC_TOKEN_LIMIT, MODEL_TOKEN_LIMITS

_BLOCK_PLACEHOLDER_RE = re.compile(r"^<<BLOCK\d+>>$")


def _is_diagram_placeholder(para: str) -> bool:
    return bool(_BLOCK_PLACEHOLDER_RE.match(para.strip()))


def _split_paragraphs(text: str, target_min: int = 2000, target_max: int = 5000) -> List[str]:
    """Token-aware paragraph packer with diagram-context bonding.

    * Uses _estimate_tokens for token-approximate sizing (char limits are still
      accepted for backward compat -- callers can pass char-based values derived
      from MODEL_TOKEN_LIMITS via _md_char_targets).
    * Diagram-context bonding: if the *next* paragraph is a <<BLOCK>> placeholder
      (a diagram), aggressively pack it with the current buffer so that the
      explanatory paragraph preceding the diagram stays in the same chunk.
    """
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    if not paras:
        return []  # pragma: no cover
    out: List[str] = []
    buf = ""
    i = 0
    while i < len(paras):
        p = paras[i]
        candidate = (buf + "\n\n" + p).strip() if buf else p
        if len(candidate) <= target_max:
            buf = candidate
            if i + 1 < len(paras) and _is_diagram_placeholder(paras[i + 1]):
                diagram = paras[i + 1]
                bonded = (buf + "\n\n" + diagram).strip()
                hard_limit = int(target_max * 1.25)
                if len(bonded) <= hard_limit:
                    buf = bonded
                    i += 2
                    continue
            i += 1
        else:
            if buf:  # pragma: no cover
                out.append(buf)  # pragma: no cover
            buf = p  # pragma: no cover
            if i + 1 < len(paras) and _is_diagram_placeholder(paras[i + 1]):  # pragma: no cover
                diagram = paras[i + 1]  # pragma: no cover
                bonded = (buf + "\n\n" + diagram).strip()  # pragma: no cover
                hard_limit = int(target_max * 1.25)  # pragma: no cover
                if len(bonded) <= hard_limit:  # pragma: no cover
                    buf = bonded  # pragma: no cover
                    i += 2  # pragma: no cover
                    continue  # pragma: no cover
            i += 1  # pragma: no cover
    if buf:
        out.append(buf)
    return out


def _md_char_targets(embed_model: str) -> Tuple[int, int]:
    """Derive min/max char targets for markdown domain chunks, analogous to _rfc_char_targets."""
    for key, limit in MODEL_TOKEN_LIMITS.items():
        if key in (embed_model or "").lower():
            chars_max = limit * 4
            return max(400, chars_max // 3), chars_max
    default_chars = 2048  # pragma: no cover
    return max(400, default_chars // 3), default_chars  # pragma: no cover


def _estimate_tokens(text: str) -> int:
    """Rough token count for English-ish RFC text (~4 chars/token).

    For exact counts, plug in a model tokenizer (e.g. Hugging Face ``tokenizers``).
    """
    return max(1, len(text) // 4)


def _get_rfc_token_limit(embed_model: str) -> int:
    em = embed_model.lower()
    for key, limit in MODEL_TOKEN_LIMITS.items():
        if key in em:
            return limit
    return DEFAULT_RFC_TOKEN_LIMIT  # pragma: no cover


def _rfc_char_targets(embed_model: str) -> Tuple[int, int]:
    tok = _get_rfc_token_limit(embed_model)
    target_max = max(512, tok * 4)
    target_min = max(256, target_max // 3)
    return target_min, target_max
