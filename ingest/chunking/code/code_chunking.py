"""Source-code chunk strategies; implementation is split across ``code_*`` modules.

Author: deviprasad
"""
from __future__ import annotations

from ingest.chunking.code.code_dependencies import extract_dependencies
from ingest.chunking.code.code_strategy import choose_strategy_for_path

__all__ = ["choose_strategy_for_path", "extract_dependencies"]
