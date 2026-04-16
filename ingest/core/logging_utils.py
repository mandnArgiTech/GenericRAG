"""Logging setup for ingestion.

Author: deviprasad
"""
from __future__ import annotations

import logging

logger = logging.getLogger("ingest")

def setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
