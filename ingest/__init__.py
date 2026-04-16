"""Domain RAG ingestion package (chunking, embeddings, Chroma upsert, CLI).

Layout: ``ingest.core`` (config, utils), ``ingest.sources`` (Git, Rally, remote APIs),
``ingest.chunking`` (per-format chunkers), ``ingest.runtime`` (pipeline, embeddings job).
Public symbols are re-exported here; legacy flat imports remain via ``ingest.constants``,
``ingest.code_chunking``, and ``ingest.rfc_chunking`` shims.

Author: deviprasad
"""
from __future__ import annotations

from ingest.chunking.code.code_chunking import choose_strategy_for_path
from ingest.core.constants import INGESTION_VERSION, SCRIPT_DIR
from ingest.runtime.ingest_job import ingest_run
from ingest.core.metadata_utils import iter_concept_ids
from ingest.runtime.pipeline import build_arg_parser, feed_domain_document, main
from ingest.chunking.rfc.rfc_chunking import chunk_rfc

__all__ = [
    "INGESTION_VERSION",
    "SCRIPT_DIR",
    "build_arg_parser",
    "choose_strategy_for_path",
    "chunk_rfc",
    "feed_domain_document",
    "ingest_run",
    "iter_concept_ids",
    "main",
]
