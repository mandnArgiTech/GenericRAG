"""Process-wide threading primitives for ingest.

Author: deviprasad
"""
from __future__ import annotations

import threading

shutdown_event = threading.Event()
_embed_lock = threading.Lock()
