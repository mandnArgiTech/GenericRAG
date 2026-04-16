#!/usr/bin/env python3
"""
Backward-compatible CLI entry for the PPP FSM orchestrator (delegates to ``orchestrator``).

Author: deviprasad
"""
from __future__ import annotations

import sys

from orchestrator.fsm_code_generator import main

if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
