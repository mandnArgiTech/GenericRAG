"""Load Rally-exported rows from CSV.

Author: deviprasad
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, List

def load_rally_rows_from_csv(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, newline="", encoding="utf-8", errors="replace") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            rows.append({(k or "").strip(): (v if v is not None else "") for k, v in row.items()})
    return rows

