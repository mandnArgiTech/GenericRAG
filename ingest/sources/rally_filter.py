"""Rally / CSV row filter parsing and matching.

Author: deviprasad
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

def parse_rally_filter(filter_str: Optional[str]) -> Dict[str, Any]:
    """Parse --rally-filter e.g. 'severity=1,2,3 state=Closed'."""
    if not filter_str or not str(filter_str).strip():
        return {}
    rules: Dict[str, Any] = {}
    for part in re.split(r"\s+", str(filter_str).strip()):
        if "=" not in part:
            continue
        k, v = part.split("=", 1)
        key = k.strip().lower()
        val = v.strip()
        if key == "severity":
            rules["severity"] = {x.strip() for x in val.split(",") if x.strip()}
        elif key == "state":
            rules["state"] = val
        elif key == "priority":
            rules["priority"] = val  # pragma: no cover
        else:
            rules[key] = val
    return rules


def rally_matches_user_filter(obj: Dict[str, Any], rules: Dict[str, Any]) -> bool:
    if not rules:
        return True
    if "severity" in rules:
        sev = str(obj.get("Severity") or obj.get("severity") or "").strip()
        if sev and sev not in rules["severity"]:
            return False
    if "state" in rules:
        st = str(obj.get("State") or obj.get("state") or "")
        want = str(rules["state"])
        if want and st.lower() != want.lower():
            return False
    if "priority" in rules:
        pr = str(obj.get("Priority") or obj.get("priority") or "")  # pragma: no cover
        want = str(rules["priority"])  # pragma: no cover
        if want and want.lower() not in pr.lower():  # pragma: no cover
            return False  # pragma: no cover
    return True


_BARE_MERMAID_STARTERS = re.compile(
    r"^(?:graph\s+(?:TD|TB|BT|RL|LR)|sequenceDiagram|classDiagram|stateDiagram"
    r"|erDiagram|gantt|pie|flowchart|journey|gitGraph|mindmap|timeline|quadrantChart"
    r"|sankey|xychart|block-beta|packet-beta|kanban|architecture-beta)\b",
    re.MULTILINE,
)

_DIAGRAM_TYPE_HINTS = {
    "graph": "Mermaid Flowchart",
    "flowchart": "Mermaid Flowchart",
    "sequencediagram": "Mermaid Sequence Diagram",
    "classdiagram": "Mermaid Class Diagram",
    "statediagram": "Mermaid State Diagram",
    "erdiagram": "Mermaid ER Diagram",
    "gantt": "Mermaid Gantt Chart",
    "pie": "Mermaid Pie Chart",
    "journey": "Mermaid User Journey",
    "gitgraph": "Mermaid Git Graph",
    "mindmap": "Mermaid Mind Map",
    "timeline": "Mermaid Timeline",
    "sankey": "Mermaid Sankey Diagram",
    "xychart": "Mermaid XY Chart",
    "@startuml": "PlantUML Diagram",
    "@startmindmap": "PlantUML Mind Map",
    "@startgantt": "PlantUML Gantt",
}

