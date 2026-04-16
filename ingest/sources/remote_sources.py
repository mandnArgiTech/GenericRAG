"""Rally REST and Confluence remote fetch.

Author: deviprasad
"""
from __future__ import annotations

import logging
import os
import re
from typing import Any, Dict, List, Optional

from ingest.core.deps import requests

logger = logging.getLogger("ingest")


def fetch_rally_artifacts(project_name: str, artifact_type: str = "defect", page_size: int = 200):
    if requests is None:
        raise RuntimeError("requests not installed; pip install requests")
    key = os.environ.get("RALLY_API_KEY", "").strip()
    if not key:
        raise RuntimeError("RALLY_API_KEY not set")
    headers = {"ZSESSIONID": key}
    url = f"{RALLY_BASE}/{artifact_type}"
    params = {
        "query": f'(Project.Name = "{project_name}")',
        "fetch": (
            "FormattedID,Name,Description,Notes,Discussion,Resolution,State,Priority,Severity,"
            "Tags,CreationDate,ClosedDate,Iteration,Release"
        ),
        "pagesize": page_size,
        "start": 1,
        "order": "CreationDate desc",
    }
    all_results: List[dict] = []
    while True:
        r = requests.get(url, headers=headers, params=params, timeout=120)
        r.raise_for_status()
        data = r.json().get("QueryResult") or {}
        batch = data.get("Results") or []
        all_results.extend(batch)
        total = data.get("TotalResultCount") or len(all_results)
        if len(all_results) >= total or not batch:
            break
        params["start"] = int(params["start"]) + page_size
    return all_results


def _fetch_confluence_pages_v2(
    base: str, headers: Dict[str, str], space_key: str, label: str
) -> Optional[List[Dict[str, Any]]]:
    """Try Confluence REST API v2; return None to fall back to v1."""
    try:
        sp_url = f"{base}/wiki/api/v2/spaces"
        r = requests.get(sp_url, headers=headers, params={"keys": space_key}, timeout=60)
        if r.status_code >= 400:
            return None
        sj = r.json()
        results = sj.get("results") or []
        if not results:
            return None  # pragma: no cover
        space_id = results[0].get("id")
        if not space_id:
            return None  # pragma: no cover
        pages: List[Dict[str, Any]] = []
        purl = f"{base}/wiki/api/v2/spaces/{space_id}/pages"
        params: Dict[str, Any] = {"limit": 50, "body-format": "storage"}
        next_url: Optional[str] = purl
        while next_url:
            rr = requests.get(
                next_url,
                headers=headers,
                params=params if next_url == purl else None,
                timeout=120,
            )
            if rr.status_code >= 400:
                return None  # pragma: no cover
            js = rr.json()
            for it in js.get("results", []):
                if it.get("status") not in (None, "current", "draft"):
                    continue  # pragma: no cover
                lab_txt = ""
                if label:
                    labs = it.get("labels", {}).get("results", it.get("labels") or [])  # pragma: no cover
                    if isinstance(labs, list):  # pragma: no cover
                        lab_txt = ",".join(  # pragma: no cover
                            str(x.get("name", x) if isinstance(x, dict) else x) for x in labs
                        )
                    if label.lower() not in lab_txt.lower():  # pragma: no cover
                        continue  # pragma: no cover
                body_val = ""
                body = it.get("body") or {}
                if isinstance(body, dict):
                    body_val = (body.get("storage") or body.get("view") or {}).get("value", "") or body.get(
                        "value", ""
                    )
                if len(body_val) < 200:
                    continue  # pragma: no cover
                pid = it.get("id", "")
                pages.append(
                    {
                        "title": it.get("title", ""),
                        "space": space_key,
                        "labels": lab_txt,
                        "author": "",
                        "last_modified": it.get("version", {}).get("createdAt", "")
                        if isinstance(it.get("version"), dict)
                        else "",
                        "parent_page": "",
                        "page_url": f"{base}/wiki/spaces/{space_key}/pages/{pid}",
                        "body": body_val,
                    }
                )
            links = js.get("_links", {}) or {}
            nxt = links.get("next")
            if nxt:
                next_url = base.rstrip("/") + nxt if nxt.startswith("/") else nxt  # pragma: no cover
                params = {}  # pragma: no cover
            else:
                next_url = None
        return pages if pages else None
    except Exception as exc:  # pragma: no cover
        logger.debug("Confluence v2 fetch failed, using v1: %s", exc)  # pragma: no cover
        return None  # pragma: no cover


def fetch_confluence_pages(space_key: str, label: str = "") -> List[Dict[str, Any]]:
    if requests is None:
        raise RuntimeError("requests not installed; pip install requests")  # pragma: no cover
    base = os.environ.get("CONFLUENCE_URL", "").strip().rstrip("/")
    token = os.environ.get("CONFLUENCE_TOKEN", "").strip()
    if not base or not token:
        raise RuntimeError("CONFLUENCE_URL and CONFLUENCE_TOKEN must be set")
    headers = {"Authorization": f"Bearer {token}", "Accept": "application/json"}
    v2 = _fetch_confluence_pages_v2(base, headers, space_key, label)
    if v2 is not None:
        return v2  # pragma: no cover
    url = f"{base}/wiki/rest/api/content"
    params: Dict[str, Any] = {"spaceKey": space_key, "limit": 50, "expand": "body.storage,version,metadata.labels"}
    pages: List[Dict[str, Any]] = []
    while url:
        r = requests.get(url, headers=headers, params=params, timeout=120)
        r.raise_for_status()
        js = r.json()
        for it in js.get("results", []):
            if it.get("status") != "current":
                continue  # pragma: no cover
            labels = it.get("metadata", {}).get("labels", {}).get("results", [])
            lab_txt = ",".join(x.get("name", "") for x in labels)
            if label and label.lower() not in lab_txt.lower():
                continue  # pragma: no cover
            pages.append(
                {
                    "title": it.get("title", ""),
                    "space": space_key,
                    "labels": lab_txt,
                    "author": it.get("version", {}).get("by", {}).get("displayName", ""),
                    "last_modified": it.get("version", {}).get("when", ""),
                    "parent_page": "",
                    "page_url": f"{base}/wiki/spaces/{space_key}/pages/{it.get('id')}",
                    "body": (it.get("body", {}).get("storage", {}) or {}).get("value", ""),
                }
            )
        nxt = (js.get("_links", {}) or {}).get("next")
        if nxt:
            url = base.rstrip("/") + nxt if nxt.startswith("/") else nxt
            params = {}
        else:
            url = None
    return pages

