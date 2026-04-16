"""Full multi-file / multi-mode ingestion run (Chroma upsert orchestration).

Author: deviprasad
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import os
import queue
import re
import threading
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import chromadb
from langchain_ollama import OllamaEmbeddings
from tqdm import tqdm

from ingest.core.checkpoints import (
    append_manifest,
    load_checkpoint,
    migrate_old_checkpoint,
    save_checkpoint,
    update_repos_manifest,
    validate_embedding_dimension,
    write_ingestion_config,
)
from ingest.chunking.code.code_chunking import choose_strategy_for_path, extract_dependencies
from ingest.core.constants import IGNORED_DIRS, IGNORED_EXTS, INGESTION_VERSION, WRITER_STOP
from ingest.core.deps import aiohttp, pathspec, requests, sanitize_pii
from ingest.chunking.docs_community import chunk_community, chunk_release_notes, parse_frontmatter
from ingest.runtime.embeddings import (
    embedding_worker,
    resolve_embed_ingest_settings,
    run_async_embedding_batches,
)
from ingest.core.file_utils import read_file_bytes
from ingest.sources.git_files import (
    _path_matches_any_nested_gitignore,
    _respect_gitignore,
    git_checkpoint_head_key,
    git_diff_file_sets,
    iter_files,
)
from ingest.core.logging_utils import setup_logging
from ingest.core.metadata_utils import (
    detect_content_type,
    empty_metadata,
    extract_concepts,
    finalize_metadata,
    format_concepts_field,
    iter_concept_ids,
    load_concept_registry,
    make_chunk_id,
    resolve_collection,
)
from ingest.chunking.mib_chunking import chunk_mib
from ingest.chunking.rfc.rfc_chunking import chunk_rfc
from ingest.sources.rally_csv import load_rally_rows_from_csv
from ingest.sources.rally_filter import parse_rally_filter, rally_matches_user_filter
from ingest.sources.remote_sources import fetch_confluence_pages, fetch_rally_artifacts
from ingest.core.state import _embed_lock, shutdown_event
from ingest.runtime.status_dashboard import print_status_dashboard
from ingest.chunking.tickets import chunk_customer_ticket, chunk_rally_ticket
from ingest.chunking.wiki_chunking import chunk_wiki_page

logger = logging.getLogger("ingest")


def ingest_run(args: argparse.Namespace) -> int:
    t0 = time.time()
    setup_logging(args.verbose)
    db_path = Path(args.db_path).resolve()
    db_path.mkdir(parents=True, exist_ok=True)

    if args.mode == "status":
        print_status_dashboard(db_path)
        return 0

    domain = args.domain or "general"
    collection_name = resolve_collection(args.mode, domain, args.collection)
    migrate_old_checkpoint(db_path, collection_name)
    cp_key = f"{collection_name}::checkpoint"
    source_type_map = {
        "code": "code",
        "domain": "domain_doc",
        "rfc": "rfc",
        "rally": "rally",
        "customer": "customer",
        "mib": "mib",
        "wiki": "wiki",
        "release-notes": "release_notes",
        "theory": "theory",
        "community": "community",
    }
    source_type = source_type_map.get(args.mode, "domain_doc")
    rally_filter_rules = parse_rally_filter(getattr(args, "rally_filter", None))

    concept_registry = load_concept_registry(Path(args.concept_registry))

    embed_model = os.environ.get("EMBEDDING_MODEL", "nomic-embed-text").strip()
    embedder = OllamaEmbeddings(model=embed_model)

    client = chromadb.PersistentClient(path=str(db_path))
    coll = client.get_or_create_collection(name=collection_name)

    dim_err = validate_embedding_dimension(coll, embedder, collection_name, embed_model)
    if dim_err:
        if getattr(args, "recreate_collection", False):
            logger.warning(
                "Deleting collection %r and its ingest checkpoint so it can be rebuilt "
                "with model %r (%s)",
                collection_name,
                embed_model,
                dim_err,
            )
            try:
                client.delete_collection(name=collection_name)
            except Exception as exc:  # pragma: no cover
                logger.error("delete_collection failed: %s", exc)  # pragma: no cover
                return 2  # pragma: no cover
            coll = client.get_or_create_collection(name=collection_name)
            ck = load_checkpoint(db_path)
            ck.pop(cp_key, None)
            save_checkpoint(db_path, ck)
            dim_err = validate_embedding_dimension(coll, embedder, collection_name, embed_model)
            if dim_err:
                logger.error("After recreate, embedding check still failed: %s", dim_err)  # pragma: no cover
                return 2  # pragma: no cover
        else:
            logger.error("%s", dim_err)  # pragma: no cover
            logger.error(  # pragma: no cover
                "To rebuild collection %r with the current model, pass --recreate-collection "
                "(removes all vectors in that collection and clears its ingest checkpoint).",
                collection_name,
            )
            return 2  # pragma: no cover

    write_ingestion_config(db_path, embed_model)

    checkpoint = load_checkpoint(db_path)
    file_hashes: Dict[str, str] = {}
    if not args.force and cp_key in checkpoint:
        try:
            file_hashes = json.loads(checkpoint[cp_key])
        except Exception:  # pragma: no cover
            file_hashes = {}  # pragma: no cover

    files_to_process: List[Tuple[Path, Dict[str, Any]]] = []
    root: Optional[Path] = Path(args.source).resolve() if args.source else None
    git_head_key = git_checkpoint_head_key(collection_name)
    stored_git_head = checkpoint.get(git_head_key, "")
    if not isinstance(stored_git_head, str):
        stored_git_head = ""
    new_git_head_commit: Optional[str] = None

    if args.mode == "rally" and root is None:
        if requests is None:
            logger.error("pip install requests for Rally API mode")
            return 2
        if not args.rally_project:
            logger.error("--rally-project required for rally mode without --source")  # pragma: no cover
            return 2  # pragma: no cover
        arts = fetch_rally_artifacts(args.rally_project)
        for obj in arts:
            if not rally_matches_user_filter(obj, rally_filter_rules):
                continue  # pragma: no cover
            sev = str(obj.get("Severity") or "")
            if sev.isdigit() and int(sev) > 3:
                continue  # pragma: no cover
            tags = str(obj.get("Tags") or "")
            if "duplicate" in tags.lower():
                continue  # pragma: no cover
            desc = str(obj.get("Description") or "")
            if not desc.strip():
                continue  # pragma: no cover
            fid = str(obj.get("FormattedID") or "unknown")
            files_to_process.append((Path(fid), {"virtual": True, "rally": obj}))
    elif args.mode == "wiki" and root is None:
        if requests is None:
            logger.error("pip install requests for Confluence wiki mode without --source")  # pragma: no cover
            return 2  # pragma: no cover
        if not args.confluence_space:
            logger.error("--confluence-space required when --source not set")  # pragma: no cover
            return 2  # pragma: no cover
        clabel = getattr(args, "confluence_label", None) or ""
        pages = fetch_confluence_pages(args.confluence_space, clabel)
        for pg in pages:
            body = pg.get("body") or ""
            if len(body) < 200:
                continue  # pragma: no cover
            slug = re.sub(r"\W+", "_", pg.get("title", "page"))[:80] + ".wiki"
            files_to_process.append((Path(slug), {"virtual": True, "wiki": pg}))
    else:
        if root is None:
            env_src = os.environ.get("SOURCE_FOLDER", "").strip()
            if not env_src:
                logger.error("No --source and SOURCE_FOLDER not set")
                return 2
            root = Path(env_src).resolve()  # pragma: no cover
        git_used = False
        if getattr(args, "git_diff", False):
            base_ref = (getattr(args, "git_diff_base", None) or "").strip()
            if not base_ref:
                base_ref = stored_git_head.strip() or "HEAD~1"
            mod_set, del_set, gh = git_diff_file_sets(root, base_ref)
            if mod_set is not None and gh:
                git_used = True
                new_git_head_commit = gh
                git_gi_cache: Dict[str, Any] = {}
                for rel in sorted(del_set or ()):
                    ap = str((root / rel).resolve())
                    try:
                        coll.delete(where={"source": ap})
                    except Exception as exc:
                        logger.warning("git-diff delete chunks for %s: %s", ap, exc)
                    file_hashes.pop(ap, None)
                mib_exts = {".mib", ".my"} if args.mode == "mib" else None
                for rel in sorted(mod_set):
                    p = root / rel
                    if not p.is_file():
                        continue
                    suf = p.suffix.lower()
                    if mib_exts is not None and suf not in mib_exts:
                        continue
                    if args.mode == "code" and suf in IGNORED_EXTS:
                        continue
                    if _respect_gitignore() and pathspec is not None:
                        try:
                            r = p.resolve().relative_to(root.resolve())
                            if _path_matches_any_nested_gitignore(
                                root.resolve(), r.as_posix(), git_gi_cache, is_dir=False
                            ):
                                continue
                        except Exception:
                            pass  # pragma: no cover
                    if args.mode == "rally" and suf == ".csv":
                        try:
                            for row in load_rally_rows_from_csv(p):
                                if not rally_matches_user_filter(row, rally_filter_rules):
                                    continue  # pragma: no cover
                                fid = str(
                                    row.get("FormattedID")
                                    or row.get("formatted_id")
                                    or row.get("ID")
                                    or row.get("id")
                                    or hashlib.md5(str(row).encode()).hexdigest()[:10]
                                )
                                files_to_process.append(
                                    (
                                        Path(fid),
                                        {
                                            "virtual": True,
                                            "rally": row,
                                            "_csv_path": str(p.resolve()),
                                        },
                                    )
                                )
                        except Exception as exc:  # pragma: no cover
                            logger.warning("CSV rally skip %s: %s", p, exc)  # pragma: no cover
                        continue
                    files_to_process.append((p, {"virtual": False}))
            else:
                logger.warning(
                    "git-diff: repository unusable or diff failed; falling back to full directory scan"
                )
        if not git_used:
            if args.mode == "mib":
                paths = iter_files(root, {".mib", ".my"}, skip_dirs=IGNORED_DIRS)
            elif args.mode == "code":
                paths = iter_files(root, None, skip_dirs=IGNORED_DIRS, skip_exts=IGNORED_EXTS)
            else:
                paths = iter_files(root, None, skip_dirs=IGNORED_DIRS)
        else:
            paths = []
        for p in paths:
            if args.mode == "rally" and p.suffix.lower() == ".csv":
                try:
                    for row in load_rally_rows_from_csv(p):
                        if not rally_matches_user_filter(row, rally_filter_rules):
                            continue  # pragma: no cover
                        fid = str(
                            row.get("FormattedID")
                            or row.get("formatted_id")
                            or row.get("ID")
                            or row.get("id")
                            or hashlib.md5(str(row).encode()).hexdigest()[:10]
                        )
                        files_to_process.append(
                            (
                                Path(fid),
                                {
                                    "virtual": True,
                                    "rally": row,
                                    "_csv_path": str(p.resolve()),
                                },
                            )
                        )
                except Exception as exc:  # pragma: no cover
                    logger.warning("CSV rally skip %s: %s", p, exc)  # pragma: no cover
            else:
                files_to_process.append((p, {"virtual": False}))

    chunks_deleted = 0
    if file_hashes:
        physical_keys = {str(p[0].resolve()) for p in files_to_process if not p[1].get("virtual")}
        for old in list(file_hashes.keys()):
            if old in physical_keys or old.startswith("rally:") or old.startswith(("http://", "https://")):
                continue
            if old.startswith("csv:"):  # pragma: no cover
                rest = old[4:]  # pragma: no cover
                ri = rest.rfind(":")  # pragma: no cover
                csvp = rest[:ri] if ri > 0 else rest  # pragma: no cover
                if os.path.isfile(csvp):  # pragma: no cover
                    continue  # pragma: no cover
            elif os.path.isfile(old):  # pragma: no cover
                continue  # pragma: no cover
            if getattr(args, "clean_stale", False):  # pragma: no cover
                try:  # pragma: no cover
                    coll.delete(where={"source": old})  # pragma: no cover
                    chunks_deleted += 1  # pragma: no cover
                except Exception as exc:  # pragma: no cover
                    logger.warning("stale delete failed %s: %s", old, exc)  # pragma: no cover
            del file_hashes[old]  # pragma: no cover

    repo_file_counts: Counter[str] = Counter()
    root_for_rel = root if root is not None else Path(".").resolve()

    def rel_repo_for(path: Path) -> Tuple[str, str]:
        try:
            rel = path.resolve().relative_to(root_for_rel.resolve())
        except Exception:  # pragma: no cover
            return "root", str(path)  # pragma: no cover
        parts = rel.parts
        if len(parts) > 1:
            return parts[0], str(Path(*parts[1:]))  # pragma: no cover
        return "root", parts[0] if parts else str(path)

    work_items: List[Tuple[str, str, Dict[str, str]]] = []
    concepts_found: Counter[str] = Counter()
    ctype_dist: Counter[str] = Counter()
    results_holder: Dict[str, Any] = {
        "processed": 0,
        "failed": 0,
        "errors": [],
        "chunks_created": 0,
        "chunks_updated": 0,
    }
    files_processed = 0

    ingestion_ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    for path, extra in tqdm(files_to_process, desc="Scanning", unit="file"):
        file_deps_str = ""
        if shutdown_event.is_set():
            break  # pragma: no cover
        if extra.get("virtual"):
            if "rally" in extra:
                obj = extra["rally"]
                if extra.get("_csv_path"):
                    rid = str(
                        obj.get("FormattedID")
                        or obj.get("formatted_id")
                        or obj.get("id")
                        or obj.get("ID")
                        or path.name
                    )
                    src_key = f"csv:{extra['_csv_path']}:{rid}"
                else:
                    src_key = f"rally:{obj.get('FormattedID')}"
                if not rally_matches_user_filter(obj, rally_filter_rules):
                    continue  # pragma: no cover
                blob = json.dumps(obj, sort_keys=True)
                h = hashlib.md5(blob.encode()).hexdigest()
                if not args.force and file_hashes.get(src_key) == h:
                    continue  # pragma: no cover
                pieces = chunk_rally_ticket(obj, src_key)
            elif "wiki" in extra:
                pg = extra["wiki"]
                src_key = str(pg.get("page_url") or pg.get("title"))
                h = hashlib.md5(str(pg.get("body")).encode()).hexdigest()
                if not args.force and file_hashes.get(src_key) == h:
                    continue  # pragma: no cover
                meta_pg = {
                    "page_title": pg.get("title", ""),
                    "space": pg.get("space", ""),
                    "labels": pg.get("labels", ""),
                    "author": pg.get("author", ""),
                    "last_modified": pg.get("last_modified", ""),
                    "parent_page": pg.get("parent_page", ""),
                    "page_url": pg.get("page_url", ""),
                }
                pieces = chunk_wiki_page(pg.get("body", ""), src_key, meta_pg)
            else:
                continue  # pragma: no cover
            abs_src = src_key
            ext = ".virtual"
            repo = "external"
            rel = abs_src
            mtime = ""
            size_kb = 0.0
        else:
            if not path.is_file():
                continue  # pragma: no cover
            abs_src = str(path.resolve())
            _sk, chunk_fn, limit_mb = choose_strategy_for_path(
                path,
                source_type,
                mib_keep_deprecated=getattr(args, "mib_keep_deprecated", False),
                embed_model=embed_model,
            )
            max_bytes = limit_mb * 1024 * 1024
            st = path.stat()
            if st.st_size > max_bytes:
                logger.warning(  # pragma: no cover
                    "skip large file %s (%d MB > %d MB)",
                    path,
                    st.st_size // 1024 // 1024,
                    limit_mb,
                )
                continue  # pragma: no cover
            h = file_md5(path)
            if not args.force and file_hashes.get(abs_src) == h:
                continue
            content, _enc = read_file_bytes(path)
            if content is None:
                continue  # pragma: no cover
            if args.mode == "customer":
                if sanitize_pii:
                    content = sanitize_pii(content)
                try:
                    obj = json.loads(content)
                    pieces = chunk_customer_ticket(obj, abs_src)
                except Exception:
                    fm, body = parse_frontmatter(content)
                    pieces = chunk_community(body, abs_src, fm)
                    pieces = [
                        (t, {**m, "chunk_strategy": "customer_ticket", "ticket_id": path.stem})
                        for t, m in pieces
                    ]
            elif args.mode == "rally" and path.suffix.lower() == ".csv":
                pieces = []  # pragma: no cover
                for row in load_rally_rows_from_csv(path):  # pragma: no cover
                    if not rally_matches_user_filter(row, rally_filter_rules):  # pragma: no cover
                        continue  # pragma: no cover
                    pieces.extend(chunk_rally_ticket(row, abs_src))  # pragma: no cover
            elif args.mode == "rally" and path.suffix.lower() in (".json", ".md", ".txt"):
                try:
                    obj = json.loads(content)
                    if isinstance(obj, list):
                        pieces = []
                        for it in obj:
                            if not rally_matches_user_filter(it, rally_filter_rules):
                                continue  # pragma: no cover
                            pieces.extend(chunk_rally_ticket(it, abs_src))
                    else:
                        if not rally_matches_user_filter(obj, rally_filter_rules):
                            pieces = []  # pragma: no cover
                        else:
                            pieces = chunk_rally_ticket(obj, abs_src)
                except Exception:
                    pieces = chunk_rally_ticket(
                        {"FormattedID": path.stem, "Description": content, "Name": path.stem},
                        abs_src,
                    )
            elif args.mode == "community":
                fm, body = parse_frontmatter(content)
                pieces = chunk_community(body, abs_src, fm)
            elif args.mode == "wiki" and path.suffix.lower() in (".html", ".htm", ".md"):
                pieces = chunk_wiki_page(content, abs_src, {})
            elif args.mode == "release-notes":
                pieces = chunk_release_notes(content, abs_src)
            elif args.mode == "mib":
                pieces = chunk_mib(content, path, skip_deprecated=not getattr(args, "mib_keep_deprecated", False))
            elif args.mode == "rfc":
                pieces = chunk_rfc(content, abs_src, embed_model=embed_model)
            else:
                pieces = chunk_fn(path, content)
            ext = path.suffix.lower()
            repo, rel = rel_repo_for(path)
            mtime = datetime.fromtimestamp(st.st_mtime, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            size_kb = round(st.st_size / 1024.0, 3)
            file_deps_str = extract_dependencies(content, ext) if source_type == "code" else ""

        if not pieces:
            continue  # pragma: no cover

        if not extra.get("virtual") and abs_src and not args.dry_run:
            try:
                coll.delete(where={"source": abs_src})
            except Exception:  # pragma: no cover
                pass  # pragma: no cover

        base = empty_metadata()
        base.update(
            {
                "source": abs_src,
                "source_type": source_type,
                "domain": domain,
                "repository": repo,
                "relative_path": rel,
                "extension": ext,
                "file_size_kb": str(size_kb),
                "last_modified": mtime,
                "ingestion_date": ingestion_ts,
                "ingestion_version": INGESTION_VERSION,
                "dependencies": file_deps_str,
            }
        )

        for i, (text, partial) in enumerate(pieces):
            meta = {**base, **partial}
            meta["chunk_index"] = str(partial.get("chunk_index", i))
            ctype = partial.get("content_type") or detect_content_type(text)
            meta["content_type"] = ctype
            ctype_dist[ctype] += 1
            raw_concepts = partial.get("concepts")
            if raw_concepts is not None and str(raw_concepts).strip() != "":
                rc = str(raw_concepts).strip()  # pragma: no cover
                concepts = rc if rc.startswith("|") else format_concepts_field(iter_concept_ids(rc))  # pragma: no cover
            else:
                concepts = extract_concepts(text, domain, concept_registry)
            meta["concepts"] = concepts
            for c in iter_concept_ids(concepts):
                concepts_found[c] += 1  # pragma: no cover
            meta = finalize_metadata(meta)
            try:
                cidx = int(meta.get("chunk_index") or i)
            except ValueError:  # pragma: no cover
                cidx = i  # pragma: no cover
            cid = make_chunk_id(abs_src, cidx, text)
            work_items.append((cid, text, meta))

        files_processed += 1
        if source_type == "code" and not extra.get("virtual"):
            repo_file_counts[repo] += 1
        if not args.dry_run:
            if extra.get("virtual"):
                file_hashes[src_key] = h
            else:
                file_hashes[abs_src] = h

    if args.dry_run:
        logger.info("dry-run: %d files, %d chunks (planned)", files_processed, len(work_items))
        return 0

    chunk_q: "queue.Queue[Optional[List[Tuple[str, str, Dict[str, str]]]]]" = queue.Queue(maxsize=256)
    result_q: "queue.Queue[Optional[Tuple[List[str], List[str], List[Dict[str, str]], List[List[float]]]]]" = (
        queue.Queue(maxsize=512)
    )
    def writer_loop():
        while True:
            item = result_q.get()
            if item is WRITER_STOP:
                break
            if item is None:
                results_holder["failed"] += 1  # pragma: no cover
                continue  # pragma: no cover
            ids, texts, metas, embeddings = item
            try:
                existing_ids: set = set()
                try:
                    prev = coll.get(ids=ids, include=[])
                    if prev and prev.get("ids"):
                        existing_ids = set(prev["ids"])  # pragma: no cover
                except Exception:  # pragma: no cover
                    pass  # pragma: no cover
                created = sum(1 for i in ids if i not in existing_ids)
                updated = len(ids) - created
                results_holder["chunks_created"] += created
                results_holder["chunks_updated"] += updated
                coll.upsert(ids=ids, documents=texts, metadatas=metas, embeddings=embeddings)
                results_holder["processed"] += len(ids)
            except Exception as exc:  # pragma: no cover
                logger.exception("upsert failed: %s", exc)  # pragma: no cover
                results_holder["failed"] += len(ids)  # pragma: no cover
                results_holder["errors"].append(str(exc))  # pragma: no cover

    wthread = threading.Thread(target=writer_loop, daemon=True)
    wthread.start()

    batch_size, workers_n, embed_concurrency = resolve_embed_ingest_settings()
    batches = [work_items[i : i + batch_size] for i in range(0, len(work_items), batch_size)]
    use_async_embed = (
        os.environ.get("EMBED_ASYNC", "1").strip().lower() not in ("0", "false", "no")
        and aiohttp is not None
    )

    if use_async_embed:
        try:
            outs = asyncio.run(
                run_async_embedding_batches(batches, embed_model, embed_concurrency)
            )
        except Exception as exc:  # pragma: no cover
            logger.exception("async embedding failed: %s", exc)
            results_holder["errors"].append(str(exc))
            outs = [None] * len(batches)
        for bi, item in enumerate(outs):
            if item is None:
                results_holder["failed"] += len(batches[bi]) if bi < len(batches) else 0
            else:
                result_q.put(item)
    else:
        threads: List[threading.Thread] = []
        for wid in range(workers_n):
            t = threading.Thread(
                target=embedding_worker,
                args=(embed_model, wid, chunk_q, result_q),
                name=f"embed-{wid}",
                daemon=True,
            )
            t.start()
            threads.append(t)
        try:
            for batch in tqdm(batches, desc="Embedding", unit="batch"):
                if shutdown_event.is_set():
                    break  # pragma: no cover
                chunk_q.put(batch)
        finally:
            for _ in threads:
                chunk_q.put(None)
        for t in threads:
            t.join(timeout=600)

    result_q.put(WRITER_STOP)
    wthread.join(timeout=600)

    checkpoint[cp_key] = json.dumps(file_hashes)
    if new_git_head_commit:
        checkpoint[git_head_key] = new_git_head_commit
    save_checkpoint(db_path, checkpoint)
    if repo_file_counts:
        update_repos_manifest(db_path, collection_name, dict(repo_file_counts))

    duration = time.time() - t0
    record = {
        "ingestion_id": datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6],
        "mode": args.mode,
        "domain": domain,
        "collection": collection_name,
        "timestamp": ingestion_ts,
        "files_processed": files_processed,
        "chunks_created": results_holder["chunks_created"],
        "chunks_updated": results_holder["chunks_updated"],
        "chunks_upserted": results_holder["processed"],
        "chunks_deleted_stale": chunks_deleted,
        "concepts_found": sorted(concepts_found.keys()),
        "content_type_distribution": dict(ctype_dist),
        "embedding_model": embed_model,
        "duration_seconds": round(duration, 3),
        "errors": results_holder["errors"],
    }
    append_manifest(db_path, record)
    logger.info("Done: %s", record)
    return 0 if not results_holder["errors"] else 1
