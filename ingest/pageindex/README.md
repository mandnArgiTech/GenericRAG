# PageIndex — Vectorless Retrieval Layer

PageIndex adds a **reasoning-based**, tree-indexed retrieval layer alongside the
existing Chroma vector pipeline in GenericRAG. The LLM agent reads a hierarchical
tree (directories → files, or chapters → sections) and fetches whole pages on
demand. No embeddings, no chunking.

Based on the PageIndex framework by VectifyAI (https://github.com/VectifyAI/PageIndex),
adapted for source code and structured documentation.

---

## When to use which layer

| Query style | Use |
|-------------|-----|
| "How does X work conceptually?" — semantic / fuzzy | `search_knowledge` (Chroma, vector) |
| "Where is `DEVload` defined?" — structural / navigational | `pageindex_find_function`, `get_code_structure`, `get_code_pages` |
| "Explain the diode junction limiting algorithm" — prose docs | `get_doc_structure`, `get_doc_pages` |
| "Trace the NR → DEVload → limiter chain" — multi-source | All three in sequence |

---

## Build

Code PageIndex (one page = one `.c`/`.h` file):

```bash
./run.sh --mode pageindex-code \
  --source /path/to/ngspice/src \
  --pageindex-dir ./Studio-Portable-RAG/PageIndex/code

# With local LLM summaries (SmolLM2 on RTX 4060, or qwen2.5-coder on A6000)
./run.sh --mode pageindex-code --source /path/to/ngspice/src \
  --pageindex-dir ./Studio-Portable-RAG/PageIndex/code \
  --pageindex-llm-summaries --pageindex-summary-model smollm2:1.7b
```

Doc PageIndex (one page = one markdown chapter), with doc→code cross-references:

```bash
./run.sh --mode pageindex-docs \
  --source /path/to/chapters \
  --pageindex-dir ./Studio-Portable-RAG/PageIndex/docs \
  --pageindex-code-dir ./Studio-Portable-RAG/PageIndex/code
```

---

## MCP tools exposed by `mcp_server.py`

| Tool | Index | Purpose |
|------|-------|---------|
| `get_code_document` | code | Document metadata (name, total pages, kind) |
| `get_code_structure` | code | Hierarchical directory/file tree with function & struct metadata |
| `get_code_pages` | code | Retrieve full `.c`/`.h` file content by page number |
| `get_code_by_filepath` | code | Retrieve a file by name/path |
| `pageindex_find_function` | code | Zero-similarity function lookup through tree metadata |
| `get_doc_structure` | docs | Chapter tree with code cross-references |
| `get_doc_pages` | docs | Retrieve full chapter text by chapter number |
| `pageindex_chapters_for_function` | docs | Which chapters reference a given C function |

Set `PAGEINDEX_CODE_DIR` and `PAGEINDEX_DOCS_DIR` env vars in your Cursor MCP
config to point at the built artefacts.

---

## Artefact layout

```
<PageIndex dir>/
├── structure.json         # tree + node metadata (lightweight)
├── pages.json             # full page contents (keyed by page number)
├── page_map.json          # page_number → filepath
└── cross_references.json  # (doc index only) doc_file → {function/file: code_path}
```

`structure.json` is safe to open in an editor; `pages.json` is the bulk content
store (loaded lazily at query time).

---

## No new dependencies

This module only uses what GenericRAG already ships:

- `json`, `pathlib`, `re` (stdlib) — core indexing
- `langchain_ollama` — optional LLM summaries (already a dep)

---

## Author

deviprasad
