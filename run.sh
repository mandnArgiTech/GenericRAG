#!/usr/bin/env bash
# Linux counterpart to run.ps1 — ingestion runner using Studio-Portable-RAG layout
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MODEL="nomic-embed-text"
MODE=""
DOMAIN="general"
COLLECTION=""
RALLY_PROJECT=""
RALLY_FILTER=""
CONFLUENCE_SPACE=""
CONFLUENCE_LABEL=""
MIB_KEEP_DEPRECATED=0
DRY_RUN=0
FORCE=0
CLEAN_STALE=0
RECREATE_COLLECTION=0
VERBOSE=0
REPO=""
SOURCE=""
DB_PATH=""
WORKERS=2
GIT_DIFF=0
GIT_DIFF_BASE=""
CONCEPT_REGISTRY=""
PAGEINDEX_DIR=""
PAGEINDEX_CODE_DIR=""
PAGEINDEX_LLM_SUMMARIES=0
PAGEINDEX_SUMMARY_MODEL=""

usage() {
  echo "Usage: $0 [options]"
  echo "  --model NAME              Embedding model (default: nomic-embed-text)"
  echo "  --mode MODE               code|domain|rfc|rally|customer|mib|wiki|release-notes|theory|community|status|pageindex-code|pageindex-docs"
  echo "  --domain NAME             default: general"
  echo "  --collection NAME         optional Chroma collection override"
  echo "  --rally-project NAME"
  echo "  --rally-filter EXPR"
  echo "  --confluence-space KEY"
  echo "  --confluence-label LABEL"
  echo "  --repo NAME               sub-folder filter or single-repo name (see run.ps1)"
  echo "  --source PATH             source file or directory"
  echo "  --db-path PATH            VectorDB directory"
  echo "  --workers N               embedding threads (default: 2)"
  echo "  --git-diff                Only ingest git-changed files vs base ref"
  echo "  --git-diff-base REF       Git ref to diff against"
  echo "  --concept-registry PATH   Path to concept_registry.json"
  echo "  --pageindex-dir PATH      Output dir for pageindex-code / pageindex-docs modes"
  echo "  --pageindex-code-dir PATH Existing code PageIndex to link from docs (pageindex-docs only)"
  echo "  --pageindex-llm-summaries Enable LLM summaries (requires Ollama model)"
  echo "  --pageindex-summary-model MODEL  Ollama summary model (default: smollm2:1.7b)"
  echo "  --mib-keep-deprecated"
  echo "  --dry-run  --force  --clean-stale  --recreate-collection  --verbose"
  exit 0
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model) MODEL="${2:-}"; shift 2 ;;
    --mode) MODE="${2:-}"; shift 2 ;;
    --domain) DOMAIN="${2:-}"; shift 2 ;;
    --collection) COLLECTION="${2:-}"; shift 2 ;;
    --rally-project) RALLY_PROJECT="${2:-}"; shift 2 ;;
    --rally-filter) RALLY_FILTER="${2:-}"; shift 2 ;;
    --confluence-space) CONFLUENCE_SPACE="${2:-}"; shift 2 ;;
    --confluence-label) CONFLUENCE_LABEL="${2:-}"; shift 2 ;;
    --repo) REPO="${2:-}"; shift 2 ;;
    --source) SOURCE="${2:-}"; shift 2 ;;
    --db-path) DB_PATH="${2:-}"; shift 2 ;;
    --workers) WORKERS="${2:-2}"; shift 2 ;;
    --mib-keep-deprecated) MIB_KEEP_DEPRECATED=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    --force) FORCE=1; shift ;;
    --clean-stale) CLEAN_STALE=1; shift ;;
    --recreate-collection) RECREATE_COLLECTION=1; shift ;;
    --git-diff) GIT_DIFF=1; shift ;;
    --git-diff-base) GIT_DIFF_BASE="${2:-}"; shift 2 ;;
    --concept-registry) CONCEPT_REGISTRY="${2:-}"; shift 2 ;;
    --pageindex-dir) PAGEINDEX_DIR="${2:-}"; shift 2 ;;
    --pageindex-code-dir) PAGEINDEX_CODE_DIR="${2:-}"; shift 2 ;;
    --pageindex-llm-summaries) PAGEINDEX_LLM_SUMMARIES=1; shift ;;
    --pageindex-summary-model) PAGEINDEX_SUMMARY_MODEL="${2:-}"; shift 2 ;;
    --verbose) VERBOSE=1; shift ;;
    -h|--help) usage ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
  esac
done

BASE_DIR="$(cd "$SCRIPT_DIR/Studio-Portable-RAG" 2>/dev/null && pwd)" || BASE_DIR=""
if [[ ! -x "${BASE_DIR:-}/Ollama/ollama" ]]; then
  alt="$(cd "$SCRIPT_DIR/Portable_RAG" 2>/dev/null && pwd)" || alt=""
  if [[ -n "$alt" && -x "$alt/Ollama/ollama" ]]; then
    BASE_DIR="$alt"
  fi
fi

if [[ -z "$BASE_DIR" || ! -x "$BASE_DIR/Ollama/ollama" ]]; then
  echo "[ERROR] Portable RAG not found. Run ./build.sh first (expect Studio-Portable-RAG/Ollama/ollama)." >&2
  exit 1
fi

echo "--------------------------------------------------------"
echo " Universal Domain RAG - Ingestion Runner"
echo " BaseDir         : $BASE_DIR"
echo " Mode            : ${MODE:-(legacy / default)}"
echo " Domain          : $DOMAIN"
echo " Embedding model : $MODEL"
echo " Embed workers   : $WORKERS"
echo "--------------------------------------------------------"

echo ""
echo "[GPU] Checking for NVIDIA GPU..."
if command -v nvidia-smi >/dev/null 2>&1; then
  if gpu_info="$(nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null)"; then
    [[ -n "$gpu_info" ]] && echo " GPU detected: $gpu_info" || echo " WARNING: nvidia-smi returned no output. Running on CPU."
  else
    echo " WARNING: nvidia-smi failed. Running on CPU."
  fi
else
  echo " WARNING: nvidia-smi not found. Running on CPU."
fi

echo ""
echo "[1/3] Configuring environment..."
export OLLAMA_MODELS="$BASE_DIR/Models"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OLLAMA_NUM_PARALLEL="$WORKERS"
export EMBED_WORKERS="$WORKERS"
export OLLAMA_KEEP_ALIVE="-1"
export OLLAMA_GPU_OVERHEAD="${OLLAMA_GPU_OVERHEAD:-536870912}"
export EMBEDDING_MODEL="$MODEL"

if [[ -n "$SOURCE" ]]; then
  [[ -e "$SOURCE" ]] || { echo "[ERROR] Source path not found: $SOURCE" >&2; exit 1; }
  if command -v realpath >/dev/null 2>&1; then
    export SOURCE_FOLDER="$(realpath "$SOURCE")"
  else
    if [[ -d "$SOURCE" ]]; then
      export SOURCE_FOLDER="$(cd "$SOURCE" && pwd)"
    else
      export SOURCE_FOLDER="$(cd "$(dirname "$SOURCE")" && pwd)/$(basename "$SOURCE")"
    fi
  fi
  echo " Source path     : $SOURCE_FOLDER"
else
  export SOURCE_FOLDER=""
fi

if [[ -n "$MODE" && "$MODE" != "status" && "$MODE" != "rally" && "$MODE" != "wiki" && -z "${SOURCE_FOLDER:-}" ]]; then
  case "$MODE" in
    code) default_folder="Codebase" ;;
    domain|theory) default_folder="DomainDocs" ;;
    rfc) default_folder="RFCs" ;;
    mib) default_folder="MIBs" ;;
    community|customer) default_folder="CommunityData" ;;
    release-notes) default_folder="Codebase" ;;
    *) default_folder="Codebase" ;;
  esac
  default_src="$BASE_DIR/$default_folder"
  if [[ -d "$default_src" ]]; then
    export SOURCE_FOLDER="$(cd "$default_src" && pwd)"
    echo " Default source  : $SOURCE_FOLDER (mode=$MODE -> $default_folder)"
  fi
fi

if [[ -n "$DB_PATH" ]]; then
  mkdir -p "$DB_PATH"
  export DB_PATH="$(cd "$DB_PATH" && pwd)"
  echo " VectorDB path   : $DB_PATH"
else
  export DB_PATH="$(cd "$BASE_DIR/VectorDB" && pwd)"
  echo " VectorDB path   : $DB_PATH"
fi

export INGEST_REPO=""
export REPO_NAME=""
if [[ -n "$REPO" ]]; then
  codebase_default="$BASE_DIR/Codebase"
  effective_source="${SOURCE_FOLDER:-$codebase_default}"
  source_folder_name="$(basename "$effective_source")"
  if [[ "$REPO" == "$source_folder_name" ]]; then
    export REPO_NAME="$REPO"
    echo "            Repo name      : $REPO (single-repo mode)"
  elif [[ -d "$effective_source/$REPO" ]]; then
    export INGEST_REPO="$REPO"
    echo "            Repo filter    : $REPO (sub-folder of source)"
  else
    export REPO_NAME="$REPO"
    echo "            Repo name      : $REPO (single-repo mode - alias for '$source_folder_name')"
  fi
fi

OLLAMA_PROC=""
OLLAMA_ALREADY=0
OLLAMA_BIN="$BASE_DIR/Ollama/ollama"

echo "[2/3] Starting local Ollama server..."
if curl -sf "http://127.0.0.1:11434/" >/dev/null 2>&1; then
  OLLAMA_ALREADY=1
  echo "            Existing Ollama instance detected on :11434. Reusing it."
else
  "$OLLAMA_BIN" serve >/dev/null 2>&1 &
  OLLAMA_PROC=$!
  echo "        Waiting http://127.0.0.1:11434 for readiness..."
  ready=0
  for _ in $(seq 1 30); do
    if curl -sf "http://127.0.0.1:11434/" >/dev/null 2>&1; then
      ready=1
      break
    fi
    sleep 1
  done
  if [[ "$ready" -ne 1 ]]; then
    [[ -n "$OLLAMA_PROC" ]] && kill "$OLLAMA_PROC" 2>/dev/null || true
    echo "[ERROR] Ollama did not become ready within 30 seconds." >&2
    exit 1
  fi
  echo "        Ollama is ready."
fi

echo "    Verifying GPU usage for '$MODEL'..."
if curl -sf -X POST "http://127.0.0.1:11434/api/embed" \
  -H "Content-Type: application/json" \
  -d "{\"model\": \"$MODEL\", \"input\": \"warmup\"}" >/dev/null 2>&1; then
  if ps_json="$(curl -sf "http://127.0.0.1:11434/api/ps" 2>/dev/null)"; then
    echo "    Model status: $(echo "$ps_json" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d)" 2>/dev/null || echo "$ps_json")"
  fi
else
  echo "    GPU verification skipped (embed warmup failed — model may need: ollama pull $MODEL)"
fi

echo "[3/3] Running ingestion pipeline..."
rm -rf "$BASE_DIR/ingest"
cp -a "$SCRIPT_DIR/ingest" "$BASE_DIR/ingest"
[[ -f "$SCRIPT_DIR/mcp_server.py" ]] && cp -f "$SCRIPT_DIR/mcp_server.py" "$BASE_DIR/mcp_server.py" || true
[[ -f "$SCRIPT_DIR/domain_feeder.py" ]] && cp -f "$SCRIPT_DIR/domain_feeder.py" "$BASE_DIR/domain_feeder.py" || true
[[ -f "$SCRIPT_DIR/sanitizer.py" ]] && cp -f "$SCRIPT_DIR/sanitizer.py" "$BASE_DIR/sanitizer.py" || true
[[ -f "$SCRIPT_DIR/concept_registry.json" ]] && cp -f "$SCRIPT_DIR/concept_registry.json" "$BASE_DIR/concept_registry.json" || true

PYTHON_EXE="$BASE_DIR/Python/bin/python3"
[[ -x "$PYTHON_EXE" ]] || PYTHON_EXE="$BASE_DIR/Python/bin/python"
[[ -x "$PYTHON_EXE" ]] || PYTHON_EXE="$BASE_DIR/PyEnv/bin/python3"
[[ -x "$PYTHON_EXE" ]] || { echo "[ERROR] Python not found under $BASE_DIR/Python" >&2; exit 1; }

INGEST_ARGS=( -m ingest )
[[ -n "$MODE" ]] && INGEST_ARGS+=( --mode "$MODE" )
[[ -n "$DOMAIN" ]] && INGEST_ARGS+=( --domain "$DOMAIN" )
[[ -n "$COLLECTION" ]] && INGEST_ARGS+=( --collection "$COLLECTION" )
[[ -n "$DB_PATH" ]] && INGEST_ARGS+=( --db-path "$DB_PATH" )
if [[ -n "$MODE" && "$MODE" != "status" && -n "${SOURCE_FOLDER:-}" ]]; then
  INGEST_ARGS+=( --source "$SOURCE_FOLDER" )
fi
[[ -n "$RALLY_PROJECT" ]] && INGEST_ARGS+=( --rally-project "$RALLY_PROJECT" )
[[ -n "$RALLY_FILTER" ]] && INGEST_ARGS+=( --rally-filter "$RALLY_FILTER" )
[[ -n "$CONFLUENCE_SPACE" ]] && INGEST_ARGS+=( --confluence-space "$CONFLUENCE_SPACE" )
[[ -n "$CONFLUENCE_LABEL" ]] && INGEST_ARGS+=( --confluence-label "$CONFLUENCE_LABEL" )
[[ "$MIB_KEEP_DEPRECATED" -eq 1 ]] && INGEST_ARGS+=( --mib-keep-deprecated )
[[ "$DRY_RUN" -eq 1 ]] && INGEST_ARGS+=( --dry-run )
[[ "$FORCE" -eq 1 ]] && INGEST_ARGS+=( --force )
[[ "$CLEAN_STALE" -eq 1 ]] && INGEST_ARGS+=( --clean-stale )
[[ "$RECREATE_COLLECTION" -eq 1 ]] && INGEST_ARGS+=( --recreate-collection )
[[ "$GIT_DIFF" -eq 1 ]] && INGEST_ARGS+=( --git-diff )
[[ -n "$GIT_DIFF_BASE" ]] && INGEST_ARGS+=( --git-diff-base "$GIT_DIFF_BASE" )
[[ -n "$CONCEPT_REGISTRY" ]] && INGEST_ARGS+=( --concept-registry "$CONCEPT_REGISTRY" )
[[ -n "$PAGEINDEX_DIR" ]] && INGEST_ARGS+=( --pageindex-dir "$PAGEINDEX_DIR" )
[[ -n "$PAGEINDEX_CODE_DIR" ]] && INGEST_ARGS+=( --pageindex-code-dir "$PAGEINDEX_CODE_DIR" )
[[ "$PAGEINDEX_LLM_SUMMARIES" -eq 1 ]] && INGEST_ARGS+=( --pageindex-llm-summaries )
[[ -n "$PAGEINDEX_SUMMARY_MODEL" ]] && INGEST_ARGS+=( --pageindex-summary-model "$PAGEINDEX_SUMMARY_MODEL" )
[[ "$VERBOSE" -eq 1 ]] && INGEST_ARGS+=( --verbose )

cleanup_ollama() {
  if [[ "$OLLAMA_ALREADY" -eq 1 ]]; then
    echo "Ollama left running (was already active before this script started)."
  elif [[ -n "${OLLAMA_PROC:-}" ]]; then
    echo "Shutting down Ollama server..."
    kill "$OLLAMA_PROC" 2>/dev/null || true
    wait "$OLLAMA_PROC" 2>/dev/null || true
    echo "       Ollama stopped."
  fi
}

set +u
cd "$BASE_DIR"
set -u
if "$PYTHON_EXE" "${INGEST_ARGS[@]}"; then
  exit_code=0
else
  exit_code=$?
fi
cleanup_ollama

if [[ "$exit_code" -ne 0 ]]; then
  echo "[ERROR] ingest module failed with exit code $exit_code." >&2
  exit "$exit_code"
fi

echo ""
echo " Run finished. VectorDB: $DB_PATH"
echo " Examples:"
echo "   ./run.sh --mode code --model mxbai-embed-large"
echo "   ./run.sh --mode domain --domain nms --source \"$BASE_DIR/DomainDocs\""
echo "   ./run.sh --mode status"
echo "   ./run.sh --mode mib --domain nms --source \"$BASE_DIR/MIBs\""
echo "   ./run.sh --mode pageindex-code --source \"$BASE_DIR/Codebase/ngspice/src\" --pageindex-dir \"$BASE_DIR/PageIndex/code\""
echo "   ./run.sh --mode pageindex-docs --source \"$BASE_DIR/DomainDocs/chapters\" --pageindex-dir \"$BASE_DIR/PageIndex/docs\" --pageindex-code-dir \"$BASE_DIR/PageIndex/code\""
echo ""
