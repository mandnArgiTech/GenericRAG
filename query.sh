#!/usr/bin/env bash
# Universal Domain RAG — query runner (Chroma + Ollama). Linux; mirrors run.sh layout.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Shell-only options (consumed here; rest passed to query.py)
DB_PATH=""
MODEL=""
NO_OLLAMA=0
OLLAMA_TIMEOUT=30
QUERY_ARGS=()

usage() {
  echo "Usage: $0 [query.sh options] [query.py options]"
  echo ""
  echo "query.sh options (handled by this script):"
  echo "  --db-path PATH       VectorDB directory (exported as DB_PATH; default: <BaseDir>/VectorDB)"
  echo "  --model NAME         Embedding model (exported as EMBEDDING_MODEL)"
  echo "  --no-ollama          Do not start Ollama; expect server on :11434"
  echo "  --ollama-timeout N   Seconds to wait for Ollama (default: 30)"
  echo "  -h, --help           This help"
  echo ""
  echo "All other arguments are passed to query.py. Examples:"
  echo "  $0 -q \"how does STP work\" -k 10"
  echo "  $0 -i"
  echo "  $0 --mode status"
  echo "  $0 --mode concept -q stp --domain nms"
  exit 0
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --db-path)
      DB_PATH="${2:-}"
      shift 2
      ;;
    --model)
      MODEL="${2:-}"
      shift 2
      ;;
    --no-ollama)
      NO_OLLAMA=1
      shift
      ;;
    --ollama-timeout)
      OLLAMA_TIMEOUT="${2:-30}"
      shift 2
      ;;
    -h|--help)
      usage
      ;;
    *)
      QUERY_ARGS+=("$1")
      shift
      ;;
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
echo " Universal Domain RAG - Query Runner"
echo " BaseDir         : $BASE_DIR"
echo "--------------------------------------------------------"

export OLLAMA_MODELS="$BASE_DIR/Models"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OLLAMA_KEEP_ALIVE="${OLLAMA_KEEP_ALIVE:--1}"
export OLLAMA_GPU_OVERHEAD="${OLLAMA_GPU_OVERHEAD:-536870912}"

if [[ -n "$MODEL" ]]; then
  export EMBEDDING_MODEL="$MODEL"
  echo " Embedding model : $MODEL (from --model)"
fi

if [[ -n "$DB_PATH" ]]; then
  mkdir -p "$DB_PATH"
  export DB_PATH="$(cd "$DB_PATH" && pwd)"
  echo " VectorDB path   : $DB_PATH"
else
  export DB_PATH="$(cd "$BASE_DIR/VectorDB" && pwd)"
  echo " VectorDB path   : $DB_PATH"
fi

OLLAMA_PROC=""
OLLAMA_ALREADY=0
OLLAMA_BIN="$BASE_DIR/Ollama/ollama"

cleanup_ollama() {
  if [[ "$NO_OLLAMA" -eq 1 ]]; then
    return 0
  fi
  if [[ "$OLLAMA_ALREADY" -eq 1 ]]; then
    echo "Ollama left running (was already active before query.sh started)." >&2
  elif [[ -n "${OLLAMA_PROC:-}" ]]; then
    echo "Shutting down Ollama server (started by query.sh)..." >&2
    kill "$OLLAMA_PROC" 2>/dev/null || true
    wait "$OLLAMA_PROC" 2>/dev/null || true
  fi
}

trap cleanup_ollama EXIT

if [[ "$NO_OLLAMA" -eq 0 ]]; then
  echo "[1/2] Starting / reusing local Ollama..."
  if curl -sf "http://127.0.0.1:11434/" >/dev/null 2>&1; then
    OLLAMA_ALREADY=1
    echo "            Existing Ollama on :11434 — reusing."
  else
    "$OLLAMA_BIN" serve >/dev/null 2>&1 &
    OLLAMA_PROC=$!
    echo "            Waiting for http://127.0.0.1:11434 (max ${OLLAMA_TIMEOUT}s)..."
    ready=0
    for _ in $(seq 1 "$OLLAMA_TIMEOUT"); do
      if curl -sf "http://127.0.0.1:11434/" >/dev/null 2>&1; then
        ready=1
        break
      fi
      sleep 1
    done
    if [[ "$ready" -ne 1 ]]; then
      [[ -n "$OLLAMA_PROC" ]] && kill "$OLLAMA_PROC" 2>/dev/null || true
      echo "[ERROR] Ollama did not become ready within ${OLLAMA_TIMEOUT}s." >&2
      exit 1
    fi
    echo "            Ollama is ready."
  fi
  emb_model="${EMBEDDING_MODEL:-nomic-embed-text}"
  echo "            Warmup embed: $emb_model"
  curl -sf -X POST "http://127.0.0.1:11434/api/embed" \
    -H "Content-Type: application/json" \
    -d "{\"model\": \"$emb_model\", \"input\": \"warmup\"}" >/dev/null 2>&1 \
    || echo "    (warmup skipped — run: ollama pull $emb_model)" >&2
else
  echo "[1/2] Skipping Ollama start (--no-ollama)."
fi

echo "[2/2] Running query.py..."
cp -f "$SCRIPT_DIR/query.py" "$BASE_DIR/query.py"

PYTHON_EXE="$BASE_DIR/Python/bin/python3"
[[ -x "$PYTHON_EXE" ]] || PYTHON_EXE="$BASE_DIR/Python/bin/python"
[[ -x "$PYTHON_EXE" ]] || PYTHON_EXE="$BASE_DIR/PyEnv/bin/python3"
[[ -x "$PYTHON_EXE" ]] || { echo "[ERROR] Python not found under $BASE_DIR/Python" >&2; exit 1; }

set +e
"$PYTHON_EXE" "$BASE_DIR/query.py" "${QUERY_ARGS[@]}"
exit_code=$?
set -e

if [[ "$exit_code" -ne 0 ]]; then
  echo "[ERROR] query.py exited with code $exit_code." >&2
fi

exit "$exit_code"
