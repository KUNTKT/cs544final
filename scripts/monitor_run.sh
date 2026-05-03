#!/usr/bin/env bash
# Terminal monitor for experiment progress: prediction line counts + summary presence
#
#   ./scripts/monitor_run.sh                         # latest outputs/run_*
#   ./scripts/monitor_run.sh outputs/run_xxx         # specific directory
#   ./scripts/monitor_run.sh --watch 10              # refresh every 10s (needs watch)
#   ./scripts/monitor_run.sh outputs/run_xxx --watch 5
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RUN_DIR=""
INTERVAL=10
WATCH=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --watch|-w)
      WATCH=1
      INTERVAL="${2:-10}"
      shift 2
      ;;
    *)
      if [[ -z "$RUN_DIR" && -d "$1" ]]; then
        RUN_DIR="$1"
      fi
      shift
      ;;
  esac
done

if [[ -z "$RUN_DIR" ]]; then
  RUN_DIR="$(ls -td "$ROOT"/outputs/run_* 2>/dev/null | head -1 || true)"
fi

if [[ -z "$RUN_DIR" || ! -d "$RUN_DIR" ]]; then
  echo "No run directory found. Usage: $0 [outputs/run_YYYYMMDD_HHMMSS] [--watch SECONDS]" >&2
  exit 1
fi

show() {
  echo "======== $(date '+%Y-%m-%d %H:%M:%S') ========"
  echo "Directory: $RUN_DIR"
  if [[ -f "$RUN_DIR/meta.json" ]]; then
    grep -E '"num_items"|"mode"|"backend"' "$RUN_DIR/meta.json" 2>/dev/null || true
  fi
  echo "--- prediction line counts (one line per item) ---"
  shopt -s nullglob
  local files=("$RUN_DIR"/predictions__*.jsonl)
  if [[ ${#files[@]} -eq 0 ]]; then
    echo "(no predictions__*.jsonl yet—model may still be loading)"
  else
    wc -l "${files[@]}" | sort -n
  fi
  if [[ -f "$RUN_DIR/summary.json" ]]; then
    echo "--- status: finished (summary.json present) ---"
  else
    echo "--- status: in progress (no summary.json yet) ---"
  fi
  echo ""
}

if [[ "$WATCH" -eq 1 ]]; then
  if ! command -v watch >/dev/null 2>&1; then
    echo "watch not installed; looping every ${INTERVAL}s" >&2
    while true; do
      clear 2>/dev/null || true
      show
      sleep "$INTERVAL"
    done
  else
    watch -n "$INTERVAL" "$ROOT/scripts/monitor_run.sh" "$RUN_DIR"
  fi
else
  show
fi
