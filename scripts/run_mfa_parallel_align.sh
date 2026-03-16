#!/usr/bin/env bash
set -euo pipefail

# Launch LibriTTS MFA alignments in parallel without racing on MFA's shared
# extracted acoustic-model cache.

MFA_BIN="${MFA_BIN:-$(command -v mfa || true)}"
RAW_ROOT="${RAW_ROOT:-/cold/pronunciation/datasets/libritts/raw/LibriTTS}"
MFA_ROOT="${MFA_ROOT:-/cold/pronunciation/datasets/libritts/mfa/LibriTTS}"
LOG_ROOT="${LOG_ROOT:-/cold/pronunciation/logs/mfa}"
MFA_CACHE_ROOT="${MFA_CACHE_ROOT:-$HOME/Documents/MFA}"
ACOUSTIC_MODEL_NAME="${ACOUSTIC_MODEL_NAME:-english_us_arpa}"
DICTIONARY_NAME="${DICTIONARY_NAME:-english_us_arpa}"
MODEL_READY_DIR="${MODEL_READY_DIR:-$MFA_CACHE_ROOT/extracted_models/acoustic/${ACOUSTIC_MODEL_NAME}_acoustic/${ACOUSTIC_MODEL_NAME}}"

if [[ -z "$MFA_BIN" ]]; then
  echo "mfa binary not found. Set MFA_BIN=/absolute/path/to/mfa." >&2
  exit 1
fi

mkdir -p "$MFA_ROOT" "$LOG_ROOT"

if [[ "$#" -gt 0 ]]; then
  SUBSETS=("$@")
else
  SUBSETS=("train-clean-360" "test-clean")
fi

write_lab_sidecars() {
  local subset="$1"
  python - "$RAW_ROOT" "$subset" <<'PY'
from pathlib import Path
import sys

raw_root = Path(sys.argv[1])
subset = sys.argv[2]
subset_dir = raw_root / subset
if not subset_dir.exists():
    raise SystemExit(f"subset not found: {subset_dir}")

wrote = 0
empty = 0
for txt_path in subset_dir.rglob("*.normalized.txt"):
    text = txt_path.read_text(encoding="utf-8").strip()
    lab_path = txt_path.with_suffix("").with_suffix(".lab")
    lab_path.write_text(text + ("\n" if text else ""), encoding="utf-8")
    wrote += 1
    if not text:
        empty += 1

print(f"{subset}: wrote_lab={wrote} empty_text={empty}")
PY
}

launch_align() {
  local subset="$1"
  local log_path="$LOG_ROOT/${subset}.align.log"
  local out_dir="$MFA_ROOT/${subset}"

  mkdir -p "$out_dir"
  echo "Launching ${subset}..."
  nohup "$MFA_BIN" align \
    "$RAW_ROOT/$subset" \
    "$DICTIONARY_NAME" \
    "$ACOUSTIC_MODEL_NAME" \
    "$out_dir" \
    --clean \
    > "$log_path" 2>&1 < /dev/null &
  echo $!
}

wait_for_model_cache() {
  local pid="$1"
  local waited=0
  local sleep_s=2
  local timeout_s=300

  while [[ ! -d "$MODEL_READY_DIR" ]]; do
    if ! kill -0 "$pid" 2>/dev/null; then
      echo "Initial MFA job exited before model cache became ready." >&2
      return 1
    fi
    if (( waited >= timeout_s )); then
      echo "Timed out waiting for MFA model cache at $MODEL_READY_DIR" >&2
      return 1
    fi
    sleep "$sleep_s"
    waited=$((waited + sleep_s))
  done
}

echo "Using MFA binary: $MFA_BIN"
echo "[1/${#SUBSETS[@]}] Writing .lab sidecars..."
for subset in "${SUBSETS[@]}"; do
  write_lab_sidecars "$subset"
done

declare -A PIDS=()
start_index=0

if [[ ! -d "$MODEL_READY_DIR" ]]; then
  first_subset="${SUBSETS[0]}"
  echo "[warmup] Starting ${first_subset} first so MFA can initialize shared model cache..."
  PIDS["$first_subset"]="$(launch_align "$first_subset")"
  wait_for_model_cache "${PIDS[$first_subset]}"
  start_index=1
fi

for ((i=start_index; i<${#SUBSETS[@]}; i++)); do
  subset="${SUBSETS[$i]}"
  PIDS["$subset"]="$(launch_align "$subset")"
done

echo
echo "Started:"
for subset in "${SUBSETS[@]}"; do
  if [[ -n "${PIDS[$subset]:-}" ]]; then
    echo "  ${subset} pid=${PIDS[$subset]}"
  fi
done

echo
echo "Monitor with:"
for subset in "${SUBSETS[@]}"; do
  echo "  tail -f $LOG_ROOT/${subset}.align.log"
done

echo
echo "Jobs were started with nohup; this shell can exit safely."
