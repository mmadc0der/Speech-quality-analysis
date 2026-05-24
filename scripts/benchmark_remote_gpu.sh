#!/usr/bin/env bash
set -euo pipefail

# Run pronunciation backend stage benchmarks on the remote GPU serving host.
# This script is intended for the deployment environment, not the local workstation.

: "${PRONUNCIATION_USE_HF_ENCODER:=1}"
: "${PRONUNCIATION_SCORER_DEVICE:=cuda}"
: "${PRONUNCIATION_MFA_COMMAND:?Set PRONUNCIATION_MFA_COMMAND}"
: "${PRONUNCIATION_MFA_ACOUSTIC_MODEL:?Set PRONUNCIATION_MFA_ACOUSTIC_MODEL}"
: "${PRONUNCIATION_SCORER_CHECKPOINT_PATH:?Set PRONUNCIATION_SCORER_CHECKPOINT_PATH}"

AUDIO_PATH="${1:-}"
WORD="${2:-work}"
REPEAT="${3:-10}"
OUTPUT_JSON="${OUTPUT_JSON:-/tmp/pronunciation_benchmark.json}"

export PRONUNCIATION_USE_HF_ENCODER
export PRONUNCIATION_SCORER_DEVICE
export PRONUNCIATION_MFA_COMMAND
export PRONUNCIATION_MFA_ACOUSTIC_MODEL
export PRONUNCIATION_SCORER_CHECKPOINT_PATH

audio_args=()
if [[ -n "$AUDIO_PATH" ]]; then
  audio_args=(--audio "$AUDIO_PATH")
else
  echo "No WAV path supplied; using synthetic benchmark audio."
fi

echo "=== backend benchmark: clean default (MFA_CLEAN=${PRONUNCIATION_MFA_CLEAN:-0}) ==="
python -m pronunciation_backend.benchmark \
  "${audio_args[@]}" \
  --word "$WORD" \
  --repeat "$REPEAT" \
  --json | tee "$OUTPUT_JSON"

if [[ "${RUN_MFA_CLEAN_COMPARE:-1}" == "1" ]]; then
  echo "=== backend benchmark: MFA clean enabled ==="
  PRONUNCIATION_MFA_CLEAN=1 python -m pronunciation_backend.benchmark \
    "${audio_args[@]}" \
    --word "$WORD" \
    --repeat "$REPEAT" \
    --json | tee "${OUTPUT_JSON%.json}.clean.json"
fi

if [[ "${RUN_COMPILE_COMPARE:-0}" == "1" ]]; then
  echo "=== backend benchmark: compile enabled ==="
  PRONUNCIATION_HF_COMPILE=1 PRONUNCIATION_SCORER_COMPILE=1 python -m pronunciation_backend.benchmark \
    "${audio_args[@]}" \
    --word "$WORD" \
    --repeat "$REPEAT" \
    --json | tee "${OUTPUT_JSON%.json}.compile.json"
fi

echo "Benchmark reports written under /tmp"
