#!/usr/bin/env bash
set -euo pipefail

# Measure isolated MFA CLI latency on the remote backend host.
# Compares clean/no_clean and optional micromamba wrapper vs direct mfa binary.

MFA_BIN="${MFA_BIN:-mfa}"
MFA_COMMAND="${MFA_COMMAND:-micromamba run -n mfa mfa}"
ACOUSTIC_MODEL="${ACOUSTIC_MODEL:-english_us_arpa}"
WORD="${WORD:-thought}"
AUDIO_PATH="${1:-}"
WORK_ROOT="${WORK_ROOT:-/tmp/mfa-cli-benchmark}"
REPEAT="${REPEAT:-3}"

if [[ -z "$AUDIO_PATH" ]]; then
  echo "usage: $0 /path/to/sample.wav" >&2
  exit 1
fi

mkdir -p "$WORK_ROOT/corpus" "$WORK_ROOT/out_clean" "$WORK_ROOT/out_no_clean" "$WORK_ROOT/mfa_temp"

cp "$AUDIO_PATH" "$WORK_ROOT/corpus/utterance.wav"
printf '%s\n' "$WORD" > "$WORK_ROOT/corpus/utterance.lab"
printf '%s %s\n' "$WORD" "TH AO1 T" > "$WORK_ROOT/lexicon.dict"

run_case() {
  local label="$1"
  shift
  local total=0
  local index
  for ((index = 1; index <= REPEAT; index++)); do
    local start end elapsed
    start="$(date +%s.%N)"
    "$@"
    end="$(date +%s.%N)"
    elapsed="$(python - <<PY
start = float("$start")
end = float("$end")
print(f"{(end - start) * 1000.0:.3f}")
PY
)"
    echo "$label run $index: ${elapsed} ms"
    total="$(python - <<PY
print(float("$total") + float("$elapsed"))
PY
)"
  done
  python - <<PY
repeat = int("$REPEAT")
total = float("$total")
print(f"$label mean: {total / repeat:.3f} ms over {repeat} runs")
PY
}

echo "=== wrapped MFA: align --clean ==="
run_case "wrapped-clean" bash -lc \
  "$MFA_COMMAND align --clean --temporary_directory $WORK_ROOT/mfa_temp $WORK_ROOT/corpus $WORK_ROOT/lexicon.dict $ACOUSTIC_MODEL $WORK_ROOT/out_clean"

echo "=== wrapped MFA: align --no_clean ==="
run_case "wrapped-no-clean" bash -lc \
  "$MFA_COMMAND align --no_clean --temporary_directory $WORK_ROOT/mfa_temp $WORK_ROOT/corpus $WORK_ROOT/lexicon.dict $ACOUSTIC_MODEL $WORK_ROOT/out_no_clean"

if command -v "$MFA_BIN" >/dev/null 2>&1; then
  echo "=== direct MFA: align --no_clean ==="
  run_case "direct-no-clean" \
    "$MFA_BIN" align --no_clean \
    --temporary_directory "$WORK_ROOT/mfa_temp" \
    "$WORK_ROOT/corpus" \
    "$WORK_ROOT/lexicon.dict" \
    "$ACOUSTIC_MODEL" \
    "$WORK_ROOT/out_no_clean"
fi

echo "=== MFA help: align --clean placement ==="
bash -lc "$MFA_COMMAND align --help" | rg "clean" || true
