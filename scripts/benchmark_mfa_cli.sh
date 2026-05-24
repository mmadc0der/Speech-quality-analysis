#!/usr/bin/env bash
set -euo pipefail

# Measure isolated MFA CLI latency on the remote backend host.
# Compares clean/no_clean and optional micromamba wrapper vs direct mfa binary.

MFA_BIN="${MFA_BIN:-mfa}"
MFA_COMMAND="${MFA_COMMAND:-micromamba run -n mfa mfa}"
ACOUSTIC_MODEL="${ACOUSTIC_MODEL:-english_us_arpa}"
AUDIO_PATH="${1:-}"
WORD="${2:-${WORD:-work}}"
WORK_ROOT="${WORK_ROOT:-/tmp/mfa-cli-benchmark}"
REPEAT="${REPEAT:-3}"

mkdir -p "$WORK_ROOT/corpus" "$WORK_ROOT/out_clean" "$WORK_ROOT/out_no_clean" "$WORK_ROOT/mfa_temp"

export WORD WORK_ROOT

if [[ -n "$AUDIO_PATH" ]]; then
  cp "$AUDIO_PATH" "$WORK_ROOT/corpus/utterance.wav"
else
  echo "No WAV path supplied; generating synthetic MFA audio."
  python - <<'PY'
import math
import os
import wave
from pathlib import Path

path = Path(os.environ["WORK_ROOT"]) / "corpus" / "utterance.wav"
sample_rate = 16_000
duration_ms = 1000
frequency_hz = 220.0
frame_count = int(sample_rate * duration_ms / 1000)
frames = bytearray()
for index in range(frame_count):
    sample = int(32767 * 0.2 * math.sin(2.0 * math.pi * frequency_hz * index / sample_rate))
    frames.extend(sample.to_bytes(2, byteorder="little", signed=True))
with wave.open(str(path), "wb") as handle:
    handle.setnchannels(1)
    handle.setsampwidth(2)
    handle.setframerate(sample_rate)
    handle.writeframes(bytes(frames))
PY
fi

printf '%s\n' "$WORD" > "$WORK_ROOT/corpus/utterance.lab"
python - <<'PY'
import os
from pathlib import Path

from pronunciation_backend.config import settings
from pronunciation_backend.services.lexicon import LexiconService
from pronunciation_backend.services.mfa_dictionary import runtime_dictionary_line

word = os.environ["WORD"]
work_root = Path(os.environ["WORK_ROOT"])
service = LexiconService(settings.lexicon_path, cmudict_path=settings.cmudict_path)
entry = service.get_word(word)
(work_root / "lexicon.dict").write_text(runtime_dictionary_line(entry) + "\n", encoding="utf-8")
print(f"MFA dictionary entry: {runtime_dictionary_line(entry)}")
PY

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
