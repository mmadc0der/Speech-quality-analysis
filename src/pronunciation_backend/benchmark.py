from __future__ import annotations

import argparse
import json
import math
import tempfile
import wave
from pathlib import Path
from statistics import mean

import numpy as np

from pronunciation_backend.config import settings
from pronunciation_backend.main import build_pipeline


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=np.float32), q))


def _summarize(values: list[float]) -> dict[str, float]:
    return {
        "mean_ms": round(float(mean(values)), 3) if values else 0.0,
        "p50_ms": round(_percentile(values, 50), 3),
        "p95_ms": round(_percentile(values, 95), 3),
        "min_ms": round(float(min(values)), 3) if values else 0.0,
        "max_ms": round(float(max(values)), 3) if values else 0.0,
    }


def run_benchmark(audio_path: Path, word: str, repeat: int, no_trim: bool = False) -> dict[str, object]:
    pipeline = build_pipeline(settings)
    audio_bytes = audio_path.read_bytes()

    timings: list[object] = []
    for _ in range(repeat):
        _response, timing = pipeline.assess_word_with_timings(word, audio_bytes, no_trim=no_trim)
        timings.append(timing)

    def collect(name: str) -> list[float]:
        return [float(getattr(timing, name)) for timing in timings if getattr(timing, name, None) is not None]

    alignment_subprocess = collect("alignment_subprocess_ms")
    report: dict[str, object] = {
        "word": word,
        "audio_path": str(audio_path),
        "repeat": repeat,
        "stages": {
            "audio_prep": _summarize(collect("audio_prep_ms")),
            "feature_encode": _summarize(collect("feature_encode_ms")),
            "alignment": _summarize(collect("alignment_ms")),
            "feature_build": _summarize(collect("feature_build_ms")),
            "scorer": _summarize(collect("scorer_ms")),
            "reference": _summarize(collect("reference_ms")),
            "response": _summarize(collect("response_ms")),
            "total": _summarize(collect("total_ms")),
        },
    }
    if alignment_subprocess:
        report["stages"]["alignment_subprocess"] = _summarize(alignment_subprocess)
    return report


def write_synthetic_wav(
    path: Path,
    *,
    sample_rate: int = 16_000,
    duration_ms: int = 1000,
    frequency_hz: float = 220.0,
) -> Path:
    frame_count = int(sample_rate * duration_ms / 1000)
    amplitude = 0.2
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        frames = bytearray()
        for index in range(frame_count):
            sample = int(32767 * amplitude * math.sin(2.0 * math.pi * frequency_hz * index / sample_rate))
            frames.extend(sample.to_bytes(2, byteorder="little", signed=True))
        handle.writeframes(bytes(frames))
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark pronunciation scoring stages.")
    parser.add_argument("--audio", type=Path, help="Path to an audio file to score. If omitted, a synthetic WAV is used.")
    parser.add_argument("--word", default="work", help="Target word for scoring.")
    parser.add_argument("--repeat", type=int, default=5, help="Number of benchmark runs to collect.")
    parser.add_argument("--no-trim", action="store_true", help="Skip backend auto-trim while benchmarking.")
    parser.add_argument("--synthetic-duration-ms", type=int, default=1000, help="Synthetic WAV duration when --audio is omitted.")
    parser.add_argument("--synthetic-frequency-hz", type=float, default=220.0, help="Synthetic WAV sine frequency.")
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of human-readable text.")
    args = parser.parse_args()

    if args.audio is None:
        with tempfile.TemporaryDirectory(prefix="pronunciation-benchmark-") as temp_dir:
            audio_path = write_synthetic_wav(
                Path(temp_dir) / "synthetic.wav",
                duration_ms=args.synthetic_duration_ms,
                frequency_hz=args.synthetic_frequency_hz,
            )
            report = run_benchmark(audio_path, args.word, args.repeat, no_trim=args.no_trim)
    else:
        report = run_benchmark(args.audio, args.word, args.repeat, no_trim=args.no_trim)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0

    print(f"Benchmark for {report['word']!r} using {report['audio_path']}")
    print(f"Runs: {report['repeat']}")
    for stage, summary in report["stages"].items():
        print(
            f"{stage:>14}: mean={summary['mean_ms']:.3f} ms  "
            f"p50={summary['p50_ms']:.3f} ms  p95={summary['p95_ms']:.3f} ms"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
