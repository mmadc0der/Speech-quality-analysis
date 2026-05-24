from __future__ import annotations

import argparse
import json
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
        return [float(getattr(timing, name)) for timing in timings]

    return {
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


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark pronunciation scoring stages.")
    parser.add_argument("--audio", type=Path, required=True, help="Path to an audio file to score.")
    parser.add_argument("--word", required=True, help="Target word for scoring.")
    parser.add_argument("--repeat", type=int, default=5, help="Number of benchmark runs to collect.")
    parser.add_argument("--no-trim", action="store_true", help="Skip backend auto-trim while benchmarking.")
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of human-readable text.")
    args = parser.parse_args()

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
