from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Interval:
    xmin: float
    xmax: float
    text: str


@dataclass(frozen=True)
class IntervalTier:
    name: str
    intervals: list[Interval]


def parse_textgrid(path: Path) -> dict[str, IntervalTier]:
    lines = path.read_text(encoding="utf-8").splitlines()
    tiers: dict[str, IntervalTier] = {}

    current_name: str | None = None
    current_intervals: list[Interval] = []
    current_xmin: float | None = None
    current_xmax: float | None = None
    current_text: str | None = None

    def flush_interval() -> None:
        nonlocal current_xmin, current_xmax, current_text, current_intervals
        if current_xmin is None or current_xmax is None or current_text is None:
            return
        current_intervals.append(Interval(xmin=current_xmin, xmax=current_xmax, text=current_text))
        current_xmin = None
        current_xmax = None
        current_text = None

    def flush_tier() -> None:
        nonlocal current_name, current_intervals
        if current_name is None:
            return
        flush_interval()
        tiers[current_name] = IntervalTier(name=current_name, intervals=current_intervals)
        current_name = None
        current_intervals = []

    for raw_line in lines:
        line = raw_line.strip()
        if line.startswith("item ["):
            flush_tier()
            continue
        if line.startswith("name = "):
            flush_tier()
            current_name = _parse_string_value(line)
            continue
        if line.startswith("intervals ["):
            flush_interval()
            continue
        if line.startswith("xmin = "):
            value = _parse_float_value(line)
            if current_name is not None:
                current_xmin = value
            continue
        if line.startswith("xmax = "):
            value = _parse_float_value(line)
            if current_name is not None:
                current_xmax = value
            continue
        if line.startswith("text = "):
            if current_name is not None:
                current_text = _parse_string_value(line)
            flush_interval()

    flush_tier()
    return tiers


def _parse_float_value(line: str) -> float:
    return float(line.split("=", 1)[1].strip())


def _parse_string_value(line: str) -> str:
    value = line.split("=", 1)[1].strip()
    if value.startswith('"') and value.endswith('"'):
        return value[1:-1]
    return value
