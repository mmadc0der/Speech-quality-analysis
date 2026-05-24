from __future__ import annotations

import logging
import os
import shlex
import subprocess
import tempfile
from time import perf_counter
from dataclasses import dataclass
from math import ceil, floor
from pathlib import Path

import soundfile as sf

from pronunciation_backend.models import EncodedFrames, LexiconEntry, PhoneSpan, PreparedAudio
from pronunciation_backend.services.aligner import phone_duration_weight
from pronunciation_backend.training.cmudict_utils import normalize_word_token, strip_phone_stress
from pronunciation_backend.training.textgrid_utils import Interval, parse_textgrid

SKIP_PHONE_LABELS = {"", "sp", "sil", "spn", "<eps>"}
ARPABET_VOWELS = {
    "AA",
    "AE",
    "AH",
    "AO",
    "AW",
    "AY",
    "EH",
    "ER",
    "EY",
    "IH",
    "IY",
    "OW",
    "OY",
    "UH",
    "UW",
}
MFA_ALIGNMENT_CONFIDENCE = 0.92
logger = logging.getLogger(__name__)


class AlignmentError(RuntimeError):
    """Base error for runtime alignment failures."""


class AlignmentUnavailableError(AlignmentError):
    """Raised when MFA is not configured or cannot be launched."""


class AlignmentExecutionError(AlignmentError):
    """Raised when MFA exits unsuccessfully."""


class AlignmentResultError(AlignmentError):
    """Raised when MFA output cannot be mapped into scorer spans."""


@dataclass(frozen=True)
class AlignmentTimings:
    total_ms: float
    subprocess_ms: float
    parse_ms: float
    mapping_ms: float


@dataclass
class MfaForcedAligner:
    command: str | None
    acoustic_model: str | None
    work_root: Path
    timeout_seconds: float = 30.0
    runtime_dictionary_path: Path | None = None
    clean: bool = True
    word_tier: str = "words"
    phone_tier: str = "phones"

    def align(self, entry: LexiconEntry, prepared: PreparedAudio, encoded: EncodedFrames) -> list[PhoneSpan]:
        spans, _timings = self.align_with_timing(entry, prepared, encoded)
        return spans

    def align_with_timing(
        self,
        entry: LexiconEntry,
        prepared: PreparedAudio,
        encoded: EncodedFrames,
    ) -> tuple[list[PhoneSpan], AlignmentTimings]:
        start = perf_counter()
        command_argv = self._command_argv()
        acoustic_model = self._acoustic_model()
        transcript_token = normalize_word_token(entry.word)
        if not transcript_token:
            raise AlignmentResultError(f"Unable to normalize runtime transcript for word: {entry.word!r}")

        self.work_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="mfa-align-", dir=self.work_root) as temp_dir:
            base_dir = Path(temp_dir)
            corpus_dir = base_dir / "corpus"
            output_dir = base_dir / "output"
            temp_mfa_dir = base_dir / "mfa_temp"
            corpus_dir.mkdir(parents=True, exist_ok=True)
            temp_mfa_dir.mkdir(parents=True, exist_ok=True)

            stem = "utterance"
            wav_path = corpus_dir / f"{stem}.wav"
            lab_path = corpus_dir / f"{stem}.lab"
            dict_path = self._dictionary_path(base_dir, entry)

            sf.write(wav_path, prepared.samples, prepared.sample_rate)
            lab_path.write_text(transcript_token + "\n", encoding="utf-8")

            subprocess_started = perf_counter()
            process = self._run_mfa(
                command_argv=command_argv,
                corpus_dir=corpus_dir,
                dictionary_path=dict_path,
                acoustic_model=acoustic_model,
                output_dir=output_dir,
                temp_mfa_dir=temp_mfa_dir,
            )
            subprocess_ms = (perf_counter() - subprocess_started) * 1000.0
            if process.returncode != 0:
                logger.error(
                    "MFA alignment subprocess failed",
                    extra={
                        "returncode": process.returncode,
                        "stdout": process.stdout,
                        "stderr": process.stderr,
                    },
                )
                raise AlignmentExecutionError(
                    "MFA alignment failed "
                    f"(exit_code={process.returncode}): {self._format_process_output(process.stdout, process.stderr)}"
                )

            parse_started = perf_counter()
            textgrid_path = self._locate_textgrid(output_dir)
            mapping_started = perf_counter()
            spans = self._spans_from_textgrid(
                textgrid_path=textgrid_path,
                transcript_token=transcript_token,
                canonical_phones=entry.phones,
                encoded=encoded,
            )
            mapping_ms = (perf_counter() - mapping_started) * 1000.0
            parse_ms = (mapping_started - parse_started) * 1000.0
            total_ms = (perf_counter() - start) * 1000.0
            return spans, AlignmentTimings(
                total_ms=total_ms,
                subprocess_ms=subprocess_ms,
                parse_ms=parse_ms,
                mapping_ms=mapping_ms,
            )

    def _command_argv(self) -> list[str]:
        if self.command is None:
            raise AlignmentUnavailableError(
                "MFA aligner is not configured. Set PRONUNCIATION_MFA_COMMAND to an explicit MFA launcher command."
            )
        argv = shlex.split(self.command, posix=os.name != "nt")
        if not argv:
            raise AlignmentUnavailableError(
                "PRONUNCIATION_MFA_COMMAND is empty. Provide a command that resolves to the MFA executable."
            )
        return argv

    def _acoustic_model(self) -> str:
        if self.acoustic_model is None:
            raise AlignmentUnavailableError(
                "MFA acoustic model is not configured. Set PRONUNCIATION_MFA_ACOUSTIC_MODEL."
            )
        return self.acoustic_model

    def _run_mfa(
        self,
        *,
        command_argv: list[str],
        corpus_dir: Path,
        dictionary_path: Path,
        acoustic_model: str,
        output_dir: Path,
        temp_mfa_dir: Path,
    ) -> subprocess.CompletedProcess[str]:
        args = list(command_argv)
        if self.clean:
            args.append("--clean")
        args.extend(
            [
                "align",
                "--temporary_directory",
                str(temp_mfa_dir),
                str(corpus_dir),
                str(dictionary_path),
                acoustic_model,
                str(output_dir),
            ]
        )
        try:
            return subprocess.run(
                args,
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
                check=False,
                env=os.environ.copy(),
            )
        except FileNotFoundError as exc:
            raise AlignmentUnavailableError(
                "Unable to launch MFA. Check PRONUNCIATION_MFA_COMMAND and the referenced executable path."
            ) from exc
        except subprocess.TimeoutExpired as exc:
            raise AlignmentExecutionError(
                f"MFA alignment timed out after {self.timeout_seconds:.1f}s."
            ) from exc

    def preflight(self, entry: LexiconEntry, prepared: PreparedAudio, encoded: EncodedFrames) -> AlignmentTimings:
        _, timings = self.align_with_timing(entry, prepared, encoded)
        return timings

    def _locate_textgrid(self, output_dir: Path) -> Path:
        candidates = sorted(output_dir.rglob("*.TextGrid"))
        if len(candidates) != 1:
            raise AlignmentResultError(
                f"Expected exactly one MFA TextGrid output, found {len(candidates)} under {output_dir}."
            )
        return candidates[0]

    def _dictionary_path(self, base_dir: Path, entry: LexiconEntry) -> Path:
        if self.runtime_dictionary_path is not None:
            if not self.runtime_dictionary_path.exists():
                raise AlignmentUnavailableError(
                    f"Configured runtime MFA dictionary does not exist: {self.runtime_dictionary_path}"
                )
            return self.runtime_dictionary_path

        dict_path = base_dir / "lexicon.dict"
        dictionary_phones = self._dictionary_phones(entry)
        dict_path.write_text(f"{normalize_word_token(entry.word)} {' '.join(dictionary_phones)}\n", encoding="utf-8")
        return dict_path

    def _spans_from_textgrid(
        self,
        *,
        textgrid_path: Path,
        transcript_token: str,
        canonical_phones: list[str],
        encoded: EncodedFrames,
    ) -> list[PhoneSpan]:
        tiers = parse_textgrid(textgrid_path)
        logger.info("Parsed MFA TextGrid tiers: %s", sorted(tiers.keys()))
        word_intervals = self._select_tier(tiers, self.word_tier, ("word",))
        phone_intervals = self._select_tier(tiers, self.phone_tier, ("phone",))

        matching_words = [
            interval
            for interval in word_intervals
            if normalize_word_token(interval.text) == transcript_token
        ]
        if len(matching_words) != 1:
            raise AlignmentResultError(
                f"Expected exactly one aligned word interval for {transcript_token!r}, found {len(matching_words)}. "
                f"raw_word_intervals={self._format_intervals(word_intervals)} "
                f"raw_phone_intervals={self._format_intervals(phone_intervals)}"
            )
        word_interval = matching_words[0]

        aligned_phones = [
            interval
            for interval in phone_intervals
            if interval.xmin >= word_interval.xmin
            and interval.xmax <= word_interval.xmax
            and strip_phone_stress(interval.text).lower() not in SKIP_PHONE_LABELS
        ]
        all_non_skip_phones = [
            interval
            for interval in phone_intervals
            if strip_phone_stress(interval.text).lower() not in SKIP_PHONE_LABELS
        ]
        observed_phones = [strip_phone_stress(interval.text) for interval in aligned_phones]
        expected_phones = [strip_phone_stress(phone) for phone in canonical_phones]
        if observed_phones != expected_phones:
            fallback_phones = [strip_phone_stress(interval.text) for interval in all_non_skip_phones]
            if fallback_phones == expected_phones:
                logger.warning(
                    "MFA word-tier phone selection missed canonical phones; using full phone tier fallback",
                    extra={
                        "word": transcript_token,
                        "word_interval": (word_interval.xmin, word_interval.xmax),
                        "bounded_phones": observed_phones,
                        "full_tier_phones": fallback_phones,
                    },
                )
                aligned_phones = all_non_skip_phones
                observed_phones = fallback_phones
        if observed_phones != expected_phones:
            raise AlignmentResultError(
                "MFA phone sequence mismatch: "
                f"expected={expected_phones} observed={observed_phones} "
                f"word_interval=({word_interval.xmin:.3f}, {word_interval.xmax:.3f}) "
                f"all_non_skip={[strip_phone_stress(interval.text) for interval in all_non_skip_phones]} "
                f"raw_phone_intervals={self._format_intervals(phone_intervals)}"
            )

        return self._intervals_to_spans(intervals=aligned_phones, phones=expected_phones, encoded=encoded)

    def _intervals_to_spans(
        self,
        *,
        intervals: list[Interval],
        phones: list[str],
        encoded: EncodedFrames,
    ) -> list[PhoneSpan]:
        frame_count = max(1, len(encoded.embeddings))
        frame_ms = max(encoded.frame_ms, 1e-6)
        expected_weights = [phone_duration_weight(phone) for phone in phones]
        expected_total = max(sum(expected_weights), 1e-6)

        spans: list[PhoneSpan] = []
        for index, interval in enumerate(intervals):
            start_ms = int(round(interval.xmin * 1000.0))
            end_ms = max(start_ms + 1, int(round(interval.xmax * 1000.0)))
            start_frame = max(0, min(frame_count - 1, int(floor(start_ms / frame_ms))))
            end_frame = max(start_frame + 1, int(ceil(end_ms / frame_ms)))
            end_frame = min(frame_count, end_frame)
            observed_frames = max(1, end_frame - start_frame)
            expected_frames = max(1.0, frame_count * (expected_weights[index] / expected_total))
            duration_z = (observed_frames - expected_frames) / max(1.0, expected_frames * 0.35)
            spans.append(
                PhoneSpan(
                    phoneme=phones[index],
                    start_frame=start_frame,
                    end_frame=end_frame,
                    start_ms=start_ms,
                    end_ms=end_ms,
                    alignment_confidence=MFA_ALIGNMENT_CONFIDENCE,
                    duration_z_score=round(float(duration_z), 3),
                )
            )
        return spans

    def _select_tier(
        self,
        tiers: dict[str, object],
        preferred: str,
        fallbacks: tuple[str, ...],
    ) -> list[Interval]:
        candidates = (preferred, *fallbacks)
        for name in candidates:
            tier = tiers.get(name)
            if tier is not None:
                return tier.intervals  # type: ignore[return-value]
        lowered = {key.lower(): value for key, value in tiers.items()}
        for name in candidates:
            tier = lowered.get(name.lower())
            if tier is not None:
                return tier.intervals  # type: ignore[return-value]
        raise AlignmentResultError(
            f"Could not find tier {preferred!r} in MFA TextGrid output. Available tiers: {sorted(tiers.keys())}"
        )

    def _format_process_output(self, stdout: str, stderr: str) -> str:
        combined = " | ".join(part.strip() for part in (stderr, stdout) if part and part.strip())
        if not combined:
            return "no process output"
        return combined[:500]

    def _dictionary_phones(self, entry: LexiconEntry) -> list[str]:
        if not entry.syllables or not entry.stress_pattern:
            return list(entry.phones)

        flattened = [phone for syllable in entry.syllables for phone in syllable]
        if [strip_phone_stress(phone) for phone in flattened] != [strip_phone_stress(phone) for phone in entry.phones]:
            return list(entry.phones)

        stressed: list[str] = []
        for syllable_index, syllable in enumerate(entry.syllables):
            stress_digit = self._stress_digit(entry.stress_pattern, syllable_index)
            for phone in syllable:
                base_phone = strip_phone_stress(phone)
                if base_phone in ARPABET_VOWELS:
                    stressed.append(f"{base_phone}{stress_digit}")
                else:
                    stressed.append(base_phone)
        return stressed

    def _stress_digit(self, stress_pattern: str, syllable_index: int) -> str:
        if syllable_index >= len(stress_pattern):
            return "0"
        digit = stress_pattern[syllable_index]
        return digit if digit in {"0", "1", "2"} else "0"

    def _format_intervals(self, intervals: list[Interval]) -> str:
        if not intervals:
            return "[]"
        formatted = [
            f"{interval.text!r}@({interval.xmin:.3f},{interval.xmax:.3f})"
            for interval in intervals
        ]
        return "[" + ", ".join(formatted[:12]) + (", ..." if len(formatted) > 12 else "") + "]"
