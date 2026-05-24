from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import pytest

from pronunciation_backend.models import EncodedFrames, LexiconEntry, PreparedAudio
from pronunciation_backend.services.mfa_aligner import (
    AlignmentExecutionError,
    AlignmentUnavailableError,
    MfaForcedAligner,
)


def _prepared_audio() -> PreparedAudio:
    return PreparedAudio(
        samples=np.zeros((16_000,), dtype=np.float32),
        sample_rate=16_000,
        duration_ms=1000,
        rms=0.1,
        clipping_ratio=0.0,
        silence_ratio=0.0,
        snr_estimate=20.0,
        quality_status="ok",
        original_duration_ms=1000,
        trim_start_ms=0,
        trim_end_ms=1000,
        trim_applied=False,
    )


def _encoded_frames() -> EncodedFrames:
    return EncodedFrames(
        embeddings=np.ones((10, 4), dtype=np.float32),
        frame_ms=100.0,
        energy=np.ones((10,), dtype=np.float32),
    )


def _entry() -> LexiconEntry:
    return LexiconEntry(
        word="cat",
        phones=["K", "AE", "T"],
        ipa="kæt",
        reference_audio_id="cat_en_us_01",
        syllables=[["K", "AE", "T"]],
        stress_pattern="1",
    )


def test_mfa_aligner_requires_command(tmp_path: Path) -> None:
    aligner = MfaForcedAligner(
        command=None,
        acoustic_model="english_us_arpa",
        work_root=tmp_path,
    )

    with pytest.raises(AlignmentUnavailableError, match="PRONUNCIATION_MFA_COMMAND"):
        aligner.align(_entry(), _prepared_audio(), _encoded_frames())


def test_mfa_aligner_maps_timeout_to_execution_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    aligner = MfaForcedAligner(
        command="micromamba run -n mfa mfa",
        acoustic_model="english_us_arpa",
        work_root=tmp_path,
        timeout_seconds=1.5,
    )

    def _fake_run(*args, **kwargs):  # type: ignore[no-untyped-def]
        del args, kwargs
        raise subprocess.TimeoutExpired(cmd="mfa", timeout=1.5)

    monkeypatch.setattr(subprocess, "run", _fake_run)

    with pytest.raises(AlignmentExecutionError, match="timed out"):
        aligner.align(_entry(), _prepared_audio(), _encoded_frames())


def test_mfa_aligner_invokes_subprocess_and_parses_textgrid(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    aligner = MfaForcedAligner(
        command="micromamba run -n mfa mfa",
        acoustic_model="english_us_arpa",
        work_root=tmp_path,
        timeout_seconds=5.0,
    )

    textgrid = """File type = \"ooTextFile\"
Object class = \"TextGrid\"

xmin = 0
xmax = 0.64
tiers? <exists>
size = 2
item []:
    item [1]:
        class = \"IntervalTier\"
        name = \"words\"
        xmin = 0
        xmax = 0.64
        intervals: size = 1
        intervals [1]:
            xmin = 0
            xmax = 0.64
            text = \"cat\"
    item [2]:
        class = \"IntervalTier\"
        name = \"phones\"
        xmin = 0
        xmax = 0.64
        intervals: size = 3
        intervals [1]:
            xmin = 0
            xmax = 0.16
            text = \"K\"
        intervals [2]:
            xmin = 0.16
            xmax = 0.49
            text = \"AE1\"
        intervals [3]:
            xmin = 0.49
            xmax = 0.64
            text = \"T\"
"""

    def _fake_run(args, capture_output, text, timeout, check, env):  # type: ignore[no-untyped-def]
        assert args[:5] == ["micromamba", "run", "-n", "mfa", "mfa"]
        assert args[5] == "align"
        assert args[6] == "--clean"
        assert args[7] == "--temporary_directory"
        assert capture_output is True
        assert text is True
        assert timeout == 5.0
        assert check is False
        assert isinstance(env, dict)

        assert args[8].endswith("mfa_temp")
        corpus_dir = Path(args[9])
        dict_path = Path(args[10])
        assert args[11] == "english_us_arpa"
        output_dir = Path(args[12])

        assert (corpus_dir / "utterance.lab").read_text(encoding="utf-8").strip() == "cat"
        assert "cat K AE1 T" in dict_path.read_text(encoding="utf-8")

        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "utterance.TextGrid").write_text(textgrid, encoding="utf-8")
        return subprocess.CompletedProcess(args=args, returncode=0, stdout="ok", stderr="")

    monkeypatch.setattr(subprocess, "run", _fake_run)

    spans = aligner.align(_entry(), _prepared_audio(), _encoded_frames())

    assert [span.phoneme for span in spans] == ["K", "AE", "T"]
    assert [span.start_ms for span in spans] == [0, 160, 490]
    assert [span.end_ms for span in spans] == [160, 490, 640]
    assert [span.start_frame for span in spans] == [0, 1, 4]
    assert [span.end_frame for span in spans] == [2, 5, 7]


def test_mfa_aligner_falls_back_to_full_phone_tier_when_word_bounds_miss_phones(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    aligner = MfaForcedAligner(
        command="micromamba run -n mfa mfa",
        acoustic_model="english_us_arpa",
        work_root=tmp_path,
        timeout_seconds=5.0,
    )

    textgrid = """File type = \"ooTextFile\"
Object class = \"TextGrid\"

xmin = 0
xmax = 0.64
tiers? <exists>
size = 2
item []:
    item [1]:
        class = \"IntervalTier\"
        name = \"words\"
        xmin = 0
        xmax = 0.64
        intervals: size = 1
        intervals [1]:
            xmin = 0.20
            xmax = 0.44
            text = \"cat\"
    item [2]:
        class = \"IntervalTier\"
        name = \"phones\"
        xmin = 0
        xmax = 0.64
        intervals: size = 3
        intervals [1]:
            xmin = 0.00
            xmax = 0.16
            text = \"K\"
        intervals [2]:
            xmin = 0.16
            xmax = 0.49
            text = \"AE1\"
        intervals [3]:
            xmin = 0.49
            xmax = 0.64
            text = \"T\"
"""

    def _fake_run(args, capture_output, text, timeout, check, env):  # type: ignore[no-untyped-def]
        del capture_output, text, timeout, check, env
        output_dir = Path(args[12])
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "utterance.TextGrid").write_text(textgrid, encoding="utf-8")
        return subprocess.CompletedProcess(args=args, returncode=0, stdout="ok", stderr="")

    monkeypatch.setattr(subprocess, "run", _fake_run)

    spans = aligner.align(_entry(), _prepared_audio(), _encoded_frames())

    assert [span.phoneme for span in spans] == ["K", "AE", "T"]
    assert [span.start_ms for span in spans] == [0, 160, 490]
    assert [span.end_ms for span in spans] == [160, 490, 640]


def test_mfa_aligner_uses_baked_dictionary_without_clean(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    baked_dictionary = tmp_path / "runtime.dict"
    baked_dictionary.write_text("cat K AE1 T\n", encoding="utf-8")
    aligner = MfaForcedAligner(
        command="micromamba run -n mfa mfa",
        acoustic_model="english_us_arpa",
        work_root=tmp_path,
        timeout_seconds=5.0,
        runtime_dictionary_path=baked_dictionary,
        clean=False,
    )

    def _fake_run(args, capture_output, text, timeout, check, env):  # type: ignore[no-untyped-def]
        assert args[:5] == ["micromamba", "run", "-n", "mfa", "mfa"]
        assert args[5] == "align"
        assert args[6] == "--no_clean"
        assert args[7] == "--temporary_directory"
        assert Path(args[10]).name == "runtime.dict"
        output_dir = Path(args[12])
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "utterance.TextGrid").write_text(
            """File type = \"ooTextFile\"
Object class = \"TextGrid\"

xmin = 0
xmax = 0.64
tiers? <exists>
size = 2
item []:
    item [1]:
        class = \"IntervalTier\"
        name = \"words\"
        xmin = 0
        xmax = 0.64
        intervals: size = 1
        intervals [1]:
            xmin = 0
            xmax = 0.64
            text = \"cat\"
    item [2]:
        class = \"IntervalTier\"
        name = \"phones\"
        xmin = 0
        xmax = 0.64
        intervals: size = 3
        intervals [1]:
            xmin = 0.00
            xmax = 0.16
            text = \"K\"
        intervals [2]:
            xmin = 0.16
            xmax = 0.49
            text = \"AE1\"
        intervals [3]:
            xmin = 0.49
            xmax = 0.64
            text = \"T\"
""",
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(args=args, returncode=0, stdout="ok", stderr="")

    monkeypatch.setattr(subprocess, "run", _fake_run)

    spans = aligner.align(_entry(), _prepared_audio(), _encoded_frames())

    assert [span.phoneme for span in spans] == ["K", "AE", "T"]


def test_mfa_aligner_builds_stressed_dictionary_from_syllables() -> None:
    from pronunciation_backend.services.mfa_dictionary import alignment_dictionary_phones

    entry = LexiconEntry(
        word="banana",
        phones=["B", "AH", "N", "AE", "N", "AH"],
        ipa="bənænə",
        reference_audio_id="banana_en_us_01",
        syllables=[["B", "AH"], ["N", "AE"], ["N", "AH"]],
        stress_pattern="010",
    )

    assert alignment_dictionary_phones(entry) == ["B", "AH0", "N", "AE1", "N", "AH0"]


def test_mfa_aligner_prefers_explicit_alignment_phones() -> None:
    from pronunciation_backend.services.mfa_dictionary import alignment_dictionary_phones

    entry = LexiconEntry(
        word="work",
        phones=["W", "ER", "K"],
        ipa="wɝk",
        alignment_phones=["W", "ER1", "K"],
    )

    assert alignment_dictionary_phones(entry) == ["W", "ER1", "K"]
