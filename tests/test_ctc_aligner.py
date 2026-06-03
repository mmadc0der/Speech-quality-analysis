from __future__ import annotations

import numpy as np
import pytest
import torch
import torchaudio.functional._alignment as alignment
from pathlib import Path

from pronunciation_backend.models import EncodedFrames, LexiconEntry, PreparedAudio
from pronunciation_backend.services.ctc_aligner import CtcForcedAligner
from pronunciation_backend.services.mfa_aligner import AlignmentExecutionError, AlignmentResultError


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


class FakeTokenizer:
    def get_vocab(self) -> dict[str, int]:
        return {
            "<pad>": 111,
            "<unk>": 112,
            "k": 14,
            "æ": 31,
            "t": 23,
            "ˈæ": 67,
        }

    @property
    def pad_token_id(self) -> int:
        return 111


class FakeProcessor:
    def __init__(self) -> None:
        self.tokenizer = FakeTokenizer()

    def __call__(self, samples: np.ndarray, sampling_rate: int, return_tensors: str) -> FakeInputs:
        return FakeInputs(samples)


class FakeInputs:
    def __init__(self, samples: np.ndarray) -> None:
        self.input_values = torch.zeros((1, 10))


class FakeModelOutput:
    def __init__(self, num_frames: int, num_classes: int) -> None:
        # Create logits of shape (1, num_frames, num_classes)
        self.logits = torch.zeros((1, num_frames, num_classes))


class FakeModel:
    def __init__(self) -> None:
        pass

    def eval(self) -> None:
        pass

    def to(self, device: str) -> FakeModel:
        return self

    def __call__(self, input_values: torch.Tensor) -> FakeModelOutput:
        # Return logits of shape (1, 10, 115)
        return FakeModelOutput(10, 115)


def test_ctc_aligner_aligns_successfully(monkeypatch: pytest.MonkeyPatch) -> None:
    aligner = CtcForcedAligner(
        model_id="bobboyms/wav2vec2-base-en-phoneme-ctc-41h",
        device="cpu",
    )

    # Mock transformers AutoProcessor and AutoModelForCTC
    monkeypatch.setattr("pronunciation_backend.services.ctc_aligner.AutoProcessor.from_pretrained", lambda x: FakeProcessor())
    monkeypatch.setattr("pronunciation_backend.services.ctc_aligner.AutoModelForCTC.from_pretrained", lambda x: FakeModel())

    # Mock torchaudio.functional.forced_align and merge_tokens
    def _fake_forced_align(log_probs: torch.Tensor, targets: torch.Tensor, blank: int):
        # Return aligned tokens of shape (1, 10) and scores of shape (1, 10)
        return torch.tensor([[14, 14, 14, 67, 67, 67, 67, 23, 23, 23]]), torch.zeros((1, 10))

    def _fake_merge_tokens(tokens: torch.Tensor, scores: torch.Tensor, blank: int):
        return [
            alignment.TokenSpan(token=14, start=0, end=3, score=torch.tensor(-0.1)),
            alignment.TokenSpan(token=67, start=3, end=7, score=torch.tensor(-0.05)),
            alignment.TokenSpan(token=23, start=7, end=10, score=torch.tensor(-0.2)),
        ]

    monkeypatch.setattr("torchaudio.functional.forced_align", _fake_forced_align)
    monkeypatch.setattr("torchaudio.functional.merge_tokens", _fake_merge_tokens)

    spans = aligner.align(_entry(), _prepared_audio(), _encoded_frames())

    assert [span.phoneme for span in spans] == ["K", "AE", "T"]
    assert [span.start_ms for span in spans] == [0, 300, 700]
    assert [span.end_ms for span in spans] == [300, 700, 1000]
    assert [span.start_frame for span in spans] == [0, 3, 7]
    assert [span.end_frame for span in spans] == [3, 7, 10]
    assert spans[0].alignment_confidence == pytest.approx(0.905, abs=1e-3)  # exp(-0.1) = 0.9048


def test_ctc_aligner_raises_execution_error_on_runtime_error(monkeypatch: pytest.MonkeyPatch) -> None:
    aligner = CtcForcedAligner(
        model_id="bobboyms/wav2vec2-base-en-phoneme-ctc-41h",
        device="cpu",
    )

    monkeypatch.setattr("pronunciation_backend.services.ctc_aligner.AutoProcessor.from_pretrained", lambda x: FakeProcessor())
    monkeypatch.setattr("pronunciation_backend.services.ctc_aligner.AutoModelForCTC.from_pretrained", lambda x: FakeModel())

    def _fake_forced_align_error(log_probs: torch.Tensor, targets: torch.Tensor, blank: int):
        raise RuntimeError("Alignment failed due to sequence length")

    monkeypatch.setattr("torchaudio.functional.forced_align", _fake_forced_align_error)

    with pytest.raises(AlignmentExecutionError, match="CTC forced alignment failed"):
        aligner.align(_entry(), _prepared_audio(), _encoded_frames())


def test_ctc_aligner_raises_result_error_on_mismatched_spans(monkeypatch: pytest.MonkeyPatch) -> None:
    aligner = CtcForcedAligner(
        model_id="bobboyms/wav2vec2-base-en-phoneme-ctc-41h",
        device="cpu",
    )

    monkeypatch.setattr("pronunciation_backend.services.ctc_aligner.AutoProcessor.from_pretrained", lambda x: FakeProcessor())
    monkeypatch.setattr("pronunciation_backend.services.ctc_aligner.AutoModelForCTC.from_pretrained", lambda x: FakeModel())

    def _fake_forced_align(log_probs: torch.Tensor, targets: torch.Tensor, blank: int):
        return torch.tensor([[14, 14, 14, 23, 23, 23, 23, 23, 23, 23]]), torch.zeros((1, 10))

    def _fake_merge_tokens_mismatched(tokens: torch.Tensor, scores: torch.Tensor, blank: int):
        # Returns only 2 spans instead of 3
        return [
            alignment.TokenSpan(token=14, start=0, end=3, score=torch.tensor(-0.1)),
            alignment.TokenSpan(token=23, start=3, end=10, score=torch.tensor(-0.2)),
        ]

    monkeypatch.setattr("torchaudio.functional.forced_align", _fake_forced_align)
    monkeypatch.setattr("torchaudio.functional.merge_tokens", _fake_merge_tokens_mismatched)

    with pytest.raises(AlignmentResultError, match="expected 3"):
        aligner.align(_entry(), _prepared_audio(), _encoded_frames())
