from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pronunciation_backend.models import EncodedFrames, LexiconEntry, PreparedAudio
from pronunciation_backend.services.phone_ctc_aligner import (
    FrameEmissions,
    PhoneCtcAligner,
    PhoneVocabulary,
    decode_monotonic_phone_segments,
    decode_phone_ctc_spans,
)


def _encoded_frames(frame_count: int = 8) -> EncodedFrames:
    return EncodedFrames(
        embeddings=np.ones((frame_count, 4), dtype=np.float32),
        frame_ms=10.0,
        energy=np.ones((frame_count,), dtype=np.float32),
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
    )


def _entry() -> LexiconEntry:
    return LexiconEntry(
        word="cat",
        phones=["K", "AE", "T"],
        ipa="kæt",
    )


def test_decode_monotonic_phone_segments_follows_synthetic_emissions() -> None:
    vocabulary = PhoneVocabulary(labels=("<blank>", "K", "AE", "T"))
    probabilities = np.full((8, 4), 0.02, dtype=np.float32)
    probabilities[0:2, vocabulary.id_for_phone("K")] = 0.92
    probabilities[2:5, vocabulary.id_for_phone("AE")] = 0.91
    probabilities[5:8, vocabulary.id_for_phone("T")] = 0.93
    probabilities = probabilities / probabilities.sum(axis=1, keepdims=True)

    segments = decode_monotonic_phone_segments(
        FrameEmissions(probabilities),
        ["K", "AE", "T"],
        vocabulary,
    )

    assert [(segment.phoneme, segment.start_frame, segment.end_frame) for segment in segments] == [
        ("K", 0, 2),
        ("AE", 2, 5),
        ("T", 5, 8),
    ]
    assert all(segment.confidence > 0.9 for segment in segments)


def test_decode_phone_ctc_spans_maps_segments_to_existing_phone_span_contract() -> None:
    vocabulary = PhoneVocabulary(labels=("<blank>", "K", "AE", "T"))
    log_probs = np.log(
        np.asarray(
            [
                [0.01, 0.96, 0.02, 0.01],
                [0.01, 0.95, 0.03, 0.01],
                [0.01, 0.03, 0.95, 0.01],
                [0.01, 0.02, 0.96, 0.01],
                [0.01, 0.02, 0.95, 0.02],
                [0.01, 0.02, 0.03, 0.94],
                [0.01, 0.01, 0.03, 0.95],
                [0.01, 0.01, 0.02, 0.96],
            ],
            dtype=np.float32,
        )
    )

    spans = decode_phone_ctc_spans(
        FrameEmissions(log_probs, log_probabilities=True),
        _entry(),
        _encoded_frames(),
        vocabulary,
    )

    assert [span.phoneme for span in spans] == ["K", "AE", "T"]
    assert [span.start_frame for span in spans] == [0, 2, 5]
    assert [span.end_frame for span in spans] == [2, 5, 8]
    assert [span.start_ms for span in spans] == [0, 20, 50]
    assert [span.end_ms for span in spans] == [20, 50, 80]
    assert all(span.alignment_confidence >= 0.9 for span in spans)


def test_phone_ctc_aligner_requires_inference_model(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "phone_ctc.pt"
    checkpoint_path.write_bytes(b"placeholder")
    aligner = PhoneCtcAligner(
        checkpoint_path=checkpoint_path,
        vocabulary=PhoneVocabulary.from_phones(["K", "AE", "T"]),
    )

    with pytest.raises(NotImplementedError, match="Phone CTC inference is not implemented"):
        aligner.align(_entry(), _prepared_audio(), _encoded_frames())
