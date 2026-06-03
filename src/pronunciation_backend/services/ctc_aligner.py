from __future__ import annotations

import logging
import re
import torch
import torchaudio.functional as F
from dataclasses import dataclass
from pathlib import Path
from math import ceil, floor
import numpy as np

from transformers import AutoProcessor, AutoModelForCTC

from pronunciation_backend.models import EncodedFrames, LexiconEntry, PhoneSpan, PreparedAudio
from pronunciation_backend.services.aligner import phone_duration_weight
from pronunciation_backend.services.mfa_aligner import AlignmentExecutionError, AlignmentResultError

logger = logging.getLogger(__name__)

# ARPABET to IPA mapping for bobboyms/wav2vec2-base-en-phoneme-ctc-41h
ARPABET_TO_CTC_VOCAB = {
    "AA": "ɑː",
    "AE": "æ",
    "AH": "ʌ",
    "AO": "ɔː",
    "AW": "aʊ",
    "AY": "aɪ",
    "B": "b",
    "CH": "tʃ",
    "D": "d",
    "DH": "ð",
    "EH": "ɛ",
    "ER": "ɚ",
    "EY": "eɪ",
    "F": "f",
    "G": "ɡ",  # script g (U+0261)
    "HH": "h",
    "IH": "ɪ",
    "IY": "iː",
    "JH": "dʒ",
    "K": "k",
    "L": "l",
    "M": "m",
    "N": "n",
    "NG": "ŋ",
    "OW": "oʊ",
    "OY": "ɔɪ",
    "P": "p",
    "R": "ɹ",
    "S": "s",
    "SH": "ʃ",
    "T": "t",
    "TH": "θ",
    "UH": "ʊ",
    "UW": "uː",
    "V": "v",
    "W": "w",
    "Y": "j",
    "Z": "z",
    "ZH": "ʒ",
}


@dataclass
class AlignedPhone:
    phone: str
    start_ms: int
    end_ms: int
    score: float


class CtcForcedAligner:
    def __init__(
        self,
        model_id: str,
        device: str,
    ):
        self.model_id = model_id
        self.device = device
        self._processor = None
        self._model = None

    def _ensure_model(self) -> None:
        if self._processor is not None and self._model is not None:
            return
        logger.info("Loading CTC forced aligner model: %s on %s", self.model_id, self.device)
        self._processor = AutoProcessor.from_pretrained(self.model_id)
        self._model = AutoModelForCTC.from_pretrained(self.model_id)
        self._model.eval()
        self._model.to(self.device)

    def align_audio(self, samples: np.ndarray, sample_rate: int, arpabet_phones: list[str]) -> list[AlignedPhone]:
        self._ensure_model()
        if not arpabet_phones:
            return []

        vocab = self._processor.tokenizer.get_vocab()
        blank_id = self._processor.tokenizer.pad_token_id
        if blank_id is None:
            blank_id = 111  # fallback for the bobboyms model

        # Map ARPABET phones to CTC token IDs
        token_ids = []
        for phone in arpabet_phones:
            match = re.match(r"^([A-Z]+)([0-2])?$", phone.upper())
            if not match:
                base, stress = phone.upper(), None
            else:
                base, stress = match.groups()

            ipa = ARPABET_TO_CTC_VOCAB.get(base, base.lower())
            token = ipa
            if stress == "1":
                stressed = "ˈ" + ipa
                if stressed in vocab:
                    token = stressed
            elif stress == "2":
                stressed = "ˌ" + ipa
                if stressed in vocab:
                    token = stressed

            if token in vocab:
                token_ids.append(vocab[token])
            elif ipa in vocab:
                token_ids.append(vocab[ipa])
            else:
                raise AlignmentResultError(f"Phoneme {phone} (mapped to {token!r}) is not in CTC vocabulary.")

        # Prepare input tensor
        samples_float = np.asarray(samples, dtype=np.float32)
        inputs = self._processor(samples_float, sampling_rate=sample_rate, return_tensors="pt")
        input_values = inputs.input_values.to(self.device)

        with torch.inference_mode():
            logits = self._model(input_values).logits
            log_probs = torch.log_softmax(logits, dim=-1)

        targets = torch.tensor([token_ids], dtype=torch.long, device=self.device)

        # Run forced alignment
        try:
            aligned_tokens, scores = F.forced_align(log_probs, targets, blank=blank_id)
        except RuntimeError as exc:
            raise AlignmentExecutionError(f"CTC forced alignment failed: {exc}") from exc
        
        # Merge tokens to spans
        spans = F.merge_tokens(aligned_tokens[0], scores[0], blank=blank_id)

        if len(spans) != len(arpabet_phones):
            raise AlignmentResultError(
                f"CTC forced alignment returned {len(spans)} spans, expected {len(arpabet_phones)}."
            )

        # Convert spans back to AlignedPhone
        aligned_phones = []
        num_frames = log_probs.shape[1]
        duration_ms = len(samples) / sample_rate * 1000.0
        frame_duration_ms = duration_ms / max(1, num_frames)

        for index, span in enumerate(spans):
            start_ms = int(round(span.start * frame_duration_ms))
            end_ms = int(round(span.end * frame_duration_ms))
            end_ms = max(start_ms + 1, end_ms)
            
            aligned_phones.append(
                AlignedPhone(
                    phone=arpabet_phones[index],
                    start_ms=start_ms,
                    end_ms=end_ms,
                    score=float(span.score.exp().item()),  # convert log prob back to probability
                )
            )
        return aligned_phones

    def align(self, entry: LexiconEntry, prepared: PreparedAudio, encoded: EncodedFrames) -> list[PhoneSpan]:
        aligned_phones = self.align_audio(prepared.samples, prepared.sample_rate, entry.phones)
        
        frame_count = max(1, len(encoded.embeddings))
        frame_ms = max(encoded.frame_ms, 1e-6)
        expected_weights = [phone_duration_weight(phone) for phone in entry.phones]
        expected_total = max(sum(expected_weights), 1e-6)

        spans: list[PhoneSpan] = []
        for index, ap in enumerate(aligned_phones):
            start_frame = max(0, min(frame_count - 1, int(floor(ap.start_ms / frame_ms))))
            end_frame = max(start_frame + 1, int(ceil(ap.end_ms / frame_ms)))
            end_frame = min(frame_count, end_frame)
            observed_frames = max(1, end_frame - start_frame)
            expected_frames = max(1.0, frame_count * (expected_weights[index] / expected_total))
            duration_z = (observed_frames - expected_frames) / max(1.0, expected_frames * 0.35)
            
            spans.append(
                PhoneSpan(
                    phoneme=ap.phone,
                    start_frame=start_frame,
                    end_frame=end_frame,
                    start_ms=ap.start_ms,
                    end_ms=ap.end_ms,
                    alignment_confidence=round(ap.score, 3),
                    duration_z_score=round(float(duration_z), 3),
                )
            )
        return spans
