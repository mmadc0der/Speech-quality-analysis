from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from pronunciation_backend.config import Settings
from pronunciation_backend.models import EncodedFrames, PreparedAudio

try:
    import torch
    from transformers import AutoFeatureExtractor, AutoModel
except ImportError:  # pragma: no cover - optional runtime
    torch = None
    AutoFeatureExtractor = None
    AutoModel = None


@dataclass
class SSLFeatureEncoder:
    """Frozen speech feature extractor for runtime and offline artifact generation."""

    settings: Settings
    _processor: object | None = field(default=None, init=False, repr=False)
    _model: object | None = field(default=None, init=False, repr=False)
    _warned_fallback: bool = field(default=False, init=False, repr=False)

    def encode(self, audio: PreparedAudio) -> EncodedFrames:
        if self.settings.use_hf_encoder and torch is not None:
            try:
                return self._encode_with_hf(audio)
            except Exception as exc:
                if not self._warned_fallback:
                    print(f"warning: HF encoder unavailable, falling back to CPU features: {exc}")
                    self._warned_fallback = True
                return self._encode_fallback(audio)
        if self.settings.use_hf_encoder and not self._warned_fallback:
            print("warning: torch/transformers unavailable, falling back to CPU features")
            self._warned_fallback = True
        return self._encode_fallback(audio)

    def _encode_with_hf(self, audio: PreparedAudio) -> EncodedFrames:
        if self._processor is None or self._model is None:
            self._processor = AutoFeatureExtractor.from_pretrained(self.settings.backbone_id)
            self._model = AutoModel.from_pretrained(self.settings.backbone_id)
            self._model.eval()
            self._model.to(self.settings.device)

        array = np.asarray(audio.samples, dtype=np.float32)
        inputs = self._processor(
            array,
            sampling_rate=audio.sample_rate,
            return_tensors="pt",
        )
        inputs = {key: value.to(self.settings.device) for key, value in inputs.items()}

        with torch.no_grad():
            outputs = self._model(**inputs)

        hidden = outputs.last_hidden_state.squeeze(0).detach().cpu().numpy()
        frame_ms = max(10.0, audio.duration_ms / max(1, hidden.shape[0]))
        energy = self._frame_energy(array, frame_ms)
        return EncodedFrames(
            embeddings=hidden.astype(np.float32).tolist(),
            frame_ms=frame_ms,
            energy=energy,
        )

    def _encode_fallback(self, audio: PreparedAudio) -> EncodedFrames:
        samples = np.asarray(audio.samples, dtype=np.float32)
        frame_size = int(0.02 * audio.sample_rate)
        hop_size = int(0.01 * audio.sample_rate)
        if len(samples) < frame_size:
            padded = np.pad(samples, (0, frame_size - len(samples)))
            samples = padded

        frames: list[list[float]] = []
        energy: list[float] = []
        for start in range(0, len(samples) - frame_size + 1, hop_size):
            window = samples[start : start + frame_size]
            spectrum = np.abs(np.fft.rfft(window))
            band_edges = np.array_split(spectrum, 8)
            embedding = [float(np.mean(band)) for band in band_edges]
            frames.append(embedding)
            energy.append(float(np.sqrt(np.mean(np.square(window)))))

        if not frames:
            frames = [[0.0] * 8]
            energy = [0.0]

        return EncodedFrames(
            embeddings=frames,
            frame_ms=(hop_size / audio.sample_rate) * 1000.0,
            energy=energy,
        )

    def _frame_energy(self, samples: np.ndarray, frame_ms: float) -> list[float]:
        frame_size = max(1, int((frame_ms / 1000.0) * self.settings.sample_rate))
        energies: list[float] = []
        for start in range(0, len(samples), frame_size):
            window = samples[start : start + frame_size]
            if len(window) == 0:
                continue
            energies.append(float(np.sqrt(np.mean(np.square(window)))))
        return energies or [0.0]
