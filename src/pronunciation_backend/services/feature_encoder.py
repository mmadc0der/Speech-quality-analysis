from __future__ import annotations

from contextlib import nullcontext
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

    def encode_many(self, audios: list[PreparedAudio]) -> list[EncodedFrames]:
        if not audios:
            return []
        if self.settings.use_hf_encoder and torch is not None:
            try:
                return self._encode_many_with_hf(audios)
            except RuntimeError as exc:
                if self._is_oom_error(exc):
                    if len(audios) == 1:
                        if self._is_cuda_device():
                            self._clear_cuda_cache()
                        if not self._warned_fallback:
                            print("warning: GPU OOM on single audio batch, falling back to CPU features for that item")
                            self._warned_fallback = True
                        return [self._encode_fallback(audios[0])]
                    if self._is_cuda_device():
                        self._clear_cuda_cache()
                    midpoint = max(1, len(audios) // 2)
                    left = self.encode_many(audios[:midpoint])
                    right = self.encode_many(audios[midpoint:])
                    return left + right
                raise
            except Exception as exc:
                if not self._warned_fallback:
                    print(f"warning: HF batch encoder unavailable, falling back to CPU features: {exc}")
                    self._warned_fallback = True
                return [self._encode_fallback(audio) for audio in audios]
        if self.settings.use_hf_encoder and not self._warned_fallback:
            print("warning: torch/transformers unavailable, falling back to CPU features")
            self._warned_fallback = True
        return [self._encode_fallback(audio) for audio in audios]

    def _encode_with_hf(self, audio: PreparedAudio) -> EncodedFrames:
        self._ensure_hf_model()

        array = np.asarray(audio.samples, dtype=np.float32)
        inputs = self._processor(
            array,
            sampling_rate=audio.sample_rate,
            return_tensors="pt",
        )
        inputs = {key: value.to(self.settings.device) for key, value in inputs.items()}

        outputs = self._forward_hf(inputs)

        hidden = outputs.last_hidden_state.squeeze(0).detach().to("cpu", dtype=torch.float32).numpy()
        del outputs
        frame_ms = max(10.0, audio.duration_ms / max(1, hidden.shape[0]))
        energy = self._frame_energy(array, frame_ms)
        return EncodedFrames(
            embeddings=hidden.astype(np.float32).tolist(),
            frame_ms=frame_ms,
            energy=energy,
        )

    def _encode_many_with_hf(self, audios: list[PreparedAudio]) -> list[EncodedFrames]:
        self._ensure_hf_model()

        arrays = [np.asarray(audio.samples, dtype=np.float32) for audio in audios]
        inputs = self._processor(
            arrays,
            sampling_rate=audios[0].sample_rate,
            return_tensors="pt",
            padding=True,
        )
        inputs = {key: value.to(self.settings.device) for key, value in inputs.items()}

        outputs = self._forward_hf(inputs)

        hidden_batch = outputs.last_hidden_state.detach().to("cpu", dtype=torch.float32)
        attention_mask = inputs.get("attention_mask")
        if attention_mask is None or not hasattr(self._model, "_get_feat_extract_output_lengths"):
            return [self._encode_with_hf(audio) for audio in audios]

        sample_lengths = attention_mask.sum(dim=-1).detach().cpu()
        output_lengths = self._model._get_feat_extract_output_lengths(sample_lengths).tolist()
        del outputs

        encoded_items: list[EncodedFrames] = []
        for index, (audio, array) in enumerate(zip(audios, arrays)):
            hidden_length = max(1, int(output_lengths[index]))
            hidden = hidden_batch[index, :hidden_length].numpy()
            frame_ms = max(10.0, audio.duration_ms / max(1, hidden.shape[0]))
            energy = self._frame_energy(array, frame_ms)
            encoded_items.append(
                EncodedFrames(
                    embeddings=hidden.astype(np.float32).tolist(),
                    frame_ms=frame_ms,
                    energy=energy,
                )
            )
        return encoded_items

    def _ensure_hf_model(self) -> None:
        if self._processor is not None and self._model is not None:
            return
        self._processor = AutoFeatureExtractor.from_pretrained(self.settings.backbone_id)
        self._model = AutoModel.from_pretrained(self.settings.backbone_id, low_cpu_mem_usage=True)
        self._model.eval()
        self._model.to(self.settings.device)

    def _forward_hf(self, inputs: dict[str, object]) -> object:
        autocast_context = (
            torch.autocast(device_type="cuda", dtype=torch.float16)
            if self._is_cuda_device()
            else nullcontext()
        )
        with torch.inference_mode():
            with autocast_context:
                return self._model(**inputs)

    def _is_cuda_device(self) -> bool:
        return torch is not None and str(self.settings.device).startswith("cuda")

    def _is_oom_error(self, exc: BaseException) -> bool:
        message = str(exc).lower()
        return "out of memory" in message or "cuda error: out of memory" in message

    def _clear_cuda_cache(self) -> None:
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()

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
