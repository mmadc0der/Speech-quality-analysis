from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass, field

import numpy as np

from pronunciation_backend.config import Settings
from pronunciation_backend.models import EncodedFrames, PhoneFeatures, PhoneSpan, PreparedAudio

try:
    import torch
    from transformers import AutoFeatureExtractor, AutoModel
except ImportError:  # pragma: no cover - optional runtime
    torch = None
    AutoFeatureExtractor = None
    AutoModel = None


@dataclass
class EncodedBatchView:
    frame_counts: list[int]
    frame_mss: list[float]
    energies: list[np.ndarray]
    hidden_batch: object | None = field(default=None, repr=False)
    encoded_items: list[EncodedFrames] | None = field(default=None, repr=False)


@dataclass
class SSLFeatureEncoder:
    """Frozen speech feature extractor for runtime and offline artifact generation."""

    settings: Settings
    _processor: object | None = field(default=None, init=False, repr=False)
    _model: object | None = field(default=None, init=False, repr=False)

    def encode(self, audio: PreparedAudio) -> EncodedFrames:
        if not self.settings.use_hf_encoder:
            raise RuntimeError("HF encoder is disabled in settings, but CPU fallback has been removed.")
        if torch is None:
            raise RuntimeError("torch or transformers not installed, but required for HF encoder.")
        return self._encode_with_hf(audio)

    def encode_many(self, audios: list[PreparedAudio]) -> list[EncodedFrames]:
        if not audios:
            return []
        if not self.settings.use_hf_encoder:
            raise RuntimeError("HF encoder is disabled in settings, but CPU fallback has been removed.")
        if torch is None:
            raise RuntimeError("torch or transformers not installed, but required for HF encoder.")
        try:
            return self._encode_many_with_hf(audios)
        except RuntimeError as exc:
            if self._is_oom_error(exc):
                if len(audios) == 1:
                    raise RuntimeError(f"GPU OOM on single audio batch (duration: {audios[0].duration_ms}ms). Audio is too long to fit in VRAM.") from exc
                if self._is_cuda_device():
                    self._clear_cuda_cache()
                midpoint = max(1, len(audios) // 2)
                return self.encode_many(audios[:midpoint]) + self.encode_many(audios[midpoint:])
            raise

    def encode_many_for_pooling(self, audios: list[PreparedAudio]) -> EncodedBatchView:
        if not audios:
            return EncodedBatchView(frame_counts=[], frame_mss=[], energies=[], encoded_items=[])
        if not self.settings.use_hf_encoder:
            raise RuntimeError("HF encoder is disabled in settings, but CPU fallback has been removed.")
        if torch is None:
            raise RuntimeError("torch or transformers not installed, but required for HF encoder.")
        try:
            return self._encode_many_for_pooling_with_hf(audios)
        except RuntimeError as exc:
            if self._is_oom_error(exc):
                if len(audios) == 1:
                    raise RuntimeError(f"GPU OOM on single audio batch (duration: {audios[0].duration_ms}ms). Audio is too long to fit in VRAM.") from exc
                if self._is_cuda_device():
                    self._clear_cuda_cache()
                midpoint = max(1, len(audios) // 2)
                left = self.encode_many_for_pooling(audios[:midpoint])
                right = self.encode_many_for_pooling(audios[midpoint:])
                return self._concat_batch_views(left, right)
            raise

    def build_phone_features(self, batch_view: EncodedBatchView, index: int, spans: list[PhoneSpan]) -> list[PhoneFeatures]:
        if batch_view.hidden_batch is not None and torch is not None:
            hidden = batch_view.hidden_batch[index, : batch_view.frame_counts[index]]
            return self._build_phone_features_gpu(hidden, batch_view.energies[index], spans, batch_view.frame_mss[index])
        if batch_view.encoded_items is None:
            raise RuntimeError("Encoded batch view does not contain CPU encodings.")
        encoded = batch_view.encoded_items[index]
        return self._build_phone_features_cpu(encoded.embeddings, encoded.energy, spans, encoded.frame_ms)

    def release_batch_view(self, batch_view: EncodedBatchView) -> None:
        batch_view.hidden_batch = None
        batch_view.encoded_items = None

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
            embeddings=hidden.astype(np.float32),
            frame_ms=frame_ms,
            energy=energy,
        )

    def _encode_many_with_hf(self, audios: list[PreparedAudio]) -> list[EncodedFrames]:
        batch_view = self._encode_many_for_pooling_with_hf(audios)
        encoded_items: list[EncodedFrames] = []
        hidden_batch = batch_view.hidden_batch
        if hidden_batch is None:
            raise RuntimeError("Expected hidden_batch for HF batch encoding.")
        for index, (frame_count, frame_ms, energy) in enumerate(zip(batch_view.frame_counts, batch_view.frame_mss, batch_view.energies)):
            hidden = hidden_batch[index, :frame_count].detach().to("cpu", dtype=torch.float32).numpy()
            encoded_items.append(
                EncodedFrames(
                    embeddings=hidden.astype(np.float32),
                    frame_ms=frame_ms,
                    energy=energy,
                )
            )
        self.release_batch_view(batch_view)
        return encoded_items

    def _encode_many_for_pooling_with_hf(self, audios: list[PreparedAudio]) -> EncodedBatchView:
        self._ensure_hf_model()

        arrays = [np.asarray(audio.samples, dtype=np.float32) for audio in audios]
        inputs = self._processor(
            arrays,
            sampling_rate=audios[0].sample_rate,
            return_tensors="pt",
            padding=True,
            return_attention_mask=True,
        )
        inputs = {key: value.to(self.settings.device) for key, value in inputs.items()}

        outputs = self._forward_hf(inputs)
        attention_mask = inputs.get("attention_mask")
        if attention_mask is None:
            attention_mask = torch.ones_like(inputs["input_values"])
            
        if not hasattr(self._model, "_get_feat_extract_output_lengths"):
            raise RuntimeError(f"Model {type(self._model)} does not support _get_feat_extract_output_lengths")

        sample_lengths = attention_mask.sum(dim=-1).detach().cpu()
        output_lengths = [max(1, int(length)) for length in self._model._get_feat_extract_output_lengths(sample_lengths).tolist()]
        frame_mss = [max(10.0, audio.duration_ms / max(1, frame_count)) for audio, frame_count in zip(audios, output_lengths)]
        energies = [self._frame_energy(array, frame_ms) for array, frame_ms in zip(arrays, frame_mss)]
        return EncodedBatchView(
            frame_counts=output_lengths,
            frame_mss=frame_mss,
            energies=energies,
            hidden_batch=outputs.last_hidden_state.detach(),
        )

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

    def _build_phone_features_gpu(
        self,
        hidden: object,
        energy: np.ndarray,
        spans: list[PhoneSpan],
        frame_ms: float,
    ) -> list[PhoneFeatures]:
        features: list[PhoneFeatures] = []
        late_threshold_ms = frame_ms * 1.5
        hidden_tensor = hidden.float()

        for index, span in enumerate(spans):
            segment = hidden_tensor[span.start_frame : span.end_frame]
            if segment.numel() == 0:
                segment = torch.zeros((1, hidden_tensor.shape[-1]), device=hidden_tensor.device, dtype=torch.float32)
            mean_embedding = segment.mean(dim=0).detach().to("cpu", dtype=torch.float32).tolist()
            variance = float(segment.var(unbiased=False).item())
            segment_energy = energy[span.start_frame : span.end_frame]
            energy_mean = float(segment_energy.mean()) if segment_energy.size > 0 else 0.0
            features.append(
                PhoneFeatures(
                    phoneme=span.phoneme,
                    start_ms=span.start_ms,
                    end_ms=span.end_ms,
                    mean_embedding=mean_embedding,
                    variance=variance,
                    duration_ms=max(1, span.end_ms - span.start_ms),
                    duration_z_score=span.duration_z_score,
                    alignment_confidence=span.alignment_confidence,
                    energy_mean=energy_mean,
                    starts_late=index > 0 and span.start_ms - spans[index - 1].end_ms > late_threshold_ms,
                )
            )
        return features

    def _build_phone_features_cpu(
        self,
        embeddings: np.ndarray,
        energy: np.ndarray,
        spans: list[PhoneSpan],
        frame_ms: float,
    ) -> list[PhoneFeatures]:
        frame_array = np.asarray(embeddings, dtype=np.float32)
        energy_array = np.asarray(energy, dtype=np.float32) if energy.size > 0 else np.zeros((1,), dtype=np.float32)
        late_threshold_ms = frame_ms * 1.5

        features: list[PhoneFeatures] = []
        for index, span in enumerate(spans):
            segment = frame_array[span.start_frame : span.end_frame]
            if segment.size == 0:
                segment = np.zeros((1, frame_array.shape[1]), dtype=np.float32)
            segment_energy = energy_array[span.start_frame : span.end_frame]
            if segment_energy.size == 0:
                segment_energy = np.zeros((1,), dtype=np.float32)
            features.append(
                PhoneFeatures(
                    phoneme=span.phoneme,
                    start_ms=span.start_ms,
                    end_ms=span.end_ms,
                    mean_embedding=segment.mean(axis=0).astype(np.float32).tolist(),
                    variance=float(segment.var()),
                    duration_ms=max(1, span.end_ms - span.start_ms),
                    duration_z_score=span.duration_z_score,
                    alignment_confidence=span.alignment_confidence,
                    energy_mean=float(segment_energy.mean()),
                    starts_late=index > 0 and span.start_ms - spans[index - 1].end_ms > late_threshold_ms,
                )
            )
        return features

    def _concat_batch_views(self, left: EncodedBatchView, right: EncodedBatchView) -> EncodedBatchView:
        if left.hidden_batch is not None or right.hidden_batch is not None:
            left = self._materialize_batch_view(left)
            right = self._materialize_batch_view(right)
        return EncodedBatchView(
            frame_counts=left.frame_counts + right.frame_counts,
            frame_mss=left.frame_mss + right.frame_mss,
            energies=left.energies + right.energies,
            encoded_items=(left.encoded_items or []) + (right.encoded_items or []),
        )

    def _materialize_batch_view(self, batch_view: EncodedBatchView) -> EncodedBatchView:
        if batch_view.hidden_batch is None:
            return batch_view
        encoded_items: list[EncodedFrames] = []
        hidden_batch = batch_view.hidden_batch
        for index, (frame_count, frame_ms, energy) in enumerate(zip(batch_view.frame_counts, batch_view.frame_mss, batch_view.energies)):
            hidden = hidden_batch[index, :frame_count].detach().to("cpu", dtype=torch.float32).numpy()
            encoded_items.append(
                EncodedFrames(
                    embeddings=hidden.astype(np.float32),
                    frame_ms=frame_ms,
                    energy=energy,
                )
            )
        self.release_batch_view(batch_view)
        return EncodedBatchView(
            frame_counts=[len(item.embeddings) for item in encoded_items],
            frame_mss=[item.frame_ms for item in encoded_items],
            energies=[item.energy for item in encoded_items],
            encoded_items=encoded_items,
        )

    def _is_cuda_device(self) -> bool:
        return torch is not None and str(self.settings.device).startswith("cuda")

    def _is_oom_error(self, exc: BaseException) -> bool:
        message = str(exc).lower()
        return "out of memory" in message or "cuda error: out of memory" in message

    def _clear_cuda_cache(self) -> None:
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _frame_energy(self, samples: np.ndarray, frame_ms: float) -> np.ndarray:
        frame_size = max(1, int((frame_ms / 1000.0) * self.settings.sample_rate))
        energies: list[float] = []
        for start in range(0, len(samples), frame_size):
            window = samples[start : start + frame_size]
            if len(window) == 0:
                continue
            energies.append(float(np.sqrt(np.mean(np.square(window)))))
        if not energies:
            return np.zeros((1,), dtype=np.float32)
        return np.asarray(energies, dtype=np.float32)
