from __future__ import annotations

from dataclasses import dataclass

import torch

from pronunciation_backend.models import PhoneFeatures
from pronunciation_backend.training.dataset import get_phoneme_id


@dataclass(frozen=True)
class PhoneFeatureTensorMapper:
    acoustic_dim: int = 768

    def build_inputs(self, phone_features: list[PhoneFeatures]) -> dict[str, torch.Tensor]:
        if not phone_features:
            raise ValueError("phone_features must contain at least one phoneme segment")

        acoustic_embeddings: list[list[float]] = []
        phoneme_ids: list[int] = []
        for features in phone_features:
            if len(features.mean_embedding) < self.acoustic_dim:
                raise ValueError(
                    f"Expected mean_embedding to have at least {self.acoustic_dim} dims, "
                    f"got {len(features.mean_embedding)} for phoneme {features.phoneme!r}"
                )
            acoustic_embeddings.append(features.mean_embedding[: self.acoustic_dim])
            phoneme_ids.append(get_phoneme_id(features.phoneme))

        seq_len = len(phone_features)
        return {
            "acoustic_embeddings": torch.tensor([acoustic_embeddings], dtype=torch.float32),
            "phoneme_ids": torch.tensor([phoneme_ids], dtype=torch.long),
            "attention_mask": torch.ones((1, seq_len), dtype=torch.bool),
        }
