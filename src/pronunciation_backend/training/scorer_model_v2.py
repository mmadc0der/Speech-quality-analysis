from __future__ import annotations

from collections.abc import Mapping
from typing import TypedDict

import torch
import torch.nn as nn

from pronunciation_backend.training.acoustic_encoder_v2 import (
    AcousticEncoderBlock,
    AcousticEncoderV2,
    EncoderBlockConfig,
    RMSNorm,
    build_encoder_block_config,
)
from pronunciation_backend.training.scoring_targets import expected_human_score_from_probs, expected_score_from_probs


class ScorerV2Outputs(TypedDict):
    quality_logits: torch.Tensor
    omission_logit: torch.Tensor
    class_probs: torch.Tensor
    expected_score: torch.Tensor
    expected_human_score: torch.Tensor


def scorer_model_kwargs_from_config(config: Mapping[str, object] | None) -> dict[str, object]:
    config_dict = config if isinstance(config, Mapping) else {}
    use_qk_norm = bool(config_dict.get("use_qk_norm", not bool(config_dict.get("disable_qk_norm", False))))
    kwargs: dict[str, object] = {
        "acoustic_input_dim": int(config_dict.get("acoustic_input_dim", 768)),
        "d_model": int(config_dict.get("d_model", 384)),
        "num_heads": int(config_dict.get("num_heads", 6)),
        "acoustic_layers": int(config_dict.get("acoustic_layers", 6)),
        "scorer_layers": int(config_dict.get("scorer_layers", 2)),
        "ffn_dim": int(config_dict.get("ffn_dim", 1_536)),
        "phoneme_vocab_size": int(config_dict.get("phoneme_vocab_size", 42)),
        "phoneme_embed_dim": int(config_dict.get("phoneme_embed_dim", 48)),
        "dropout": float(config_dict.get("dropout", 0.05)),
        "rope_base": float(config_dict.get("rope_base", 10_000.0)),
        "use_qk_norm": use_qk_norm,
        "architecture_version": str(config_dict.get("architecture_version", "v2_compat")),
    }
    optional_keys = (
        "block_layout",
        "norm_scheme",
        "branch_scale_init",
        "attention_score_mode",
        "qk_norm_mode",
        "positional_mode",
        "rope_adaptation_scope",
        "rope_reference_seq_len",
    )
    for key in optional_keys:
        if key in config_dict:
            kwargs[key] = config_dict[key]
    return kwargs


class PhonemeScorerModelV2(nn.Module):
    def __init__(
        self,
        *,
        acoustic_input_dim: int = 768,
        d_model: int = 384,
        num_heads: int = 6,
        acoustic_layers: int = 6,
        scorer_layers: int = 2,
        ffn_dim: int = 1_536,
        phoneme_vocab_size: int = 42,
        phoneme_embed_dim: int = 48,
        dropout: float = 0.05,
        rope_base: float = 10_000.0,
        use_qk_norm: bool = True,
        architecture_version: str = "v2_compat",
        block_config: EncoderBlockConfig | None = None,
        block_layout: str | None = None,
        norm_scheme: str | None = None,
        branch_scale_init: float | None = None,
        attention_score_mode: str | None = None,
        qk_norm_mode: str | None = None,
        positional_mode: str | None = None,
        rope_adaptation_scope: str | None = None,
        rope_reference_seq_len: int | None = None,
    ) -> None:
        super().__init__()
        self.block_config = build_encoder_block_config(
            architecture_version=architecture_version,
            use_qk_norm=use_qk_norm,
            block_config=block_config,
            block_layout=block_layout,
            norm_scheme=norm_scheme,
            branch_scale_init=branch_scale_init,
            attention_score_mode=attention_score_mode,
            qk_norm_mode=qk_norm_mode,
            positional_mode=positional_mode,
            rope_adaptation_scope=rope_adaptation_scope,
            rope_reference_seq_len=rope_reference_seq_len,
        )
        self.acoustic_encoder = AcousticEncoderV2(
            input_dim=acoustic_input_dim,
            d_model=d_model,
            num_heads=num_heads,
            num_layers=acoustic_layers,
            ffn_dim=ffn_dim,
            dropout=dropout,
            rope_base=rope_base,
            use_qk_norm=use_qk_norm,
            block_config=self.block_config,
        )
        self.phoneme_embedding = nn.Embedding(
            num_embeddings=phoneme_vocab_size,
            embedding_dim=phoneme_embed_dim,
            padding_idx=0,
        )
        self.phoneme_proj = nn.Sequential(
            RMSNorm(phoneme_embed_dim),
            nn.Linear(phoneme_embed_dim, d_model, bias=False),
            nn.Dropout(dropout),
        )
        self.fusion_norm = RMSNorm(d_model * 4)
        self.fusion_proj = nn.Linear(d_model * 4, d_model, bias=False)
        self.fusion_dropout = nn.Dropout(dropout)
        self.scorer_blocks = nn.ModuleList(
            [
                AcousticEncoderBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    ffn_dim=ffn_dim,
                    dropout=dropout,
                    rope_base=rope_base,
                    use_qk_norm=use_qk_norm,
                    block_config=self.block_config,
                )
                for _ in range(scorer_layers)
            ]
        )
        self.output_norm = RMSNorm(d_model)
        self.quality_head = nn.Linear(d_model, 3, bias=False)
        self.omission_head = nn.Linear(d_model, 1, bias=False)

    def load_pretrained_acoustic_encoder(
        self,
        checkpoint_payload: dict[str, object],
        *,
        strict: bool = True,
    ) -> None:
        state_dict = checkpoint_payload.get("model_state_dict")
        if not isinstance(state_dict, dict):
            raise ValueError("checkpoint_payload must contain model_state_dict")

        encoder_state = {
            key.removeprefix("encoder."): value
            for key, value in state_dict.items()
            if key.startswith("encoder.")
        }
        if not encoder_state:
            raise ValueError("checkpoint_payload does not contain encoder.* weights")
        self.acoustic_encoder.load_state_dict(encoder_state, strict=strict)

    def set_acoustic_encoder_trainable(self, trainable: bool) -> None:
        for param in self.acoustic_encoder.parameters():
            param.requires_grad = trainable

    def forward(
        self,
        acoustic_embeddings: torch.Tensor,
        phoneme_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> ScorerV2Outputs:
        attention_mask = attention_mask.to(device=acoustic_embeddings.device, dtype=torch.bool)
        acoustic_state = self.acoustic_encoder(
            acoustic_embeddings=acoustic_embeddings,
            attention_mask=attention_mask,
        )

        phoneme_state = self.phoneme_proj(self.phoneme_embedding(phoneme_ids))
        fused = torch.cat(
            (
                acoustic_state,
                phoneme_state,
                acoustic_state - phoneme_state,
                acoustic_state * phoneme_state,
            ),
            dim=-1,
        )
        fused = self.fusion_proj(self.fusion_norm(fused))
        fused = self.fusion_dropout(fused) * attention_mask.unsqueeze(-1)

        scorer_state = fused
        for block in self.scorer_blocks:
            scorer_state = block(scorer_state, attention_mask)
        scorer_state = self.output_norm(scorer_state) * attention_mask.unsqueeze(-1)

        quality_logits = self.quality_head(scorer_state)
        omission_logit = self.omission_head(scorer_state).squeeze(-1)
        class_probs = torch.softmax(quality_logits, dim=-1)
        expected_score = expected_score_from_probs(class_probs) * attention_mask
        expected_human_score = expected_human_score_from_probs(class_probs) * attention_mask

        return ScorerV2Outputs(
            quality_logits=quality_logits,
            omission_logit=omission_logit,
            class_probs=class_probs,
            expected_score=expected_score,
            expected_human_score=expected_human_score,
        )


class PhonemeScorerModelV3(PhonemeScorerModelV2):
    def __init__(self, *args, **kwargs) -> None:
        kwargs.setdefault("architecture_version", "v3")
        super().__init__(*args, **kwargs)
