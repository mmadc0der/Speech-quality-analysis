from __future__ import annotations

from dataclasses import dataclass, replace

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x * norm * self.weight


class HeadwiseRMSNorm(nn.Module):
    def __init__(self, num_heads: int, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(num_heads, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x * norm * self.weight.unsqueeze(0).unsqueeze(2)


class LayerScale(nn.Module):
    def __init__(self, dim: int, *, init: float | None = None) -> None:
        super().__init__()
        self.scale = None if init is None else nn.Parameter(torch.full((dim,), float(init)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.scale is None:
            return x
        return x * self.scale


@dataclass(frozen=True)
class EncoderBlockConfig:
    architecture_version: str = "v2_compat"
    block_layout: str = "sequential_prenorm"
    norm_scheme: str = "rmsnorm"
    branch_scale_init: float | None = None
    ffn_mode: str = "swiglu"
    attention_score_mode: str = "default"
    qk_norm_mode: str = "shared_head_dim"
    positional_mode: str = "rope"
    rope_adaptation_scope: str = "batch_seq_len"
    rope_reference_seq_len: int = 256

    def __post_init__(self) -> None:
        if self.architecture_version not in {"v2_compat", "v3"}:
            raise ValueError(f"Unsupported architecture_version={self.architecture_version!r}.")
        if self.block_layout not in {"sequential_prenorm", "parallel_prenorm"}:
            raise ValueError(f"Unsupported block_layout={self.block_layout!r}.")
        if self.norm_scheme not in {"rmsnorm", "sandwich_rmsnorm"}:
            raise ValueError(f"Unsupported norm_scheme={self.norm_scheme!r}.")
        if self.ffn_mode != "swiglu":
            raise ValueError(f"Unsupported ffn_mode={self.ffn_mode!r}; only 'swiglu' is implemented.")
        if self.attention_score_mode not in {"default", "learned_temperature", "talking_heads"}:
            raise ValueError(f"Unsupported attention_score_mode={self.attention_score_mode!r}.")
        if self.qk_norm_mode not in {"disabled", "shared_head_dim", "per_head_qk", "per_head_qkv"}:
            raise ValueError(f"Unsupported qk_norm_mode={self.qk_norm_mode!r}.")
        if self.positional_mode not in {"rope", "rope_auto_adaptive"}:
            raise ValueError(f"Unsupported positional_mode={self.positional_mode!r}.")
        if self.rope_adaptation_scope != "batch_seq_len":
            raise ValueError(f"Unsupported rope_adaptation_scope={self.rope_adaptation_scope!r}.")
        if self.rope_reference_seq_len <= 0:
            raise ValueError("rope_reference_seq_len must be positive.")

    @classmethod
    def v2_compat(cls) -> "EncoderBlockConfig":
        return cls()

    @classmethod
    def v3(cls) -> "EncoderBlockConfig":
        return cls(
            architecture_version="v3",
            block_layout="parallel_prenorm",
            norm_scheme="sandwich_rmsnorm",
            branch_scale_init=1e-3,
            ffn_mode="swiglu",
            attention_score_mode="learned_temperature",
            qk_norm_mode="per_head_qk",
            positional_mode="rope_auto_adaptive",
            rope_adaptation_scope="batch_seq_len",
            rope_reference_seq_len=256,
        )


def build_encoder_block_config(
    *,
    architecture_version: str = "v2_compat",
    use_qk_norm: bool = True,
    block_config: EncoderBlockConfig | None = None,
    block_layout: str | None = None,
    norm_scheme: str | None = None,
    branch_scale_init: float | None = None,
    attention_score_mode: str | None = None,
    qk_norm_mode: str | None = None,
    positional_mode: str | None = None,
    rope_adaptation_scope: str | None = None,
    rope_reference_seq_len: int | None = None,
) -> EncoderBlockConfig:
    resolved = block_config or (
        EncoderBlockConfig.v3() if architecture_version == "v3" else EncoderBlockConfig.v2_compat()
    )
    overrides: dict[str, object] = {}
    if block_layout is not None:
        overrides["block_layout"] = block_layout
    if norm_scheme is not None:
        overrides["norm_scheme"] = norm_scheme
    if branch_scale_init is not None:
        overrides["branch_scale_init"] = branch_scale_init
    if attention_score_mode is not None:
        overrides["attention_score_mode"] = attention_score_mode
    if qk_norm_mode is not None:
        overrides["qk_norm_mode"] = qk_norm_mode
    if positional_mode is not None:
        overrides["positional_mode"] = positional_mode
    if rope_adaptation_scope is not None:
        overrides["rope_adaptation_scope"] = rope_adaptation_scope
    if rope_reference_seq_len is not None:
        overrides["rope_reference_seq_len"] = rope_reference_seq_len
    if not use_qk_norm:
        overrides["qk_norm_mode"] = "disabled"
    if not overrides:
        return resolved
    return replace(resolved, **overrides)


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, *, base: float = 10_000.0) -> None:
        super().__init__()
        if dim % 2 != 0:
            raise ValueError(f"RoPE head dim must be even, got {dim}.")
        self.dim = dim
        self.base = float(base)
        self._seq_len_cached = 0
        self._cos_cached: torch.Tensor | None = None
        self._sin_cached: torch.Tensor | None = None
        self._cache_base: float | None = None

    def forward(
        self,
        *,
        seq_len: int,
        device: torch.device,
        dtype: torch.dtype,
        base: float | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        resolved_base = float(base if base is not None else self.base)
        if (
            self._cos_cached is None
            or self._sin_cached is None
            or self._seq_len_cached < seq_len
            or self._cos_cached.device != device
            or self._cos_cached.dtype != dtype
            or self._cache_base != resolved_base
        ):
            positions = torch.arange(seq_len, device=device, dtype=torch.float32)
            inv_freq = 1.0 / (
                resolved_base
                ** (torch.arange(0, self.dim, 2, device=device, dtype=torch.float32) / self.dim)
            )
            freqs = torch.outer(positions, inv_freq)
            angles = torch.cat((freqs, freqs), dim=-1)
            self._cos_cached = angles.cos().to(dtype=dtype)
            self._sin_cached = angles.sin().to(dtype=dtype)
            self._seq_len_cached = seq_len
            self._cache_base = resolved_base
        return self._cos_cached[:seq_len], self._sin_cached[:seq_len]


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).flatten(start_dim=-2)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    return (x * cos) + (_rotate_half(x) * sin)


def _batch_valid_seq_len(attention_mask: torch.Tensor) -> int:
    if attention_mask.numel() == 0:
        return 1
    valid_lengths = attention_mask.to(dtype=torch.int64).sum(dim=-1)
    return max(int(valid_lengths.max().item()), 1)


def sample_mask_positions(
    attention_mask: torch.Tensor,
    *,
    mask_ratio: float = 0.15,
    block_size: int = 1,
    min_masks: int = 1,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    if attention_mask.ndim != 2:
        raise ValueError(f"attention_mask must have shape [batch, seq], got {tuple(attention_mask.shape)}")
    if not 0.0 <= mask_ratio <= 1.0:
        raise ValueError(f"mask_ratio must be in [0, 1], got {mask_ratio}")
    if block_size <= 0:
        raise ValueError(f"block_size must be positive, got {block_size}")
    if min_masks < 0:
        raise ValueError(f"min_masks must be non-negative, got {min_masks}")

    mask_cpu = attention_mask.detach().to(device="cpu", dtype=torch.bool)
    sampled = torch.zeros_like(mask_cpu)

    for row_index in range(mask_cpu.size(0)):
        valid_positions = torch.nonzero(mask_cpu[row_index], as_tuple=False).flatten()
        valid_count = int(valid_positions.numel())
        if valid_count == 0:
            continue

        target_masks = int(round(valid_count * mask_ratio))
        if mask_ratio > 0.0:
            target_masks = max(min_masks, target_masks)
        target_masks = min(valid_count, target_masks)
        if target_masks == 0:
            continue

        if block_size == 1:
            order = torch.randperm(valid_count, generator=generator)
            chosen = valid_positions[order[:target_masks]]
            sampled[row_index, chosen] = True
            continue

        span_starts = torch.randperm(valid_count, generator=generator)
        masked_so_far = 0
        for start_index in span_starts.tolist():
            block = valid_positions[start_index : start_index + block_size]
            if block.numel() == 0:
                continue
            new_positions = block[~sampled[row_index, block]]
            sampled[row_index, new_positions] = True
            masked_so_far += int(new_positions.numel())
            if masked_so_far >= target_masks:
                break

    return sampled.to(device=attention_mask.device)


class AcousticSelfAttention(nn.Module):
    def __init__(
        self,
        *,
        d_model: int,
        num_heads: int,
        dropout: float,
        rope_base: float,
        use_qk_norm: bool,
        block_config: EncoderBlockConfig | None = None,
    ) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError(f"d_model={d_model} must be divisible by num_heads={num_heads}.")
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        if self.head_dim % 2 != 0:
            raise ValueError(
                f"head_dim={self.head_dim} must be even so rotary embeddings can be applied."
            )
        self.dropout = dropout
        self.block_config = block_config or build_encoder_block_config(use_qk_norm=use_qk_norm)
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)
        self.rope = RotaryEmbedding(self.head_dim, base=rope_base)
        self.q_norm = self._build_projection_norm("q")
        self.k_norm = self._build_projection_norm("k")
        self.v_norm = self._build_projection_norm("v")
        self.log_attn_temperature = (
            nn.Parameter(torch.zeros(self.num_heads))
            if self.block_config.attention_score_mode in {"learned_temperature", "talking_heads"}
            else None
        )
        self.pre_softmax_head_mix = (
            nn.Parameter(torch.eye(self.num_heads))
            if self.block_config.attention_score_mode == "talking_heads"
            else None
        )
        self.post_softmax_head_mix = (
            nn.Parameter(torch.eye(self.num_heads))
            if self.block_config.attention_score_mode == "talking_heads"
            else None
        )
        self.attn_dropout = nn.Dropout(dropout)
        self.out_dropout = nn.Dropout(dropout)

    def _build_projection_norm(self, projection_name: str) -> nn.Module:
        if self.block_config.qk_norm_mode == "disabled":
            return nn.Identity()
        if projection_name == "v" and self.block_config.qk_norm_mode != "per_head_qkv":
            return nn.Identity()
        if self.block_config.qk_norm_mode == "shared_head_dim":
            return RMSNorm(self.head_dim)
        return HeadwiseRMSNorm(self.num_heads, self.head_dim)

    def _resolved_rope_base(self, attention_mask: torch.Tensor) -> float:
        if self.block_config.positional_mode == "rope":
            return self.rope.base
        effective_seq_len = _batch_valid_seq_len(attention_mask)
        reference = max(self.block_config.rope_reference_seq_len, 1)
        return max(2.0, self.rope.base * (float(effective_seq_len) / float(reference)))

    def _apply_attention_temperature(self, q: torch.Tensor) -> torch.Tensor:
        if self.log_attn_temperature is None:
            return q
        temperature = torch.exp(self.log_attn_temperature).view(1, self.num_heads, 1, 1)
        return q * temperature

    def _talking_heads_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        key_mask: torch.Tensor,
    ) -> torch.Tensor:
        scale = self.head_dim**-0.5
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        if self.pre_softmax_head_mix is not None:
            scores = torch.einsum("bhij,hk->bkij", scores, self.pre_softmax_head_mix)
        scores = scores.masked_fill(~key_mask, torch.finfo(scores.dtype).min)
        probs = torch.softmax(scores, dim=-1)
        if self.post_softmax_head_mix is not None:
            probs = torch.einsum("bhij,hk->bkij", probs, self.post_softmax_head_mix)
        probs = self.attn_dropout(probs)
        return torch.matmul(probs, v)

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        cos, sin = self.rope(
            seq_len=seq_len,
            device=x.device,
            dtype=q.dtype,
            base=self._resolved_rope_base(attention_mask),
        )
        q = self.q_norm(apply_rope(q, cos, sin))
        k = self.k_norm(apply_rope(k, cos, sin))
        v = self.v_norm(v)
        q = self._apply_attention_temperature(q)

        key_mask = attention_mask[:, None, None, :].to(device=x.device, dtype=torch.bool)
        if self.block_config.attention_score_mode == "talking_heads":
            attn = self._talking_heads_attention(q, k, v, key_mask=key_mask)
        else:
            attn = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=key_mask,
                dropout_p=self.dropout if self.training else 0.0,
            )
        attn = attn.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        return self.out_dropout(self.o_proj(attn))


class SwiGLUFeedForward(nn.Module):
    def __init__(self, *, d_model: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(d_model, hidden_dim, bias=False)
        self.up_proj = nn.Linear(d_model, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, d_model, bias=False)
        self.out_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.silu(self.gate_proj(x)) * self.up_proj(x)
        return self.out_dropout(self.down_proj(x))


class AcousticEncoderBlock(nn.Module):
    def __init__(
        self,
        *,
        d_model: int,
        num_heads: int,
        ffn_dim: int,
        dropout: float,
        rope_base: float,
        use_qk_norm: bool,
        block_config: EncoderBlockConfig | None = None,
    ) -> None:
        super().__init__()
        self.block_config = block_config or build_encoder_block_config(use_qk_norm=use_qk_norm)
        self.branch_norm = (
            RMSNorm(d_model) if self.block_config.block_layout == "parallel_prenorm" else None
        )
        self.attn_norm = (
            None if self.block_config.block_layout == "parallel_prenorm" else RMSNorm(d_model)
        )
        self.attn = AcousticSelfAttention(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
            rope_base=rope_base,
            use_qk_norm=use_qk_norm,
            block_config=self.block_config,
        )
        self.ffn_norm = (
            None if self.block_config.block_layout == "parallel_prenorm" else RMSNorm(d_model)
        )
        self.ffn = SwiGLUFeedForward(d_model=d_model, hidden_dim=ffn_dim, dropout=dropout)
        self.attn_post_norm = (
            RMSNorm(d_model) if self.block_config.norm_scheme == "sandwich_rmsnorm" else None
        )
        self.ffn_post_norm = (
            RMSNorm(d_model) if self.block_config.norm_scheme == "sandwich_rmsnorm" else None
        )
        self.attn_scale = LayerScale(d_model, init=self.block_config.branch_scale_init)
        self.ffn_scale = LayerScale(d_model, init=self.block_config.branch_scale_init)

    def _mask_output(self, x: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return x * attention_mask.unsqueeze(-1)

    def _sequential_forward(self, x: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        residual = x
        assert self.attn_norm is not None
        attn_out = self.attn(self.attn_norm(x), attention_mask)
        if self.attn_post_norm is not None:
            attn_out = self.attn_post_norm(attn_out)
        x = residual + self.attn_scale(attn_out)
        x = self._mask_output(x, attention_mask)

        residual = x
        assert self.ffn_norm is not None
        ffn_out = self.ffn(self.ffn_norm(x))
        if self.ffn_post_norm is not None:
            ffn_out = self.ffn_post_norm(ffn_out)
        x = residual + self.ffn_scale(ffn_out)
        return self._mask_output(x, attention_mask)

    def _parallel_forward(self, x: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        assert self.branch_norm is not None
        branch_input = self.branch_norm(x)
        attn_out = self.attn(branch_input, attention_mask)
        if self.attn_post_norm is not None:
            attn_out = self.attn_post_norm(attn_out)
        ffn_out = self.ffn(branch_input)
        if self.ffn_post_norm is not None:
            ffn_out = self.ffn_post_norm(ffn_out)
        x = x + self.attn_scale(attn_out) + self.ffn_scale(ffn_out)
        return self._mask_output(x, attention_mask)

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        if self.block_config.block_layout == "parallel_prenorm":
            return self._parallel_forward(x, attention_mask)
        return self._sequential_forward(x, attention_mask)


class AcousticEncoderV2(nn.Module):
    def __init__(
        self,
        *,
        input_dim: int = 768,
        d_model: int = 384,
        num_heads: int = 6,
        num_layers: int = 6,
        ffn_dim: int = 1_536,
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
        self.input_dim = input_dim
        self.d_model = d_model
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
        self.input_norm = RMSNorm(input_dim)
        self.input_proj = nn.Linear(input_dim, d_model, bias=False)
        self.input_dropout = nn.Dropout(dropout)
        self.mask_token = nn.Parameter(torch.zeros(d_model))
        self.blocks = nn.ModuleList(
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
                for _ in range(num_layers)
            ]
        )
        self.final_norm = RMSNorm(d_model)
        nn.init.normal_(self.mask_token, mean=0.0, std=0.02)

    def project_inputs(
        self,
        acoustic_embeddings: torch.Tensor,
        *,
        attention_mask: torch.Tensor,
        mask_positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if acoustic_embeddings.ndim != 3:
            raise ValueError(
                f"acoustic_embeddings must have shape [batch, seq, dim], got {tuple(acoustic_embeddings.shape)}"
            )
        if acoustic_embeddings.size(-1) != self.input_dim:
            raise ValueError(
                f"Expected acoustic_embeddings[..., {self.input_dim}], got {tuple(acoustic_embeddings.shape)}"
            )

        projected = self.input_proj(self.input_norm(acoustic_embeddings))
        projected = self.input_dropout(projected)

        if mask_positions is not None:
            mask_positions = mask_positions.to(device=projected.device, dtype=torch.bool) & attention_mask
            projected = torch.where(
                mask_positions.unsqueeze(-1),
                self.mask_token.view(1, 1, -1),
                projected,
            )

        return projected * attention_mask.unsqueeze(-1)

    def forward(
        self,
        acoustic_embeddings: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        *,
        mask_positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if attention_mask is None:
            attention_mask = torch.ones(
                acoustic_embeddings.shape[:2],
                device=acoustic_embeddings.device,
                dtype=torch.bool,
            )
        else:
            attention_mask = attention_mask.to(device=acoustic_embeddings.device, dtype=torch.bool)

        x = self.project_inputs(
            acoustic_embeddings,
            attention_mask=attention_mask,
            mask_positions=mask_positions,
        )
        for block in self.blocks:
            x = block(x, attention_mask)
        x = self.final_norm(x)
        return x * attention_mask.unsqueeze(-1)


class AcousticEncoderV3(AcousticEncoderV2):
    def __init__(self, *args, **kwargs) -> None:
        kwargs.setdefault("architecture_version", "v3")
        super().__init__(*args, **kwargs)
