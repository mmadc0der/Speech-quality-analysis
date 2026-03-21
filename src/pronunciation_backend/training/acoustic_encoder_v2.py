from __future__ import annotations

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


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, *, base: float = 10_000.0) -> None:
        super().__init__()
        if dim % 2 != 0:
            raise ValueError(f"RoPE head dim must be even, got {dim}.")
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._seq_len_cached = 0
        self._cos_cached: torch.Tensor | None = None
        self._sin_cached: torch.Tensor | None = None

    def forward(
        self,
        *,
        seq_len: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if (
            self._cos_cached is None
            or self._sin_cached is None
            or self._seq_len_cached < seq_len
            or self._cos_cached.device != device
            or self._cos_cached.dtype != dtype
        ):
            positions = torch.arange(seq_len, device=device, dtype=torch.float32)
            freqs = torch.outer(positions, self.inv_freq.to(device=device))
            angles = torch.cat((freqs, freqs), dim=-1)
            self._cos_cached = angles.cos().to(dtype=dtype)
            self._sin_cached = angles.sin().to(dtype=dtype)
            self._seq_len_cached = seq_len
        return self._cos_cached[:seq_len], self._sin_cached[:seq_len]


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).flatten(start_dim=-2)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    return (x * cos) + (_rotate_half(x) * sin)


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
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)
        self.rope = RotaryEmbedding(self.head_dim, base=rope_base)
        self.q_norm = RMSNorm(self.head_dim) if use_qk_norm else nn.Identity()
        self.k_norm = RMSNorm(self.head_dim) if use_qk_norm else nn.Identity()
        self.out_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        cos, sin = self.rope(seq_len=seq_len, device=x.device, dtype=q.dtype)
        q = self.q_norm(apply_rope(q, cos, sin))
        k = self.k_norm(apply_rope(k, cos, sin))

        key_mask = attention_mask[:, None, None, :].to(device=x.device, dtype=torch.bool)
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
    ) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(d_model)
        self.attn = AcousticSelfAttention(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
            rope_base=rope_base,
            use_qk_norm=use_qk_norm,
        )
        self.ffn_norm = RMSNorm(d_model)
        self.ffn = SwiGLUFeedForward(d_model=d_model, hidden_dim=ffn_dim, dropout=dropout)

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.attn_norm(x)
        x = self.attn(x, attention_mask)
        x = x + residual
        x = x * attention_mask.unsqueeze(-1)

        residual = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + residual
        return x * attention_mask.unsqueeze(-1)


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
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model
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
