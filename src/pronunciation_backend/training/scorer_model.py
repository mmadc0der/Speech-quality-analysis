import math
from typing import TypedDict

import torch
import torch.nn as nn

class ScorerOutputs(TypedDict):
    match_score: torch.Tensor      # (batch, seq_len) - regression 0-100
    duration_score: torch.Tensor   # (batch, seq_len) - regression 0-100
    presence_logit: torch.Tensor   # (batch, seq_len) - binary logit


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 256):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, d_model]
        """
        return x + self.pe[:, :x.size(1), :]


class PhonemeScorerModel(nn.Module):
    """
    Contextual network predicting pronunciation scores for each phoneme in a word.
    
    Inputs per phoneme:
      - mean_embedding (e.g. 768-dim from HuBERT)
      - variance (1-dim)
      - duration_z_score (1-dim)
      - energy_mean (1-dim)
      - starts_late (1-dim)
      - expected_phoneme (int ID for embedding)
    """
    
    def __init__(
        self,
        acoustic_dim: int = 771,
        num_phonemes: int = 42, # 39 base + PAD + UNK
        phoneme_embed_dim: int = 32,
        d_model: int = 256,
        n_layers: int = 2,
        n_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.acoustic_dim = acoustic_dim
        self.phoneme_embed_dim = phoneme_embed_dim
        self.d_model = d_model
        
        # 1. Phoneme Embedding
        self.phoneme_embedding = nn.Embedding(
            num_embeddings=num_phonemes,
            embedding_dim=phoneme_embed_dim,
            padding_idx=0
        )
        
        # 2. Input Projection
        self.input_proj = nn.Sequential(
            nn.Linear(acoustic_dim + phoneme_embed_dim, d_model),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout)
        )
        
        # 3. Contextual Encoder (Transformer)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 2,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        # Disable nested tensor conversion to avoid prototype warnings and keep
        # runtime behavior more predictable across train/validation passes.
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers,
            enable_nested_tensor=False,
        )
        
        # 4. Multi-Task Scoring Heads
        # Each head takes the contextualized d_model vector and predicts its target
        self.match_head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Linear(64, 1)
        )
        
        self.duration_head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Linear(64, 1)
        )
        
        self.presence_head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Linear(64, 1)
        )

    def forward(
        self,
        acoustic_features: torch.Tensor,
        phoneme_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> ScorerOutputs:
        """
        Args:
            acoustic_features: [batch_size, seq_len, acoustic_dim]
            phoneme_ids: [batch_size, seq_len] of int
            attention_mask: [batch_size, seq_len] of bool (True means valid, False means padding)
        """
        # Embed expected phonemes
        phoneme_emb = self.phoneme_embedding(phoneme_ids) # [batch, seq_len, embed_dim]
        
        # Concatenate acoustic + phoneme embeddings
        x_concat = torch.cat([acoustic_features, phoneme_emb], dim=-1) # [batch, seq_len, acoustic_dim + embed_dim]
        
        # Project to d_model
        x = self.input_proj(x_concat)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # PyTorch TransformerEncoder expects src_key_padding_mask to be True for PAD positions!
        # Our attention_mask is True for VALID, False for PAD.
        # So we need to invert it for PyTorch's src_key_padding_mask.
        src_key_padding_mask = ~attention_mask
        
        # Pass through Transformer
        x_context = self.transformer(x, src_key_padding_mask=src_key_padding_mask) # [batch, seq_len, d_model]
        
        # Pass through parallel heads
        match_score = self.match_head(x_context).squeeze(-1)       # [batch, seq_len]
        duration_score = self.duration_head(x_context).squeeze(-1) # [batch, seq_len]
        presence_logit = self.presence_head(x_context).squeeze(-1) # [batch, seq_len]
        
        return ScorerOutputs(
            match_score=match_score,
            duration_score=duration_score,
            presence_logit=presence_logit
        )
