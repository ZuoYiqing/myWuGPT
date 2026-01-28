"""Minimal decoder-only GPT (causal transformer) implemented with torch.nn.

This module defines:
- TokenEmbedding
- CausalSelfAttention
- MLP (GELU)
- TransformerBlock
- GPT model
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

if hasattr(nn, "RMSNorm"):
    RMSNorm = nn.RMSNorm
else:

    class RMSNorm(nn.Module):
        """RMSNorm (no bias), following the standard formulation."""

        def __init__(self, n_embd: int, eps: float = 1e-6) -> None:
            super().__init__()
            self.eps = eps
            self.weight = nn.Parameter(torch.ones(n_embd))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            rms = torch.mean(x**2, dim=-1, keepdim=True)
            x = x * torch.rsqrt(rms + self.eps)
            return x * self.weight


@dataclass
class GPTConfig:
    """Configuration for GPT."""

    vocab_size: int
    n_layer: int
    n_head: int
    n_embd: int
    block_size: int
    dropout: float = 0.0
    use_pos_emb: bool = True


class TokenEmbedding(nn.Module):
    """Token embedding lookup."""

    def __init__(self, vocab_size: int, n_embd: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, n_embd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: (B, T) token ids

        Returns:
            (B, T, C) token embeddings
        """
        return self.embedding(x)


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention."""

    def __init__(self, n_head: int, n_embd: int, block_size: int, dropout: float) -> None:
        super().__init__()
        if n_embd % n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")

        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.block_size = block_size

        # Projection for query, key, value
        self.qkv_proj = nn.Linear(n_embd, 3 * n_embd)
        self.out_proj = nn.Linear(n_embd, n_embd)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        # Causal mask (1, 1, T, T)
        mask = torch.tril(torch.ones(block_size, block_size))
        self.register_buffer("causal_mask", mask.view(1, 1, block_size, block_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: (B, T, C) input embeddings

        Returns:
            (B, T, C) attention output
        """
        bsz, seq_len, n_embd = x.shape

        # (B, T, 3C) -> (B, T, 3, n_head, head_dim)
        qkv = self.qkv_proj(x)
        qkv = qkv.view(bsz, seq_len, 3, self.n_head, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # each: (B, n_head, T, head_dim)

        # Scaled dot-product attention
        attn_scores = (q @ k.transpose(-2, -1)) / (self.head_dim**0.5)
        causal_mask = self.causal_mask[:, :, :seq_len, :seq_len]
        attn_scores = attn_scores.masked_fill(causal_mask == 0, float("-inf"))
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        # (B, n_head, T, head_dim) -> (B, T, C)
        attn_output = attn_weights @ v
        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, seq_len, n_embd)
        attn_output = self.out_proj(attn_output)
        return self.resid_dropout(attn_output)


class MLP(nn.Module):
    """Feed-forward network (SwiGLU) with SiLU gating."""

    def __init__(self, n_embd: int, dropout: float) -> None:
        super().__init__()
        hidden_dim = int(4 * n_embd * 2 / 3)
        if hidden_dim % 2 != 0:
            hidden_dim += 1
        self.w_gate = nn.Linear(n_embd, hidden_dim)
        self.w_up = nn.Linear(n_embd, hidden_dim)
        self.act = nn.SiLU()
        self.w_down = nn.Linear(hidden_dim, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: (B, T, C)

        Returns:
            (B, T, C)
        """
        gate = self.act(self.w_gate(x))
        up = self.w_up(x)
        hidden = gate * up
        x = self.w_down(hidden)
        return self.dropout(x)


class TransformerBlock(nn.Module):
    """A single transformer block with pre-LN."""

    def __init__(self, n_head: int, n_embd: int, block_size: int, dropout: float) -> None:
        super().__init__()
        self.ln1 = RMSNorm(n_embd, eps=1e-6)
        self.attn = CausalSelfAttention(n_head, n_embd, block_size, dropout)
        self.ln2 = RMSNorm(n_embd, eps=1e-6)
        self.mlp = MLP(n_embd, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: (B, T, C)

        Returns:
            (B, T, C)
        """
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    """Decoder-only GPT model."""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.config = config

        self.token_emb = TokenEmbedding(config.vocab_size, config.n_embd)
        if config.use_pos_emb:
            self.pos_emb = nn.Embedding(config.block_size, config.n_embd)
        else:
            self.pos_emb = None
        self.drop = nn.Dropout(config.dropout)

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    n_head=config.n_head,
                    n_embd=config.n_embd,
                    block_size=config.block_size,
                    dropout=config.dropout,
                )
                for _ in range(config.n_layer)
            ]
        )
        self.ln_f = RMSNorm(config.n_embd, eps=1e-6)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        if isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: (B, T) token ids

        Returns:
            logits: (B, T, vocab_size)
        """
        bsz, seq_len = x.shape
        if seq_len > self.config.block_size:
            raise ValueError("Sequence length exceeds block_size")

        token_emb = self.token_emb(x)  # (B, T, C)
        if self.pos_emb is not None:
            positions = torch.arange(seq_len, device=x.device)
            pos_emb = self.pos_emb(positions)[None, :, :]  # (1, T, C)
            x = token_emb + pos_emb
        else:
            x = token_emb
        x = self.drop(x)

        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits

    def num_parameters(self, include_embedding: bool = True) -> int:
        """Return the number of parameters.

        Args:
            include_embedding: whether to include embedding parameters

        Returns:
            Total parameter count.
        """
        if include_embedding:
            return sum(p.numel() for p in self.parameters())
        emb_params = sum(p.numel() for p in self.token_emb.parameters())
        if self.pos_emb is not None:
            emb_params += sum(p.numel() for p in self.pos_emb.parameters())
        return sum(p.numel() for p in self.parameters()) - emb_params
