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
        # RoPE requires even head_dim to rotate pairs of channels.
        if self.head_dim % 2 != 0:
            raise ValueError("head_dim must be even for RoPE")

        # Projection for query, key, value
        self.qkv_proj = nn.Linear(n_embd, 3 * n_embd)
        self.out_proj = nn.Linear(n_embd, n_embd)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        # Causal mask (1, 1, T, T)
        mask = torch.tril(torch.ones(block_size, block_size))
        self.register_buffer("causal_mask", mask.view(1, 1, block_size, block_size))

        # Precompute RoPE tables (cos/sin) up to block_size; buffers are non-trainable.
        inv_freq = 1.0 / (
            10000 ** (torch.arange(0, self.head_dim, 2, dtype=torch.float32) / self.head_dim)
        )
        positions = torch.arange(block_size, dtype=torch.float32)
        freqs = torch.einsum("i,j->ij", positions, inv_freq)
        self.register_buffer("rope_cos", freqs.cos())
        self.register_buffer("rope_sin", freqs.sin())

    def _apply_rope(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        # Rotate even/odd feature pairs as in the RoPE formulation.
        x_even = x[..., ::2]
        x_odd = x[..., 1::2]
        x_rotated_even = x_even * cos - x_odd * sin
        x_rotated_odd = x_even * sin + x_odd * cos
        return torch.stack((x_rotated_even, x_rotated_odd), dim=-1).flatten(-2)

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

        # Slice RoPE tables to sequence length and broadcast over batch/head dims.
        cos = self.rope_cos[:seq_len].view(1, 1, seq_len, -1)
        sin = self.rope_sin[:seq_len].view(1, 1, seq_len, -1)
        q = self._apply_rope(q, cos, sin)
        k = self._apply_rope(k, cos, sin)

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
    """Feed-forward network (MoE with SwiGLU experts)."""

    def __init__(self, n_embd: int, dropout: float, num_experts: int = 4) -> None:
        super().__init__()
        # Expert hidden size follows the SwiGLU recommendation and is rounded to even.
        hidden_dim = int(4 * n_embd * 2 / 3)
        if hidden_dim % 2 != 0:
            hidden_dim += 1

        class SwiGLUExpert(nn.Module):
            """Single SwiGLU expert: SiLU(gate) * up -> down, dropout after down."""

            def __init__(self, embd: int, hid: int, drop: float) -> None:
                super().__init__()
                self.w_gate = nn.Linear(embd, hid)
                self.w_up = nn.Linear(embd, hid)
                self.act = nn.SiLU()
                self.w_down = nn.Linear(hid, embd)
                self.dropout = nn.Dropout(drop)

            def forward(self, x_in: torch.Tensor) -> torch.Tensor:
                gate = self.act(self.w_gate(x_in))
                up = self.w_up(x_in)
                hidden = gate * up
                out = self.w_down(hidden)
                return self.dropout(out)

        self.num_experts = num_experts
        self.experts = nn.ModuleList(
            [SwiGLUExpert(n_embd, hidden_dim, dropout) for _ in range(num_experts)]
        )
        # Router maps each token to expert logits; use top-2 gating per token.
        self.router = nn.Linear(n_embd, num_experts, bias=False)
        # Cache for expert load statistics (can be read externally after forward).
        self.last_router_stats: dict[str, torch.Tensor] | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: (B, T, C)

        Returns:
            (B, T, C)
        """
        logits = self.router(x)  # (B, T, N)
        topk_vals, topk_idx = torch.topk(logits, k=2, dim=-1)
        topk_weights = torch.softmax(topk_vals, dim=-1)  # (B, T, 2)

        # Build a dense weight tensor per expert for straightforward mixing.
        weights = torch.zeros_like(logits)
        weights.scatter_(-1, topk_idx, topk_weights)

        # Compute each expert output and blend with routing weights.
        expert_outputs = [expert(x) for expert in self.experts]  # each: (B, T, C)
        mixed = torch.zeros_like(x)
        for idx, expert_out in enumerate(expert_outputs):
            mixed = mixed + expert_out * weights[..., idx : idx + 1]

        # Cache selection statistics for optional load monitoring.
        with torch.no_grad():
            token_counts = weights.gt(0).sum(dim=(0, 1))
            total_tokens = weights.shape[0] * weights.shape[1]
            self.last_router_stats = {
                "token_counts": token_counts,
                "token_fraction": token_counts / max(total_tokens, 1),
            }

        return mixed


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
        return sum(p.numel() for p in self.parameters()) - emb_params
