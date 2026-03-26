"""
GPT-style decoder-only transformer for autoregressive next-token prediction
over shapegraph token sequences.

The model predicts the next VQ-VAE codebook index (or GOAL token) given
a causal context window of previous tokens.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.qkv = nn.Linear(embed_dim, 3 * embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, H, T, D)
        q, k, v = qkv.unbind(0)

        # Scaled dot-product with causal mask
        att = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        causal_mask = torch.triu(
            torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=1
        )
        att = att.masked_fill(causal_mask, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        out = (att @ v).transpose(1, 2).reshape(B, T, C)
        return self.resid_dropout(self.out_proj(out))


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.attn = CausalSelfAttention(embed_dim, num_heads, dropout)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.GELU(),
            nn.Linear(4 * embed_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class ShapegraphGPT(nn.Module):
    """GPT-style autoregressive model over shapegraph token sequences.

    Args:
        vocab_size: number of token types (codebook_size + special tokens)
        context_length: maximum sequence length
        embed_dim: transformer hidden dimension
        num_heads: attention heads per layer
        num_layers: number of transformer blocks
        dropout: dropout rate
    """

    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        embed_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 6,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.context_length = context_length
        self.vocab_size = vocab_size

        self.token_emb = nn.Embedding(vocab_size, embed_dim)
        self.pos_emb = nn.Embedding(context_length, embed_dim)
        self.drop = nn.Dropout(dropout)

        self.blocks = nn.Sequential(*[
            TransformerBlock(embed_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])
        self.ln_f = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, vocab_size, bias=False)

        # Weight tying
        self.head.weight = self.token_emb.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        """
        Args:
            idx: (B, T) integer token indices

        Returns:
            logits: (B, T, vocab_size) next-token logits
        """
        B, T = idx.shape
        assert T <= self.context_length, (
            f"Sequence length {T} exceeds context_length {self.context_length}"
        )

        pos = torch.arange(T, device=idx.device)
        x = self.drop(self.token_emb(idx) + self.pos_emb(pos))
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x)
        return logits

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> torch.Tensor:
        """Autoregressively generate tokens.

        Args:
            idx: (B, T) conditioning token indices
            max_new_tokens: number of tokens to generate
            temperature: sampling temperature (1.0 = no change)
            top_k: if set, only sample from top-k logits

        Returns:
            (B, T + max_new_tokens) extended token sequence
        """
        for _ in range(max_new_tokens):
            # Crop to context length
            idx_cond = idx[:, -self.context_length:]
            logits = self(idx_cond)
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = float("-inf")

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, idx_next], dim=1)

        return idx
