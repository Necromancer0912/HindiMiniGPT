"""From-scratch decoder-only Transformer (MiniGPT), built twice over as a single
config-switchable codebase:

  arch="vanilla" : learned positional embeddings, LayerNorm, ReLU 2-layer MLP.
  arch="modern"  : RoPE (no learned position table), RMSNorm, SwiGLU MLP.

Both share the same MultiHeadAttention / TransformerBlock classes so the two
architectures are a true controlled ablation (same code path, one flag flipped).

Padding note: classification sequences are right-padded (pad tokens appended after
real tokens). Because attention is causal, a real token can never attend to a later
padded position regardless of arch/mask flags, so no separate key-padding mask is
needed inside attention itself -- only the pooling layer needs to know which
positions are real.
"""
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import ModelConfig


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return norm * self.weight


def make_norm(arch: str, dim: int) -> nn.Module:
    return RMSNorm(dim) if arch == "modern" else nn.LayerNorm(dim)


# ---------------------------------------------------------------------------
# Rotary position embeddings (LLaMA-style split-half convention)
# ---------------------------------------------------------------------------
def build_rope_cache(seq_len: int, head_dim: int, device, base: float = 10000.0):
    theta = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    pos = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(pos, theta)  # (T, head_dim/2)
    emb = torch.cat([freqs, freqs], dim=-1)  # (T, head_dim)
    return emb.cos(), emb.sin()


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x: (B, nh, T, hd); cos/sin: (T, hd) -> broadcast over (B, nh)
    return x * cos + rotate_half(x) * sin


# ---------------------------------------------------------------------------
# Attention
# ---------------------------------------------------------------------------
class MultiHeadAttention(nn.Module):
    """Explicit Q/K/V projections + scaled dot-product attention + causal mask,
    implemented directly (no nn.MultiheadAttention / nn.TransformerEncoder).
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.head_dim = cfg.head_dim
        self.arch = cfg.arch

        self.qkv_proj = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=False)
        self.out_proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)
        self.attn_dropout = nn.Dropout(cfg.dropout)
        self.resid_dropout = nn.Dropout(cfg.dropout)

        causal = torch.tril(torch.ones(cfg.block_size, cfg.block_size, dtype=torch.bool))
        self.register_buffer("causal_mask", causal, persistent=False)

    def forward(self, x: torch.Tensor, rope: Optional[tuple] = None) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.qkv_proj(x)  # (B, T, 3C)
        q, k, v = qkv.split(C, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # (B, nh, T, hd)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        if self.arch == "modern" and rope is not None:
            cos, sin = rope
            q = apply_rope(q, cos[:T], sin[:T])
            k = apply_rope(k, cos[:T], sin[:T])

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # (B, nh, T, T)
        scores = scores.masked_fill(~self.causal_mask[:T, :T], float("-inf"))
        attn = F.softmax(scores, dim=-1)
        attn = self.attn_dropout(attn)

        out = attn @ v  # (B, nh, T, hd)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.resid_dropout(self.out_proj(out))
        return out


# ---------------------------------------------------------------------------
# Feed-forward
# ---------------------------------------------------------------------------
class FeedForward(nn.Module):
    """Vanilla two-layer ReLU MLP, as specified by the assignment."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cfg.n_embd, 4 * cfg.n_embd),
            nn.ReLU(),
            nn.Linear(4 * cfg.n_embd, cfg.n_embd),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SwiGLU(nn.Module):
    """Gated MLP used by LLaMA-style modern architectures."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        hidden = int(8 * cfg.n_embd / 3)
        hidden = ((hidden + 7) // 8) * 8  # round to multiple of 8
        self.w1 = nn.Linear(cfg.n_embd, hidden, bias=False)
        self.w3 = nn.Linear(cfg.n_embd, hidden, bias=False)
        self.w2 = nn.Linear(hidden, cfg.n_embd, bias=False)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


def make_ffn(cfg: ModelConfig) -> nn.Module:
    return SwiGLU(cfg) if cfg.arch == "modern" else FeedForward(cfg)


# ---------------------------------------------------------------------------
# Transformer block (pre-norm + residual)
# ---------------------------------------------------------------------------
class TransformerBlock(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.norm1 = make_norm(cfg.arch, cfg.n_embd)
        self.attn = MultiHeadAttention(cfg)
        self.norm2 = make_norm(cfg.arch, cfg.n_embd)
        self.ffn = make_ffn(cfg)

    def forward(self, x: torch.Tensor, rope: Optional[tuple] = None) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), rope=rope)
        x = x + self.ffn(self.norm2(x))
        return x


# ---------------------------------------------------------------------------
# Backbone
# ---------------------------------------------------------------------------
class MiniGPT(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.use_rope = cfg.arch == "modern"
        if not self.use_rope:
            self.pos_emb = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.dropout = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layer)])
        self.final_norm = make_norm(cfg.arch, cfg.n_embd)

        if self.use_rope:
            cos, sin = build_rope_cache(cfg.block_size, cfg.head_dim, device=torch.device("cpu"))
            self.register_buffer("rope_cos", cos, persistent=False)
            self.register_buffer("rope_sin", sin, persistent=False)

        self.apply(self._init_weights)
        # scaled residual init (GPT-2 style) so deep stacks stay stable at init
        for name, p in self.named_parameters():
            if name.endswith("out_proj.weight") or name.endswith("w2.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        B, T = idx.shape
        assert T <= self.cfg.block_size, f"sequence length {T} exceeds block_size {self.cfg.block_size}"

        x = self.token_emb(idx)
        if not self.use_rope:
            pos = torch.arange(T, device=idx.device)
            x = x + self.pos_emb(pos)[None, :, :]
        x = self.dropout(x)

        rope = (self.rope_cos, self.rope_sin) if self.use_rope else None
        for block in self.blocks:
            x = block(x, rope=rope)
        return self.final_norm(x)


# ---------------------------------------------------------------------------
# Language modeling head
# ---------------------------------------------------------------------------
class GPTLanguageModel(nn.Module):
    def __init__(self, backbone: MiniGPT, vocab_size: int, tie_weights: bool = True):
        super().__init__()
        self.backbone = backbone
        self.lm_head = nn.Linear(backbone.cfg.n_embd, vocab_size, bias=False)
        if tie_weights:
            self.lm_head.weight = backbone.token_emb.weight

    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None, label_smoothing: float = 0.0):
        hidden = self.backbone(idx)
        logits = self.lm_head(hidden)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                label_smoothing=label_smoothing,
            )
        return logits, loss


# ---------------------------------------------------------------------------
# Classification head
# ---------------------------------------------------------------------------
class AttentionPool(nn.Module):
    """Learned-query attention pooling over the sequence dimension."""

    def __init__(self, n_embd: int):
        super().__init__()
        self.proj = nn.Linear(n_embd, n_embd)
        self.query = nn.Parameter(torch.randn(n_embd) * 0.02)
        self.scale = n_embd ** -0.5

    def forward(self, hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # hidden: (B,T,C), mask: (B,T) bool, True = real token
        scores = (self.proj(hidden) @ self.query) * self.scale  # (B,T)
        scores = scores.masked_fill(~mask, float("-inf"))
        weights = F.softmax(scores, dim=-1).unsqueeze(-1)  # (B,T,1)
        return (weights * hidden).sum(dim=1)


def pool_last(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    last_idx = mask.sum(dim=1) - 1  # (B,)
    return hidden[torch.arange(hidden.size(0), device=hidden.device), last_idx]


def pool_mean(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask_f = mask.unsqueeze(-1).float()
    return (hidden * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1.0)


class GPTClassifier(nn.Module):
    """Attaches a 3-way classification head to a (pretrained) MiniGPT backbone.

    Sequence alignment: since the backbone is causal, the "last" pooling strategy
    extracts the final real token's hidden state (the only position that has seen
    the whole sequence) to make the decision -- the assignment-required approach.
    "mean" and "attn" are additional strategies compared in notebook 03.
    """

    def __init__(self, backbone: MiniGPT, n_classes: int = 3, pooling: str = "last"):
        super().__init__()
        assert pooling in ("last", "mean", "attn")
        self.backbone = backbone
        self.pooling = pooling
        n_embd = backbone.cfg.n_embd
        if pooling == "attn":
            self.attn_pool = AttentionPool(n_embd)
        self.head = nn.Linear(n_embd, n_classes)

    def pool(self, hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if self.pooling == "last":
            return pool_last(hidden, mask)
        if self.pooling == "mean":
            return pool_mean(hidden, mask)
        return self.attn_pool(hidden, mask)

    def forward(self, idx: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        hidden = self.backbone(idx)
        pooled = self.pool(hidden, mask)
        return self.head(pooled)

    def forward_chunks(self, idx: torch.Tensor, mask: torch.Tensor, owner: torch.Tensor, n_examples: int) -> torch.Tensor:
        """Runs the backbone once over all (chunk-level) sequences, pools each
        chunk, then averages chunk features back to one vector per source example
        (chunk-and-aggregate strategy for long documents).
        """
        hidden = self.backbone(idx)
        chunk_feats = self.pool(hidden, mask)  # (n_chunks, C)
        C = chunk_feats.size(-1)
        summed = torch.zeros(n_examples, C, device=chunk_feats.device, dtype=chunk_feats.dtype)
        counts = torch.zeros(n_examples, 1, device=chunk_feats.device, dtype=chunk_feats.dtype)
        summed.index_add_(0, owner, chunk_feats)
        counts.index_add_(0, owner, torch.ones(chunk_feats.size(0), 1, device=chunk_feats.device))
        pooled = summed / counts.clamp(min=1.0)
        return self.head(pooled)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
