"""Correctness sanity checks for the from-scratch attention implementation."""
import torch
import torch.nn.functional as F

from src.config import ModelConfig
from src.model import MultiHeadAttention


@torch.no_grad()
def check_attention_matches_reference(tol: float = 1e-4) -> bool:
    """Compares our hand-written causal attention against PyTorch's fused
    scaled_dot_product_attention(is_causal=True) on identical Q/K/V, vanilla arch.
    """
    torch.manual_seed(0)
    B, T, n_head, head_dim = 2, 16, 4, 8
    C = n_head * head_dim
    cfg = ModelConfig(n_embd=C, n_head=n_head, block_size=T, dropout=0.0, arch="vanilla")
    mha = MultiHeadAttention(cfg).eval()

    x = torch.randn(B, T, C)
    q, k, v = mha.qkv_proj(x).split(C, dim=2)
    q = q.view(B, T, n_head, head_dim).transpose(1, 2)
    k = k.view(B, T, n_head, head_dim).transpose(1, 2)
    v = v.view(B, T, n_head, head_dim).transpose(1, 2)

    ref = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    ref = ref.transpose(1, 2).contiguous().view(B, T, C)
    ref = mha.out_proj(ref)

    ours = mha(x)
    max_diff = (ref - ours).abs().max().item()
    assert max_diff < tol, f"attention mismatch vs reference, max diff={max_diff}"
    return True


@torch.no_grad()
def check_causal_property() -> bool:
    """Verifies token t's output is unaffected by changes to tokens after t."""
    torch.manual_seed(0)
    B, T, n_head, head_dim = 1, 10, 2, 4
    C = n_head * head_dim
    cfg = ModelConfig(n_embd=C, n_head=n_head, block_size=T, dropout=0.0, arch="vanilla")
    mha = MultiHeadAttention(cfg).eval()

    x = torch.randn(B, T, C)
    x_mod = x.clone()
    t_cut = 4
    x_mod[:, t_cut + 1 :, :] = torch.randn_like(x_mod[:, t_cut + 1 :, :])

    out1 = mha(x)
    out2 = mha(x_mod)
    max_diff = (out1[:, : t_cut + 1] - out2[:, : t_cut + 1]).abs().max().item()
    assert max_diff < 1e-5, f"causal leakage detected, max diff={max_diff}"
    return True


if __name__ == "__main__":
    check_attention_matches_reference()
    check_causal_property()
    print("All attention sanity checks passed.")
