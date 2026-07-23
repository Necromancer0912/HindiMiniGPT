"""Manual LoRA (no peft dependency -- peft/bitsandbytes are flaky on Windows).

Wraps an existing nn.Linear, freezing its weight and adding a low-rank update
`(alpha/r) * B(A(x))` that is the only trainable part.
"""
import math

import torch
import torch.nn as nn

from src.model import MiniGPT, MultiHeadAttention


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int = 8, alpha: int = 16):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False

        in_f, out_f = base.in_features, base.out_features
        self.lora_A = nn.Parameter(torch.zeros(r, in_f))
        self.lora_B = nn.Parameter(torch.zeros(out_f, r))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.scale = alpha / r

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        lora_out = (x @ self.lora_A.t()) @ self.lora_B.t()
        return base_out + self.scale * lora_out


def inject_lora(backbone: MiniGPT, r: int = 8, alpha: int = 16) -> int:
    """Replaces qkv_proj/out_proj in every attention block with LoRA-wrapped
    versions in place. Returns the number of LoRA-wrapped linears.
    """
    count = 0
    for module in backbone.modules():
        if isinstance(module, MultiHeadAttention):
            module.qkv_proj = LoRALinear(module.qkv_proj, r=r, alpha=alpha)
            module.out_proj = LoRALinear(module.out_proj, r=r, alpha=alpha)
            count += 2
    return count


def freeze_backbone(backbone: MiniGPT) -> None:
    for p in backbone.parameters():
        p.requires_grad = False


def trainable_parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
