"""Shared classifier-construction and forward-pass helpers.

Centralizes what scripts/train_cls.py and src/cross_val.py both need, so backbone
loading (in particular the vanilla-vs-modern architecture switch) only has one
implementation to get right.
"""
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from src.config import ModelConfig
from src.lora import freeze_backbone, inject_lora
from src.model import GPTClassifier, MiniGPT


def compute_class_weights(labels, n_classes: int = 3) -> torch.Tensor:
    """Inverse-frequency class weights, normalized to mean 1.0 so the overall loss
    scale (and therefore the learning rate) is unaffected -- only the relative
    per-class weighting changes.
    """
    counts = np.bincount(labels, minlength=n_classes).astype(np.float64)
    counts[counts == 0] = 1.0  # guard against an empty class in a small fold
    weights = 1.0 / counts
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


def truncate_forward(model, batch, device, amp_dtype, amp):
    ids, mask, labels = batch
    ids, mask, labels = ids.to(device), mask.to(device), labels.to(device)
    with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp):
        logits = model(ids, mask)
    return logits, labels


def chunks_forward(model, batch, device, amp_dtype, amp):
    ids, mask, owner, labels = batch
    ids, mask, owner, labels = ids.to(device), mask.to(device), owner.to(device), labels.to(device)
    n_examples = labels.size(0)
    with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp):
        logits = model.forward_chunks(ids, mask, owner, n_examples)
    return logits, labels


def get_forward_fn(strategy: str):
    return truncate_forward if strategy == "truncate" else chunks_forward


def build_backbone(backbone_ckpt: Path, arch: str, vocab_size: int, pad_id: int, block_size: int, n_layer: int, n_head: int, n_embd: int, device) -> MiniGPT:
    """Builds a MiniGPT and loads pretrained backbone weights from an LM checkpoint.

    `arch` MUST match the checkpoint's architecture (vanilla uses learned position
    embeddings + LayerNorm; modern uses RoPE + RMSNorm) -- loading a modern checkpoint
    into a vanilla-shaped backbone silently drops the wrong keys instead of erroring.
    """
    model_cfg = ModelConfig(
        vocab_size=vocab_size, block_size=block_size, n_layer=n_layer, n_head=n_head,
        n_embd=n_embd, dropout=0.1, arch=arch, pad_id=pad_id,
    )
    backbone = MiniGPT(model_cfg)
    backbone_ckpt = Path(backbone_ckpt)
    if backbone_ckpt.exists():
        ckpt = torch.load(backbone_ckpt, map_location="cpu", weights_only=False)
        state = ckpt["model"]
        backbone_state = {k[len("backbone."):]: v for k, v in state.items() if k.startswith("backbone.")}
        missing, unexpected = backbone.load_state_dict(backbone_state, strict=False)
        if missing or unexpected:
            print(f"WARNING loading {backbone_ckpt}: missing={len(missing)} unexpected={len(unexpected)} "
                  f"-- check that arch='{arch}' matches how this checkpoint was trained")
        else:
            print(f"Loaded pretrained backbone from {backbone_ckpt} (arch={arch}, missing=0, unexpected=0)")
    else:
        print(f"WARNING: {backbone_ckpt} not found -- randomly initialized backbone.")
    return backbone.to(device)


def build_classifier(
    backbone_ckpt: Path,
    arch: str,
    vocab_size: int,
    pad_id: int,
    block_size: int,
    n_layer: int,
    n_head: int,
    n_embd: int,
    pooling: str,
    ft_mode: str,
    device,
    lora_r: int = 8,
    lora_alpha: int = 16,
    n_classes: int = 3,
) -> GPTClassifier:
    """One-stop constructor: load backbone -> apply ft_mode (frozen/lora/full) -> wrap head."""
    backbone = build_backbone(backbone_ckpt, arch, vocab_size, pad_id, block_size, n_layer, n_head, n_embd, device)

    if ft_mode == "frozen":
        freeze_backbone(backbone)
    elif ft_mode == "lora":
        freeze_backbone(backbone)
        n = inject_lora(backbone, r=lora_r, alpha=lora_alpha)
        print(f"Injected LoRA into {n} linear layers")
        backbone = backbone.to(device)  # re-home newly created LoRA params on device
    elif ft_mode != "full":
        raise ValueError(f"unknown ft_mode: {ft_mode}")

    model = GPTClassifier(backbone, n_classes=n_classes, pooling=pooling).to(device)
    return model
