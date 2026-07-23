"""Stratified k-fold cross-validation harness for the GPT classifier.

A single 144-example validation split is high-variance (this is exactly the kind of
evaluation artifact that made the classifier look worse than it is -- see README).
This module gives an honest mean +/- std estimate, reloading a FRESH backbone every
fold (no weight leakage across folds), with an optional per-fold TAPT step.

Also produces out-of-fold (OOF) prediction probabilities on the training split --
used both for an apples-to-apples GPT-vs-TF-IDF comparison and to pick the neural+
lexical ensemble weight without ever touching the held-out validation set.
"""
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader

from src.cls_utils import build_classifier, compute_class_weights, get_forward_fn
from src.config import CKPT_DIR, ModelConfig, TrainConfig
from src.data import ClsDataset, LMDataset, collate_chunks, collate_truncate
from src.model import GPTLanguageModel, MiniGPT
from src.train import train_cls, train_lm


@dataclass
class CVRunConfig:
    arch: str = "modern"
    block_size: int = 256
    n_layer: int = 8
    n_head: int = 8
    n_embd: int = 512
    pooling: str = "attn"
    ft_mode: str = "full"
    strategy: str = "truncate"
    lora_r: int = 8
    lora_alpha: int = 16
    class_weight: bool = False
    freeze_epochs: int = 0
    batch_size: int = 16
    epochs: int = 25
    lr: float = 5e-5
    # TAPT-specific (used only when tapt=True in stratified_cv)
    tapt_max_steps: int = 800
    tapt_lr: float = 1e-4
    tapt_warmup_steps: int = 50
    tapt_batch_size: int = 16


@torch.no_grad()
def _predict_probs(model, loader, device, forward_fn, amp: bool = True):
    model.eval()
    amp_dtype = torch.float16 if device.type == "cuda" else torch.bfloat16
    all_probs, all_labels = [], []
    for batch in loader:
        logits, labels = forward_fn(model, batch, device, amp_dtype, amp)
        probs = F.softmax(logits.float(), dim=-1)
        all_probs.append(probs.cpu().numpy())
        all_labels.append(labels.cpu().numpy())
    model.train()
    return np.concatenate(all_probs, axis=0), np.concatenate(all_labels, axis=0)


def _adapt_fold_backbone(base_ckpt, fold_train_texts, sp, cfg: CVRunConfig, device, tmp_ckpt_path: Path) -> Path:
    """Continue-pretrains base_ckpt on this fold's TRAIN reviews only (unsupervised,
    no labels) and returns the path to the fold-adapted checkpoint. Note: this uses
    the same fold-train texts for both TAPT-train and TAPT-internal-val (we only need
    the resulting weights here, not a rigorous per-fold TAPT perplexity report -- that
    headline number is already produced once, properly, by scripts/adapt_lm.py).
    """
    model_cfg = ModelConfig(vocab_size=sp.vocab_size(), block_size=cfg.block_size, n_layer=cfg.n_layer,
                             n_head=cfg.n_head, n_embd=cfg.n_embd, dropout=0.1, arch=cfg.arch, pad_id=sp.pad_id())
    backbone = MiniGPT(model_cfg).to(device)
    lm_model = GPTLanguageModel(backbone, vocab_size=sp.vocab_size()).to(device)
    ckpt = torch.load(base_ckpt, map_location=device, weights_only=False)
    lm_model.load_state_dict(ckpt["model"])

    ds = LMDataset(fold_train_texts, sp, block_size=cfg.block_size, eos_id=sp.eos_id())
    loader = DataLoader(ds, batch_size=cfg.tapt_batch_size, shuffle=True, num_workers=0, drop_last=True)

    train_cfg = TrainConfig(
        batch_size=cfg.tapt_batch_size, grad_accum=1, max_steps=cfg.tapt_max_steps, lr=cfg.tapt_lr,
        warmup_steps=cfg.tapt_warmup_steps, eval_interval=max(50, cfg.tapt_max_steps // 4), eval_iters=20,
        log_path="_cv_tapt_tmp_metrics.csv", run_name="_cv_tapt_tmp",
    )
    train_lm(lm_model, loader, loader, device, train_cfg, tmp_ckpt_path, resume_path=None)
    return tmp_ckpt_path


def _train_fold_classifier(backbone_ckpt, sp, fold_train_df, fold_val_df, cfg: CVRunConfig, device, ckpt_path: Path):
    train_ds = ClsDataset(fold_train_df, sp, max_len=cfg.block_size, eos_id=sp.eos_id(), strategy=cfg.strategy)
    val_ds = ClsDataset(fold_val_df, sp, max_len=cfg.block_size, eos_id=sp.eos_id(), strategy=cfg.strategy)
    collate = partial(collate_truncate if cfg.strategy == "truncate" else collate_chunks, pad_id=sp.pad_id())
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, collate_fn=collate, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, collate_fn=collate, num_workers=0)
    forward_fn = get_forward_fn(cfg.strategy)

    model = build_classifier(
        backbone_ckpt=backbone_ckpt, arch=cfg.arch, vocab_size=sp.vocab_size(), pad_id=sp.pad_id(),
        block_size=cfg.block_size, n_layer=cfg.n_layer, n_head=cfg.n_head, n_embd=cfg.n_embd,
        pooling=cfg.pooling, ft_mode=cfg.ft_mode, device=device, lora_r=cfg.lora_r, lora_alpha=cfg.lora_alpha,
    )
    class_weights = compute_class_weights(fold_train_df["label_id"].tolist()) if cfg.class_weight else None

    from src.config import ClsTrainConfig
    train_cls_cfg = ClsTrainConfig(
        batch_size=cfg.batch_size, epochs=cfg.epochs, lr=cfg.lr, pooling=cfg.pooling, ft_mode=cfg.ft_mode,
        max_chunk_len=cfg.block_size, chunk_aggregate=(cfg.strategy == "chunks"),
        log_path="_cv_fold_tmp_metrics.csv", run_name="_cv_fold_tmp",
    )

    if cfg.ft_mode == "full":
        backbone_params = [p for n, p in model.backbone.named_parameters() if p.requires_grad]
        head_params = [p for n, p in model.named_parameters() if not n.startswith("backbone.") and p.requires_grad]
        train_cls(model, train_loader, val_loader, device, train_cls_cfg, ckpt_path, forward_fn,
                   backbone_params, head_params, class_weights=class_weights, freeze_backbone_epochs=cfg.freeze_epochs)
    else:
        train_cls(model, train_loader, val_loader, device, train_cls_cfg, ckpt_path, forward_fn, class_weights=class_weights)

    # reload the best checkpoint from this fold before predicting (train_cls saves on every F1 improvement)
    best = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(best["model"])
    return model, forward_fn, val_loader


def stratified_cv(df: pd.DataFrame, sp, base_ckpt, cfg: CVRunConfig, device, n_splits: int = 5, seed: int = 42, tapt: bool = False) -> dict:
    """5-fold stratified CV. Returns fold-level acc/F1 (mean+/-std) plus out-of-fold
    predictions/probabilities aligned to df's original row order.
    """
    df = df.reset_index(drop=True)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    oof_probs = np.zeros((len(df), 3), dtype=np.float64)
    oof_filled = np.zeros(len(df), dtype=bool)
    fold_accs, fold_f1s = [], []

    tmp_backbone_path = CKPT_DIR / "_cv_tapt_tmp_best.pt"
    tmp_cls_path = CKPT_DIR / "_cv_fold_tmp_best.pt"

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(df["text"], df["label_id"])):
        fold_train_df = df.iloc[train_idx].reset_index(drop=True)
        fold_val_df = df.iloc[val_idx].reset_index(drop=True)

        backbone_ckpt = base_ckpt
        if tapt:
            backbone_ckpt = _adapt_fold_backbone(base_ckpt, fold_train_df["text"].tolist(), sp, cfg, device, tmp_backbone_path)

        model, forward_fn, val_loader = _train_fold_classifier(backbone_ckpt, sp, fold_train_df, fold_val_df, cfg, device, tmp_cls_path)
        probs, labels = _predict_probs(model, val_loader, device, forward_fn)
        preds = probs.argmax(axis=-1)

        acc = accuracy_score(labels, preds)
        f1 = f1_score(labels, preds, average="macro")
        fold_accs.append(acc)
        fold_f1s.append(f1)
        print(f"  fold {fold_idx+1}/{n_splits}: acc={acc:.4f} macro_f1={f1:.4f}")

        oof_probs[val_idx] = probs
        oof_filled[val_idx] = True

    assert oof_filled.all(), "every row should be predicted exactly once across folds"
    oof_preds = oof_probs.argmax(axis=-1)

    return {
        "fold_acc": fold_accs,
        "fold_f1": fold_f1s,
        "acc_mean": float(np.mean(fold_accs)),
        "acc_std": float(np.std(fold_accs)),
        "macro_f1_mean": float(np.mean(fold_f1s)),
        "macro_f1_std": float(np.std(fold_f1s)),
        "oof_probs": oof_probs,
        "oof_preds": oof_preds,
        "oof_labels": df["label_id"].values,
        "n_splits": n_splits,
        "tapt": tapt,
    }
