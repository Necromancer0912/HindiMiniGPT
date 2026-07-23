"""Training / evaluation loops for the language model and the classifier.

Both loops write a live CSV row per eval interval so a long background run can be
monitored by tailing the file, and both checkpoint the best model + optimizer state
so a run is resumable.
"""
import csv
import math
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader

from src.config import ClsTrainConfig, LOGS_DIR, TrainConfig


def _lr_lambda(step: int, warmup_steps: int, max_steps: int, min_lr_ratio: float):
    if step < warmup_steps:
        return (step + 1) / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    progress = min(progress, 1.0)
    cosine = 0.5 * (1 + math.cos(math.pi * progress))
    return min_lr_ratio + (1 - min_lr_ratio) * cosine


def _csv_writer(path: Path, fieldnames):
    is_new = not path.exists()
    f = open(path, "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    if is_new:
        writer.writeheader()
    return f, writer


LM_FIELDS = ["step", "train_loss", "val_loss", "val_ppl", "lr", "grad_norm", "tokens_seen", "elapsed_s", "eta_s"]


@torch.no_grad()
def evaluate_lm(model: nn.Module, loader: DataLoader, device, max_iters: int, amp: bool):
    model.eval()
    total_loss, total_tokens = 0.0, 0
    amp_dtype = torch.float16 if device.type == "cuda" else torch.bfloat16
    it = iter(loader)
    for _ in range(max_iters):
        try:
            x, y = next(it)
        except StopIteration:
            break
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp):
            _, loss = model(x, y, label_smoothing=0.0)
        n_tok = y.numel()
        total_loss += loss.item() * n_tok
        total_tokens += n_tok
    model.train()
    val_loss = total_loss / max(1, total_tokens)
    val_ppl = math.exp(min(val_loss, 20))  # clamp to avoid overflow on bad early steps
    return val_loss, val_ppl


def train_lm(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device,
    cfg: TrainConfig,
    ckpt_path: Path,
    resume_path: Optional[Path] = None,
):
    optim_groups = [
        {"params": [p for n, p in model.named_parameters() if p.dim() >= 2], "weight_decay": cfg.weight_decay},
        {"params": [p for n, p in model.named_parameters() if p.dim() < 2], "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(optim_groups, lr=cfg.lr, betas=cfg.betas)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda s: _lr_lambda(s, cfg.warmup_steps, cfg.max_steps, cfg.min_lr_ratio)
    )
    scaler = torch.amp.GradScaler("cuda", enabled=(cfg.amp and device.type == "cuda"))
    amp_dtype = torch.float16 if device.type == "cuda" else torch.bfloat16

    start_step = 0
    best_val_ppl = float("inf")
    if resume_path is not None and resume_path.exists():
        ckpt = torch.load(resume_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_step = ckpt["step"]
        best_val_ppl = ckpt.get("best_val_ppl", float("inf"))
        print(f"Resumed from step {start_step}, best_val_ppl={best_val_ppl:.3f}")

    log_path = LOGS_DIR / cfg.log_path
    log_file, writer = _csv_writer(log_path, LM_FIELDS)

    model.train()
    train_iter = iter(train_loader)
    tokens_seen = 0
    t_start = time.time()
    running_loss = 0.0
    running_count = 0

    step = start_step
    while step < cfg.max_steps:
        optimizer.zero_grad(set_to_none=True)
        accum_loss = 0.0
        for _ in range(cfg.grad_accum):
            try:
                x, y = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                x, y = next(train_iter)
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=cfg.amp):
                _, loss = model(x, y, label_smoothing=cfg.label_smoothing)
                loss = loss / cfg.grad_accum
            scaler.scale(loss).backward()
            accum_loss += loss.item()
            tokens_seen += x.numel()

        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        running_loss += accum_loss
        running_count += 1
        step += 1

        if step % cfg.eval_interval == 0 or step == cfg.max_steps:
            val_loss, val_ppl = evaluate_lm(model, val_loader, device, cfg.eval_iters, cfg.amp)
            train_loss = running_loss / max(1, running_count)
            running_loss, running_count = 0.0, 0

            elapsed = time.time() - t_start
            steps_left = cfg.max_steps - step
            step_time = elapsed / max(1, step - start_step)
            eta = steps_left * step_time

            row = {
                "step": step,
                "train_loss": round(train_loss, 4),
                "val_loss": round(val_loss, 4),
                "val_ppl": round(val_ppl, 3),
                "lr": scheduler.get_last_lr()[0],
                "grad_norm": round(float(grad_norm), 4),
                "tokens_seen": tokens_seen,
                "elapsed_s": round(elapsed, 1),
                "eta_s": round(eta, 1),
            }
            writer.writerow(row)
            log_file.flush()
            print(
                f"step {step}/{cfg.max_steps} | train_loss {train_loss:.4f} | "
                f"val_loss {val_loss:.4f} | val_ppl {val_ppl:.3f} | "
                f"tok/s {tokens_seen/elapsed:.0f} | eta {eta/60:.1f}min"
            )

            if val_ppl < best_val_ppl:
                best_val_ppl = val_ppl
                torch.save(
                    {
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "step": step,
                        "best_val_ppl": best_val_ppl,
                    },
                    ckpt_path,
                )

    log_file.close()
    return best_val_ppl


CLS_FIELDS = ["epoch", "train_loss", "val_loss", "val_acc", "val_macro_f1", "lr", "elapsed_s"]


@torch.no_grad()
def evaluate_cls(model, loader, device, amp: bool, forward_fn):
    model.eval()
    all_preds, all_labels = [], []
    total_loss, n = 0.0, 0
    amp_dtype = torch.float16 if device.type == "cuda" else torch.bfloat16
    for batch in loader:
        logits, labels = forward_fn(model, batch, device, amp_dtype, amp)
        loss = F.cross_entropy(logits, labels)
        total_loss += loss.item() * labels.size(0)
        n += labels.size(0)
        all_preds.extend(logits.argmax(-1).cpu().tolist())
        all_labels.extend(labels.cpu().tolist())
    model.train()
    acc = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average="macro")
    return total_loss / max(1, n), acc, macro_f1, all_preds, all_labels


def train_cls(
    model,
    train_loader,
    val_loader,
    device,
    cfg: ClsTrainConfig,
    ckpt_path: Path,
    forward_fn,
    backbone_params=None,
    head_params=None,
    class_weights: Optional[torch.Tensor] = None,
    freeze_backbone_epochs: int = 0,
):
    # Gradual unfreezing (ULMFiT-style) only makes sense when we hold separate
    # backbone/head param groups to begin with (i.e. ft_mode="full"); a permanently
    # frozen or LoRA backbone has nothing to "unfreeze" at epoch k.
    use_gradual_unfreeze = freeze_backbone_epochs > 0 and backbone_params is not None and head_params is not None

    def build_optimizer(backbone_frozen: bool):
        if backbone_frozen:
            return torch.optim.AdamW(head_params, lr=cfg.lr, weight_decay=cfg.weight_decay)
        if backbone_params is not None and head_params is not None:
            return torch.optim.AdamW(
                [
                    {"params": backbone_params, "lr": cfg.lr * cfg.backbone_lr_mult},
                    {"params": head_params, "lr": cfg.lr},
                ],
                weight_decay=cfg.weight_decay,
            )
        return torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    if use_gradual_unfreeze:
        for p in backbone_params:
            p.requires_grad_(False)
        optimizer = build_optimizer(backbone_frozen=True)
    else:
        optimizer = build_optimizer(backbone_frozen=False)

    class_weights_t = class_weights.to(device) if class_weights is not None else None

    scaler = torch.amp.GradScaler("cuda", enabled=(cfg.amp and device.type == "cuda"))
    amp_dtype = torch.float16 if device.type == "cuda" else torch.bfloat16

    log_path = LOGS_DIR / cfg.log_path
    log_file, writer = _csv_writer(log_path, CLS_FIELDS)

    best_val_f1 = -1.0
    patience = 0
    t_start = time.time()

    for epoch in range(1, cfg.epochs + 1):
        if use_gradual_unfreeze and epoch == freeze_backbone_epochs + 1:
            for p in backbone_params:
                p.requires_grad_(True)
            optimizer = build_optimizer(backbone_frozen=False)
            print(f"Unfroze backbone at epoch {epoch} (rebuilt optimizer with discriminative LR)")

        model.train()
        running_loss, n_seen = 0.0, 0
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            logits, labels = forward_fn(model, batch, device, amp_dtype, cfg.amp)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=cfg.amp):
                loss = F.cross_entropy(logits, labels, weight=class_weights_t)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item() * labels.size(0)
            n_seen += labels.size(0)

        train_loss = running_loss / max(1, n_seen)
        val_loss, val_acc, val_f1, _, _ = evaluate_cls(model, val_loader, device, cfg.amp, forward_fn)
        elapsed = time.time() - t_start

        row = {
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "val_loss": round(val_loss, 4),
            "val_acc": round(val_acc, 4),
            "val_macro_f1": round(val_f1, 4),
            "lr": optimizer.param_groups[-1]["lr"],
            "elapsed_s": round(elapsed, 1),
        }
        writer.writerow(row)
        log_file.flush()
        print(
            f"epoch {epoch}/{cfg.epochs} | train_loss {train_loss:.4f} | "
            f"val_loss {val_loss:.4f} | val_acc {val_acc:.4f} | val_macro_f1 {val_f1:.4f}"
        )

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience = 0
            torch.save({"model": model.state_dict(), "epoch": epoch, "val_acc": val_acc, "val_f1": val_f1}, ckpt_path)
        else:
            patience += 1
            if patience >= cfg.early_stop_patience:
                print(f"Early stopping at epoch {epoch} (no improvement for {patience} epochs)")
                break

    log_file.close()
    return best_val_f1
