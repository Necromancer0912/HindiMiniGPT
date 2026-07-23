"""Distills the trained vanilla LM (teacher) into a small student model, and
separately trains a from-scratch student of identical size for comparison --
this pair is what makes the "distilled vs scratch" plot in notebook 04 honest.

Usage:
    python scripts/distill.py --mode distill --run_name student_distilled
    python scripts/distill.py --mode scratch  --run_name student_scratch
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from torch.utils.data import DataLoader

from src.config import CKPT_DIR, LOGS_DIR, ModelConfig, SEED, SPLITS_DIR, get_device, set_seed
from src.data import LMDataset
from src.efficiency import distill_loss
from src.model import GPTLanguageModel, MiniGPT, count_parameters
from src.tokenizer import load_tokenizer
from src.train import evaluate_lm, _lr_lambda


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["distill", "scratch"], default="distill")
    p.add_argument("--teacher_ckpt", default=str(CKPT_DIR / "lm_vanilla_best.pt"))
    p.add_argument("--run_name", default="student_distilled")
    p.add_argument("--block_size", type=int, default=256)
    p.add_argument("--teacher_n_layer", type=int, default=8)
    p.add_argument("--teacher_n_head", type=int, default=8)
    p.add_argument("--teacher_n_embd", type=int, default=512)
    p.add_argument("--student_n_layer", type=int, default=3)
    p.add_argument("--student_n_head", type=int, default=4)
    p.add_argument("--student_n_embd", type=int, default=256)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--max_steps", type=int, default=3000)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--warmup_steps", type=int, default=200)
    p.add_argument("--eval_interval", type=int, default=250)
    p.add_argument("--eval_iters", type=int, default=100)
    p.add_argument("--T", type=float, default=2.0)
    p.add_argument("--alpha", type=float, default=0.5)
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(SEED)
    device = get_device()
    sp = load_tokenizer()

    train_ds = LMDataset([], sp, block_size=args.block_size, eos_id=sp.eos_id(), cache_path=SPLITS_DIR / "lm_train_tokens.pt")
    val_ds = LMDataset([], sp, block_size=args.block_size, eos_id=sp.eos_id(), cache_path=SPLITS_DIR / "lm_val_tokens.pt")
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)

    student_cfg = ModelConfig(
        vocab_size=sp.vocab_size(), block_size=args.block_size,
        n_layer=args.student_n_layer, n_head=args.student_n_head, n_embd=args.student_n_embd,
        dropout=0.1, arch="vanilla", pad_id=sp.pad_id(),
    )
    student_backbone = MiniGPT(student_cfg).to(device)
    student = GPTLanguageModel(student_backbone, vocab_size=student_cfg.vocab_size).to(device)
    print(f"Student parameters: {count_parameters(student):,}")

    teacher = None
    if args.mode == "distill":
        teacher_cfg = ModelConfig(
            vocab_size=sp.vocab_size(), block_size=args.block_size,
            n_layer=args.teacher_n_layer, n_head=args.teacher_n_head, n_embd=args.teacher_n_embd,
            dropout=0.0, arch="vanilla", pad_id=sp.pad_id(),
        )
        teacher_backbone = MiniGPT(teacher_cfg)
        teacher = GPTLanguageModel(teacher_backbone, vocab_size=teacher_cfg.vocab_size).to(device)
        ckpt = torch.load(args.teacher_ckpt, map_location=device)
        teacher.load_state_dict(ckpt["model"])
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad = False
        print(f"Loaded teacher from {args.teacher_ckpt} (val_ppl at save time: {ckpt.get('best_val_ppl', 'n/a')})")

    optimizer = torch.optim.AdamW(student.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.1)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda s: _lr_lambda(s, args.warmup_steps, args.max_steps, 0.1)
    )
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))
    amp_dtype = torch.float16 if device.type == "cuda" else torch.bfloat16

    log_path = LOGS_DIR / f"{args.run_name}_metrics.csv"
    log_path.write_text("step,train_loss,val_loss,val_ppl,elapsed_s\n", encoding="utf-8")

    best_ppl = float("inf")
    ckpt_path = CKPT_DIR / f"{args.run_name}_best.pt"
    train_iter = iter(train_loader)
    t_start = time.time()

    for step in range(1, args.max_steps + 1):
        try:
            x, y = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            x, y = next(train_iter)
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=amp_dtype):
            student_logits, ce_loss = student(x, y)
            if args.mode == "distill":
                with torch.no_grad():
                    teacher_logits, _ = teacher(x)
                loss = distill_loss(student_logits, teacher_logits, y, T=args.T, alpha=args.alpha)
            else:
                loss = ce_loss
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        if step % args.eval_interval == 0 or step == args.max_steps:
            val_loss, val_ppl = evaluate_lm(student, val_loader, device, args.eval_iters, amp=True)
            elapsed = time.time() - t_start
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(f"{step},{loss.item():.4f},{val_loss:.4f},{val_ppl:.3f},{elapsed:.1f}\n")
            print(f"[{args.run_name}] step {step}/{args.max_steps} | loss {loss.item():.4f} | val_ppl {val_ppl:.3f}")
            if val_ppl < best_ppl:
                best_ppl = val_ppl
                torch.save({"model": student.state_dict(), "step": step, "best_val_ppl": best_ppl, "config": student_cfg}, ckpt_path)

    print(f"Done. Best {args.run_name} val perplexity: {best_ppl:.3f}")


if __name__ == "__main__":
    main()
