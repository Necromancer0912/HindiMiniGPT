"""Headless language-model training, meant to be launched in the background and
monitored by tailing outputs/logs/<run_name>.csv.

Usage:
    python scripts/train_lm.py --arch vanilla --run_name lm_vanilla --max_steps 6000
    python scripts/train_lm.py --arch modern  --run_name lm_modern  --max_steps 6000
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from torch.utils.data import DataLoader

from src.config import CKPT_DIR, ModelConfig, SEED, SPLITS_DIR, TrainConfig, get_device, set_seed
from src.data import LMDataset
from src.model import GPTLanguageModel, MiniGPT, count_parameters
from src.tokenizer import load_tokenizer
from src.train import train_lm


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--arch", choices=["vanilla", "modern"], default="vanilla")
    p.add_argument("--run_name", default="lm_vanilla")
    p.add_argument("--block_size", type=int, default=256)
    p.add_argument("--n_layer", type=int, default=6)
    p.add_argument("--n_head", type=int, default=6)
    p.add_argument("--n_embd", type=int, default=384)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--grad_accum", type=int, default=2)
    p.add_argument("--max_steps", type=int, default=6000)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--warmup_steps", type=int, default=300)
    p.add_argument("--eval_interval", type=int, default=250)
    p.add_argument("--eval_iters", type=int, default=100)
    p.add_argument("--resume", action="store_true", default=True)
    p.add_argument("--no_resume", dest="resume", action="store_false")
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(SEED)
    device = get_device()
    print(f"Device: {device} | arch={args.arch} | run_name={args.run_name}")

    sp = load_tokenizer()
    train_tokens_path = SPLITS_DIR / "lm_train_tokens.pt"
    val_tokens_path = SPLITS_DIR / "lm_val_tokens.pt"
    if not train_tokens_path.exists():
        raise FileNotFoundError("Run notebook 01 first to build the tokenizer and cached token tensors.")

    train_ds = LMDataset([], sp, block_size=args.block_size, eos_id=sp.eos_id(), cache_path=train_tokens_path)
    val_ds = LMDataset([], sp, block_size=args.block_size, eos_id=sp.eos_id(), cache_path=val_tokens_path)
    print(f"Train tokens: {len(train_ds.tokens):,} | Val tokens: {len(val_ds.tokens):,}")

    pin_memory = device.type == "cuda"
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=pin_memory, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=pin_memory, drop_last=True)

    model_cfg = ModelConfig(
        vocab_size=sp.vocab_size(),
        block_size=args.block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        dropout=args.dropout,
        arch=args.arch,
        pad_id=sp.pad_id(),
    )
    backbone = MiniGPT(model_cfg).to(device)
    model = GPTLanguageModel(backbone, vocab_size=model_cfg.vocab_size, tie_weights=model_cfg.tie_weights).to(device)
    print(f"Model parameters: {count_parameters(model):,}")

    train_cfg = TrainConfig(
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        max_steps=args.max_steps,
        lr=args.lr,
        warmup_steps=args.warmup_steps,
        eval_interval=args.eval_interval,
        eval_iters=args.eval_iters,
        log_path=f"{args.run_name}_metrics.csv",
        run_name=args.run_name,
    )

    ckpt_path = CKPT_DIR / f"{args.run_name}_best.pt"
    resume_path = ckpt_path if args.resume else None

    best_ppl = train_lm(model, train_loader, val_loader, device, train_cfg, ckpt_path, resume_path=resume_path)
    print(f"Training complete. Best val perplexity: {best_ppl:.3f}")


if __name__ == "__main__":
    main()
