"""Task-Adaptive Pretraining (TAPT) -- "Don't Stop Pretraining" (Gururangan et al.,
ACL 2020, arXiv:2004.10964).

The trained Wikipedia/news LM's frozen-backbone features barely beat the majority
baseline on sentiment classification (34% vs 38%) -- a classic domain-gap symptom.
This script continues pretraining that same LM, unsupervised, on the classification
task's own review text (no labels used), so its features shift toward the review
domain before the classifier ever sees a label.

Leakage guard: the TAPT corpus is built ONLY from cls_train.csv text, split further
into a small held-out slice used purely to measure perplexity before/after -- the
144-review classification validation set is never touched here.

Usage:
    python scripts/adapt_lm.py --base_ckpt outputs/checkpoints/lm_modern_best.pt --run_name lm_adapted
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from src.config import CKPT_DIR, CORPUS_DIR, LOGS_DIR, ModelConfig, SEED, SPLITS_DIR, TrainConfig, get_device, set_seed
from src.data import LMDataset, normalize_hindi_text, read_corpus
from src.model import GPTLanguageModel, MiniGPT
from src.tokenizer import load_tokenizer
from src.train import evaluate_lm, train_lm

DAPT_KEYWORDS = ["फिल्म", "निर्देशक", "अभिनेता", "एक्टर", "सिनेमा", "बॉलीवुड", "अभिनेत्री"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base_ckpt", default=str(CKPT_DIR / "lm_modern_best.pt"))
    p.add_argument("--arch", choices=["vanilla", "modern"], default="modern")
    p.add_argument("--run_name", default="lm_adapted")
    p.add_argument("--block_size", type=int, default=256)
    p.add_argument("--n_layer", type=int, default=8)
    p.add_argument("--n_head", type=int, default=8)
    p.add_argument("--n_embd", type=int, default=512)
    p.add_argument("--tapt_val_ratio", type=float, default=0.1, help="fraction of cls_train reviews held out to measure before/after perplexity")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--max_steps", type=int, default=1000)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--warmup_steps", type=int, default=50)
    p.add_argument("--eval_interval", type=int, default=100)
    p.add_argument("--eval_iters", type=int, default=50)
    p.add_argument("--dapt", action="store_true", help="also mix in keyword-filtered entertainment docs from the LM corpus (off by default)")
    p.add_argument("--dapt_max_docs", type=int, default=3000)
    return p.parse_args()


def build_lm(arch, vocab_size, pad_id, block_size, n_layer, n_head, n_embd, device):
    cfg = ModelConfig(vocab_size=vocab_size, block_size=block_size, n_layer=n_layer, n_head=n_head,
                       n_embd=n_embd, dropout=0.1, arch=arch, pad_id=pad_id)
    backbone = MiniGPT(cfg).to(device)
    return GPTLanguageModel(backbone, vocab_size=vocab_size).to(device)


def load_dapt_docs(max_docs: int) -> list:
    """Keyword-filtered entertainment-domain docs from the LM's OWN train split
    (never touches the LM val split, so this stays leakage-safe for LM perplexity too).
    """
    lm_train_ids = set((SPLITS_DIR / "lm_train_ids.txt").read_text(encoding="utf-8").splitlines())
    rows = read_corpus(CORPUS_DIR, dedup=False)
    matched = []
    for r in rows:
        if r["id"] not in lm_train_ids:
            continue
        if any(kw in r["text"] for kw in DAPT_KEYWORDS):
            matched.append(r["text"])
            if len(matched) >= max_docs:
                break
    return matched


def main():
    args = parse_args()
    set_seed(SEED)
    device = get_device()
    sp = load_tokenizer()

    # --- Build the TAPT corpus from classification-TRAIN reviews only ---
    cls_train_df = pd.read_csv(SPLITS_DIR / "cls_train.csv")
    review_texts = [normalize_hindi_text(t) for t in cls_train_df["text"].tolist()]
    tapt_train_texts, tapt_heldout_texts = train_test_split(review_texts, test_size=args.tapt_val_ratio, random_state=SEED)
    print(f"TAPT corpus: {len(tapt_train_texts)} train reviews, {len(tapt_heldout_texts)} held-out (perplexity check only)")

    if args.dapt:
        dapt_docs = load_dapt_docs(args.dapt_max_docs)
        print(f"DAPT: adding {len(dapt_docs)} keyword-filtered entertainment docs")
        tapt_train_texts = tapt_train_texts + dapt_docs

    # --- Load base (pre-TAPT) LM and measure held-out review perplexity BEFORE adapting ---
    base_model = build_lm(args.arch, sp.vocab_size(), sp.pad_id(), args.block_size, args.n_layer, args.n_head, args.n_embd, device)
    ckpt = torch.load(args.base_ckpt, map_location=device, weights_only=False)
    base_model.load_state_dict(ckpt["model"])
    print(f"Loaded base LM from {args.base_ckpt} (arch={args.arch})")

    heldout_ds = LMDataset(tapt_heldout_texts, sp, block_size=args.block_size, eos_id=sp.eos_id())
    heldout_loader = DataLoader(heldout_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)
    before_loss, before_ppl = evaluate_lm(base_model, heldout_loader, device, max_iters=200, amp=True)
    print(f"BEFORE TAPT: review-domain val_ppl = {before_ppl:.3f}")

    # --- Continue-pretrain on the TAPT corpus ---
    tapt_train_ds = LMDataset(tapt_train_texts, sp, block_size=args.block_size, eos_id=sp.eos_id())
    tapt_train_loader = DataLoader(tapt_train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)
    print(f"TAPT train tokens: {len(tapt_train_ds.tokens):,}")

    train_cfg = TrainConfig(
        batch_size=args.batch_size, grad_accum=1, max_steps=args.max_steps, lr=args.lr,
        warmup_steps=args.warmup_steps, eval_interval=args.eval_interval, eval_iters=args.eval_iters,
        log_path=f"{args.run_name}_metrics.csv", run_name=args.run_name,
    )
    ckpt_path = CKPT_DIR / f"{args.run_name}_best.pt"
    train_lm(base_model, tapt_train_loader, heldout_loader, device, train_cfg, ckpt_path, resume_path=None)

    # --- Reload the best-adapted checkpoint and measure AFTER perplexity on the same held-out slice ---
    adapted_model = build_lm(args.arch, sp.vocab_size(), sp.pad_id(), args.block_size, args.n_layer, args.n_head, args.n_embd, device)
    adapted_ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    adapted_model.load_state_dict(adapted_ckpt["model"])
    after_loss, after_ppl = evaluate_lm(adapted_model, heldout_loader, device, max_iters=200, amp=True)
    print(f"AFTER TAPT:  review-domain val_ppl = {after_ppl:.3f}")

    delta_pct = (before_ppl - after_ppl) / before_ppl * 100
    print(f"TAPT {'REDUCED' if delta_pct > 0 else 'INCREASED'} review-domain perplexity by {abs(delta_pct):.1f}%")

    summary_path = LOGS_DIR / "tapt_before_after.csv"
    pd.DataFrame([
        {"stage": "before_tapt", "review_val_ppl": before_ppl, "n_heldout_reviews": len(tapt_heldout_texts)},
        {"stage": "after_tapt", "review_val_ppl": after_ppl, "n_heldout_reviews": len(tapt_heldout_texts)},
    ]).to_csv(summary_path, index=False)
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
