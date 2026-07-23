"""Headless classifier training/fine-tuning on top of a pretrained MiniGPT backbone.

Compares, via CLI flags, the dimensions explored in notebook 03:
  --arch {vanilla,modern}           which pretrained backbone architecture (MUST match --backbone_ckpt)
  --pooling {last,mean,attn}        feature-extraction strategy
  --ft_mode {frozen,full,lora}      how much of the backbone is trainable
  --strategy {truncate,chunks}      how long reviews (> block_size) are handled
  --class_weight                    inverse-frequency class-weighted loss
  --freeze_epochs N                 gradual unfreezing: freeze backbone for N epochs, then unfreeze (ft_mode=full only)

Usage:
    python scripts/train_cls.py --pooling attn --ft_mode full --strategy truncate --run_name cls_attn_full
"""
import argparse
import sys
from functools import partial
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from torch.utils.data import DataLoader

from src.cls_utils import build_classifier, compute_class_weights, get_forward_fn
from src.config import CKPT_DIR, ClsTrainConfig, SEED, SPLITS_DIR, get_device, set_seed
from src.data import ClsDataset, collate_chunks, collate_truncate
from src.lora import trainable_parameter_count
from src.tokenizer import load_tokenizer
from src.train import train_cls


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--arch", choices=["vanilla", "modern"], default="modern")
    p.add_argument("--backbone_ckpt", default=str(CKPT_DIR / "lm_modern_best.pt"))
    p.add_argument("--pooling", choices=["last", "mean", "attn"], default="last")
    p.add_argument("--ft_mode", choices=["frozen", "full", "lora"], default="full")
    p.add_argument("--strategy", choices=["truncate", "chunks"], default="truncate")
    p.add_argument("--run_name", default="cls_run")
    p.add_argument("--block_size", type=int, default=256)
    p.add_argument("--n_layer", type=int, default=8)
    p.add_argument("--n_head", type=int, default=8)
    p.add_argument("--n_embd", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--lora_r", type=int, default=8)
    p.add_argument("--lora_alpha", type=int, default=16)
    p.add_argument("--class_weight", action="store_true")
    p.add_argument("--freeze_epochs", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(SEED)
    device = get_device()
    print(f"Device: {device} | arch={args.arch} | pooling={args.pooling} | ft_mode={args.ft_mode} | "
          f"strategy={args.strategy} | class_weight={args.class_weight} | freeze_epochs={args.freeze_epochs}")

    sp = load_tokenizer()
    train_df = pd.read_csv(SPLITS_DIR / "cls_train.csv")
    val_df = pd.read_csv(SPLITS_DIR / "cls_val.csv")
    train_df["label_id"] = train_df["label_id"].astype(int)
    val_df["label_id"] = val_df["label_id"].astype(int)

    train_ds = ClsDataset(train_df, sp, max_len=args.block_size, eos_id=sp.eos_id(), strategy=args.strategy)
    val_ds = ClsDataset(val_df, sp, max_len=args.block_size, eos_id=sp.eos_id(), strategy=args.strategy)

    collate = partial(collate_truncate if args.strategy == "truncate" else collate_chunks, pad_id=sp.pad_id())
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate, num_workers=0)
    forward_fn = get_forward_fn(args.strategy)

    model = build_classifier(
        backbone_ckpt=args.backbone_ckpt, arch=args.arch, vocab_size=sp.vocab_size(), pad_id=sp.pad_id(),
        block_size=args.block_size, n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd,
        pooling=args.pooling, ft_mode=args.ft_mode, device=device,
        lora_r=args.lora_r, lora_alpha=args.lora_alpha,
    )
    print(f"Trainable parameters: {trainable_parameter_count(model):,} / total {sum(p.numel() for p in model.parameters()):,}")

    class_weights = compute_class_weights(train_df["label_id"].tolist(), n_classes=3) if args.class_weight else None
    if class_weights is not None:
        print(f"Class weights: {class_weights.tolist()}")

    cfg = ClsTrainConfig(
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        pooling=args.pooling,
        ft_mode=args.ft_mode,
        max_chunk_len=args.block_size,
        chunk_aggregate=(args.strategy == "chunks"),
        log_path=f"{args.run_name}_metrics.csv",
        run_name=args.run_name,
    )

    if args.ft_mode == "full":
        backbone_params = [p for n, p in model.backbone.named_parameters() if p.requires_grad]
        head_params = [p for n, p in model.named_parameters() if not n.startswith("backbone.") and p.requires_grad]
        best_f1 = train_cls(
            model, train_loader, val_loader, device, cfg, CKPT_DIR / f"{args.run_name}_best.pt", forward_fn,
            backbone_params, head_params, class_weights=class_weights, freeze_backbone_epochs=args.freeze_epochs,
        )
    else:
        best_f1 = train_cls(
            model, train_loader, val_loader, device, cfg, CKPT_DIR / f"{args.run_name}_best.pt", forward_fn,
            class_weights=class_weights,
        )

    print(f"Training complete. Best val macro-F1: {best_f1:.4f}")


if __name__ == "__main__":
    main()
