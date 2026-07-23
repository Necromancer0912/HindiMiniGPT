"""Orchestrates the full rigorous-evaluation pipeline for Task 3 and writes every
table notebook 03 needs:

  outputs/logs/cls_baselines.csv          majority + TF-IDF (single-split & 5-fold CV)
  outputs/logs/cls_gpt_cv.csv             best GPT config, 5-fold CV, non-adapted vs TAPT-adapted
  outputs/logs/cls_tapt_before_after.csv  frozen/full/attn on Wikipedia-LM vs TAPT-adapted-LM (fixed split)
  outputs/logs/cls_ensemble.csv           GPT+TF-IDF late fusion, weight chosen on OOF train probs only

Prerequisites (run first):
  python scripts/adapt_lm.py --run_name lm_adapted
  python scripts/train_cls.py --arch modern --backbone_ckpt outputs/checkpoints/lm_modern_best.pt  --ft_mode frozen --pooling attn --run_name cls_frozen_wiki
  python scripts/train_cls.py --arch modern --backbone_ckpt outputs/checkpoints/lm_adapted_best.pt --ft_mode frozen --pooling attn --run_name cls_frozen_tapt
  python scripts/train_cls.py --arch modern --backbone_ckpt outputs/checkpoints/lm_adapted_best.pt --ft_mode full   --pooling attn --class_weight --freeze_epochs 2 --run_name cls_best_tapt
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score

from src.baselines import fit_tfidf_oof_probs, majority_baseline, tfidf_baseline, tfidf_baseline_cv
from src.config import CKPT_DIR, LOGS_DIR, ModelConfig, SEED, SPLITS_DIR, get_device, set_seed
from src.cls_utils import get_forward_fn
from src.cross_val import CVRunConfig, _predict_probs, stratified_cv
from src.data import ClsDataset, collate_truncate
from src.model import GPTClassifier, MiniGPT
from src.tokenizer import load_tokenizer
from torch.utils.data import DataLoader
from functools import partial

BEST_CFG = CVRunConfig(arch="modern", pooling="attn", ft_mode="full", strategy="truncate",
                        class_weight=True, freeze_epochs=2, epochs=25, lr=5e-5)


def run_baselines(train_df, val_df, full_df):
    print("\n=== 1. Classical baselines ===")
    rows = []
    maj = majority_baseline(train_df["label_id"], val_df["label_id"])
    rows.append({"model": "majority", "split": "single", "acc": maj["acc"], "macro_f1": maj["macro_f1"], "acc_std": None, "macro_f1_std": None})

    tfidf_single = tfidf_baseline(train_df, val_df, model="svc")
    rows.append({"model": "tfidf_svc", "split": "single", "acc": tfidf_single["acc"], "macro_f1": tfidf_single["macro_f1"], "acc_std": None, "macro_f1_std": None})

    tfidf_cv = tfidf_baseline_cv(full_df, model="svc", n_splits=5, seed=SEED)
    rows.append({"model": "tfidf_svc", "split": "cv5", "acc": tfidf_cv["acc_mean"], "macro_f1": tfidf_cv["macro_f1_mean"],
                 "acc_std": tfidf_cv["acc_std"], "macro_f1_std": tfidf_cv["macro_f1_std"]})

    df_out = pd.DataFrame(rows)
    df_out.to_csv(LOGS_DIR / "cls_baselines.csv", index=False)
    print(df_out.to_string(index=False))
    return tfidf_single


def run_gpt_cv(train_df, sp, device):
    print("\n=== 2. GPT classifier 5-fold CV: non-adapted vs TAPT-adapted ===")
    rows = []
    results = {}
    for tapt in (False, True):
        tag = "tapt" if tapt else "no_tapt"
        base_ckpt = CKPT_DIR / "lm_modern_best.pt"  # same base LM either way; TAPT adapts it per-fold inside stratified_cv
        print(f"-- {tag} --")
        result = stratified_cv(train_df, sp, str(base_ckpt), BEST_CFG, device, n_splits=5, seed=SEED, tapt=tapt)
        results[tag] = result
        rows.append({
            "config": f"attn_full_{tag}", "acc_mean": result["acc_mean"], "acc_std": result["acc_std"],
            "macro_f1_mean": result["macro_f1_mean"], "macro_f1_std": result["macro_f1_std"],
        })

    df_out = pd.DataFrame(rows)
    df_out.to_csv(LOGS_DIR / "cls_gpt_cv.csv", index=False)
    print(df_out.to_string(index=False))
    return results


def compile_tapt_before_after():
    print("\n=== 3. TAPT before/after on the fixed split (frozen & full fine-tune) ===")
    run_names = ["cls_frozen_wiki", "cls_frozen_tapt", "cls_best_tapt"]
    rows = []
    for run in run_names:
        path = LOGS_DIR / f"{run}_metrics.csv"
        if not path.exists():
            print(f"  MISSING: {path} -- run scripts/train_cls.py for '{run}' first (see this file's docstring)")
            continue
        log = pd.read_csv(path)
        best = log.loc[log["val_macro_f1"].idxmax()]
        rows.append({"run": run, "best_epoch": int(best["epoch"]), "val_acc": best["val_acc"], "val_macro_f1": best["val_macro_f1"]})
    if not rows:
        print("  No fixed-split TAPT runs found yet -- skipping cls_tapt_before_after.csv")
        return None
    df_out = pd.DataFrame(rows)
    df_out.to_csv(LOGS_DIR / "cls_tapt_before_after.csv", index=False)
    print(df_out.to_string(index=False))
    return df_out


def _load_gpt_classifier_checkpoint(ckpt_path, arch, pooling, sp, device):
    """Reloads a fully-trained GPTClassifier checkpoint (backbone+head together) --
    different from cls_utils.build_classifier, which transplants an LM backbone
    into a *fresh* classifier head; here the head was already trained and saved.
    """
    model_cfg = ModelConfig(vocab_size=sp.vocab_size(), block_size=256, n_layer=8, n_head=8,
                             n_embd=512, dropout=0.0, arch=arch, pad_id=sp.pad_id())
    backbone = MiniGPT(model_cfg)
    model = GPTClassifier(backbone, n_classes=3, pooling=pooling).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


def run_ensemble(train_df, val_df, tfidf_single, gpt_cv_results, sp, device):
    print("\n=== 4. Neural + lexical ensemble (weight chosen on OOF train probs only) ===")
    best_ckpt = CKPT_DIR / "cls_best_tapt_best.pt"
    if not best_ckpt.exists() or gpt_cv_results is None:
        print(f"  MISSING: {best_ckpt} or GPT CV results -- run the prerequisite scripts first. Skipping ensemble.")
        return

    # --- out-of-fold probabilities on TRAIN only (never touches val) ---
    gpt_oof_probs = gpt_cv_results["tapt"]["oof_probs"]
    gpt_oof_labels = gpt_cv_results["tapt"]["oof_labels"]
    tfidf_oof_probs = fit_tfidf_oof_probs(train_df, model="logreg", n_splits=5, seed=SEED)

    ws = np.linspace(0.0, 1.0, 21)
    oof_f1s = []
    for w in ws:
        ensemble_probs = w * gpt_oof_probs + (1 - w) * tfidf_oof_probs
        preds = ensemble_probs.argmax(axis=-1)
        oof_f1s.append(f1_score(gpt_oof_labels, preds, average="macro"))
    best_w = float(ws[int(np.argmax(oof_f1s))])
    print(f"  Chosen ensemble weight (OOF-selected): w_gpt={best_w:.2f}, w_tfidf={1-best_w:.2f}  (OOF macro_f1={max(oof_f1s):.4f})")

    # --- apply the chosen weight ONCE on the held-out val set ---
    gpt_model = _load_gpt_classifier_checkpoint(best_ckpt, arch=BEST_CFG.arch, pooling=BEST_CFG.pooling, sp=sp, device=device)
    val_ds = ClsDataset(val_df, sp, max_len=BEST_CFG.block_size, eos_id=sp.eos_id(), strategy="truncate")
    val_loader = DataLoader(val_ds, batch_size=BEST_CFG.batch_size, shuffle=False, collate_fn=partial(collate_truncate, pad_id=sp.pad_id()), num_workers=0)
    gpt_val_probs, val_labels = _predict_probs(gpt_model, val_loader, device, get_forward_fn("truncate"))
    gpt_val_preds = gpt_val_probs.argmax(axis=-1)

    tfidf_val_probs = tfidf_single["probs"]
    tfidf_val_preds = tfidf_single["preds"]

    ensemble_val_probs = best_w * gpt_val_probs + (1 - best_w) * tfidf_val_probs
    ensemble_val_preds = ensemble_val_probs.argmax(axis=-1)

    rows = [
        {"model": "gpt_only (cls_best_tapt)", "acc": accuracy_score(val_labels, gpt_val_preds), "macro_f1": f1_score(val_labels, gpt_val_preds, average="macro")},
        {"model": "tfidf_only", "acc": tfidf_single["acc"], "macro_f1": tfidf_single["macro_f1"]},
        {"model": f"ensemble (w_gpt={best_w:.2f})", "acc": accuracy_score(val_labels, ensemble_val_preds), "macro_f1": f1_score(val_labels, ensemble_val_preds, average="macro")},
    ]
    df_out = pd.DataFrame(rows)
    df_out.to_csv(LOGS_DIR / "cls_ensemble.csv", index=False)
    print(df_out.to_string(index=False))


def main():
    set_seed(SEED)
    device = get_device()
    sp = load_tokenizer()

    train_df = pd.read_csv(SPLITS_DIR / "cls_train.csv")
    val_df = pd.read_csv(SPLITS_DIR / "cls_val.csv")
    train_df["label_id"] = train_df["label_id"].astype(int)
    val_df["label_id"] = val_df["label_id"].astype(int)
    full_df = pd.concat([train_df, val_df], ignore_index=True)

    tfidf_single = run_baselines(train_df, val_df, full_df)
    gpt_cv_results = run_gpt_cv(train_df, sp, device)
    compile_tapt_before_after()
    run_ensemble(train_df, val_df, tfidf_single, gpt_cv_results, sp, device)

    print("\nDone. Wrote outputs/logs/cls_baselines.csv, cls_gpt_cv.csv, cls_tapt_before_after.csv (if prereqs existed), cls_ensemble.csv")


if __name__ == "__main__":
    main()
