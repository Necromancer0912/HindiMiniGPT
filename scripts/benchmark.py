"""Builds the quality-vs-cost Pareto table across FP32 / FP16 / INT8 / distilled
variants: perplexity, checkpoint size, latency, throughput, peak VRAM.

Usage:
    python scripts/benchmark.py
Writes outputs/logs/efficiency_pareto.csv, consumed by notebook 04.
"""
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from torch.utils.data import DataLoader

from src.config import CKPT_DIR, LOGS_DIR, ModelConfig, SPLITS_DIR, get_device
from src.data import LMDataset
from src.efficiency import (
    benchmark_generation,
    dynamic_quantize_int8,
    state_dict_size_mb,
    state_dict_size_mb_quantized,
)
from src.model import GPTLanguageModel, MiniGPT
from src.tokenizer import load_tokenizer
from src.train import evaluate_lm

FIELDS = ["variant", "device", "val_ppl", "size_mb", "latency_per_token_ms", "throughput_tok_s", "peak_vram_mb"]


def load_lm(ckpt_path: Path, cfg: ModelConfig, device):
    backbone = MiniGPT(cfg)
    model = GPTLanguageModel(backbone, vocab_size=cfg.vocab_size).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


def main():
    device = get_device()
    sp = load_tokenizer()
    val_ds = LMDataset([], sp, block_size=256, eos_id=sp.eos_id(), cache_path=SPLITS_DIR / "lm_val_tokens.pt")
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=True, num_workers=0, drop_last=True)

    prompt = sp.encode("भारत एक विशाल देश है", out_type=int)
    rows = []

    # --- FP32 teacher, GPU --- (must match the architecture actually trained by scripts/train_lm.py)
    teacher_cfg = ModelConfig(vocab_size=sp.vocab_size(), block_size=256, n_layer=8, n_head=8, n_embd=512, arch="vanilla", pad_id=sp.pad_id())
    teacher_ckpt = CKPT_DIR / "lm_vanilla_best.pt"
    if teacher_ckpt.exists():
        model_fp32 = load_lm(teacher_ckpt, teacher_cfg, device)
        val_loss, val_ppl = evaluate_lm(model_fp32, val_loader, device, max_iters=50, amp=False)
        bench = benchmark_generation(model_fp32, prompt, max_new_tokens=50, device=device)
        rows.append({"variant": "teacher_fp32", "device": str(device), "val_ppl": round(val_ppl, 3), "size_mb": round(state_dict_size_mb(model_fp32), 2), **{k: (round(v, 3) if v is not None else None) for k, v in bench.items() if k != "median_total_s"}})
        print(f"teacher_fp32: val_ppl={val_ppl:.3f}")

        # --- FP16, GPU ---
        model_fp16 = load_lm(teacher_ckpt, teacher_cfg, device).half()
        val_loss16, val_ppl16 = evaluate_lm(model_fp16, val_loader, device, max_iters=50, amp=False)
        bench16 = benchmark_generation(model_fp16, prompt, max_new_tokens=50, device=device)
        rows.append({"variant": "teacher_fp16", "device": str(device), "val_ppl": round(val_ppl16, 3), "size_mb": round(state_dict_size_mb(model_fp16), 2), **{k: (round(v, 3) if v is not None else None) for k, v in bench16.items() if k != "median_total_s"}})
        print(f"teacher_fp16: val_ppl={val_ppl16:.3f}")

        # --- Dynamic INT8, CPU ---
        cpu = torch.device("cpu")
        model_fp32_cpu = load_lm(teacher_ckpt, teacher_cfg, cpu)
        model_int8 = dynamic_quantize_int8(model_fp32_cpu)
        val_ds_cpu_loader = DataLoader(val_ds, batch_size=32, shuffle=True, num_workers=0, drop_last=True)
        val_loss8, val_ppl8 = evaluate_lm(model_int8, val_ds_cpu_loader, cpu, max_iters=20, amp=False)
        bench8 = benchmark_generation(model_int8, prompt, max_new_tokens=50, device=cpu, n_runs=10, warmup=2)
        rows.append({"variant": "teacher_int8_cpu", "device": "cpu", "val_ppl": round(val_ppl8, 3), "size_mb": round(state_dict_size_mb_quantized(model_int8), 2), **{k: (round(v, 3) if v is not None else None) for k, v in bench8.items() if k != "median_total_s"}})
        print(f"teacher_int8_cpu: val_ppl={val_ppl8:.3f}")
    else:
        print(f"WARNING: {teacher_ckpt} not found, skipping teacher variants")

    # --- Distilled / scratch students, GPU ---
    for run_name in ["student_distilled", "student_scratch"]:
        ckpt_path = CKPT_DIR / f"{run_name}_best.pt"
        if not ckpt_path.exists():
            print(f"WARNING: {ckpt_path} not found, skipping {run_name}")
            continue
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        student_cfg = ckpt["config"]
        model = load_lm(ckpt_path, student_cfg, device)
        val_loss, val_ppl = evaluate_lm(model, val_loader, device, max_iters=50, amp=False)
        bench = benchmark_generation(model, prompt, max_new_tokens=50, device=device)
        rows.append({"variant": run_name, "device": str(device), "val_ppl": round(val_ppl, 3), "size_mb": round(state_dict_size_mb(model), 2), **{k: (round(v, 3) if v is not None else None) for k, v in bench.items() if k != "median_total_s"}})
        print(f"{run_name}: val_ppl={val_ppl:.3f}")

    out_path = LOGS_DIR / "efficiency_pareto.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
