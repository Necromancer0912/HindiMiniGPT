"""Edge-efficiency toolkit: knowledge distillation, quantization, and latency /
size / memory benchmarking -- the Pareto-frontier ingredients for notebook 04.

Windows note: bitsandbytes 8-bit is unreliable on Windows, so the INT8 point uses
torch.ao.quantization.quantize_dynamic on CPU instead. FP16 is measured on GPU via
.half().
"""
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


def distill_loss(student_logits: torch.Tensor, teacher_logits: torch.Tensor, targets: torch.Tensor, T: float = 2.0, alpha: float = 0.5) -> torch.Tensor:
    """alpha * hard-label CE + (1-alpha) * T^2 * KL(student/T || teacher/T)."""
    ce = F.cross_entropy(student_logits.view(-1, student_logits.size(-1)), targets.view(-1))
    student_log_probs = F.log_softmax(student_logits / T, dim=-1)
    teacher_probs = F.softmax(teacher_logits / T, dim=-1)
    kd = F.kl_div(student_log_probs.view(-1, student_log_probs.size(-1)), teacher_probs.view(-1, teacher_probs.size(-1)), reduction="batchmean")
    return alpha * ce + (1 - alpha) * (T * T) * kd


def dynamic_quantize_int8(model: nn.Module) -> nn.Module:
    """CPU-only dynamic quantization of all Linear layers to INT8."""
    model_cpu = model.to("cpu").eval()
    return torch.ao.quantization.quantize_dynamic(model_cpu, {nn.Linear}, dtype=torch.qint8)


def state_dict_size_mb(model: nn.Module) -> float:
    total_bytes = sum(p.numel() * p.element_size() for p in model.state_dict().values() if torch.is_tensor(p))
    return total_bytes / (1024 ** 2)


def state_dict_size_mb_quantized(model: nn.Module) -> float:
    """Quantized modules store packed int8 params that .numel()/.element_size() on
    the wrapper params can misreport; save-to-disk gives the ground truth.
    """
    tmp_path = Path("outputs") / "_tmp_quant_size.pt"
    torch.save(model.state_dict(), tmp_path)
    size_mb = tmp_path.stat().st_size / (1024 ** 2)
    tmp_path.unlink(missing_ok=True)
    return size_mb


@torch.no_grad()
def benchmark_generation(model: nn.Module, prompt_ids, max_new_tokens: int, device, n_runs: int = 20, warmup: int = 5, block_size: int = 256, eos_id: int = 2):
    """Median latency-per-token and throughput for autoregressive generation."""
    from src.generate import generate  # local import avoids a cycle at module load

    model.eval()
    for _ in range(warmup):
        generate(model, prompt_ids, max_new_tokens, block_size, device, eos_id=eos_id)

    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        generate(model, prompt_ids, max_new_tokens, block_size, device, eos_id=eos_id)
        if device.type == "cuda":
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)

    times.sort()
    median_time = times[len(times) // 2]
    latency_per_token_ms = (median_time / max_new_tokens) * 1000
    throughput_tok_s = max_new_tokens / median_time
    peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 ** 2) if device.type == "cuda" else None

    return {
        "median_total_s": median_time,
        "latency_per_token_ms": latency_per_token_ms,
        "throughput_tok_s": throughput_tok_s,
        "peak_vram_mb": peak_vram_mb,
    }
