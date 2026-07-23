"""Central configuration: model/train hyperparameters, paths, seeding, device setup."""
import os
import random
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

SEED = 42


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
        return torch.device("cuda")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Paths (project root = parent of src/)
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "NLP_Dataset_2026"
CORPUS_DIR = DATA_DIR / "hindi_corpus" / "train"
CLS_CSV = DATA_DIR / "text_classification_dataset" / "train.csv"

OUT_DIR = ROOT / "outputs"
CKPT_DIR = OUT_DIR / "checkpoints"
GEN_DIR = OUT_DIR / "generations"
PRED_DIR = OUT_DIR / "predictions"
TOKENIZER_DIR = OUT_DIR / "tokenizer"
SPLITS_DIR = OUT_DIR / "splits"
LOGS_DIR = OUT_DIR / "logs"
FIG_DIR = OUT_DIR / "figures"

for d in (CKPT_DIR, GEN_DIR, PRED_DIR, TOKENIZER_DIR, SPLITS_DIR, LOGS_DIR, FIG_DIR):
    d.mkdir(parents=True, exist_ok=True)

SPM_MODEL_PREFIX = TOKENIZER_DIR / "hindi_spm_bpe"
SPM_MODEL_PATH = SPM_MODEL_PREFIX.with_suffix(".model")


# ---------------------------------------------------------------------------
# Model / training configs
# ---------------------------------------------------------------------------
@dataclass
class ModelConfig:
    vocab_size: int = 5000
    block_size: int = 256
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    dropout: float = 0.1
    arch: str = "vanilla"  # "vanilla" | "modern"
    tie_weights: bool = True
    pad_id: int = 3

    @property
    def head_dim(self) -> int:
        assert self.n_embd % self.n_head == 0
        return self.n_embd // self.n_head


@dataclass
class TrainConfig:
    batch_size: int = 32
    grad_accum: int = 2
    max_steps: int = 8000
    lr: float = 3e-4
    min_lr_ratio: float = 0.1
    warmup_steps: int = 300
    weight_decay: float = 0.1
    betas: tuple = (0.9, 0.95)
    grad_clip: float = 1.0
    label_smoothing: float = 0.1
    amp: bool = True
    eval_interval: int = 250
    eval_iters: int = 100
    log_path: str = "lm_metrics.csv"
    ckpt_name: str = "lm_best.pt"
    run_name: str = "run"
    resume: bool = True


@dataclass
class ClsTrainConfig:
    batch_size: int = 16
    epochs: int = 25
    lr: float = 5e-5
    backbone_lr_mult: float = 0.2
    weight_decay: float = 0.02
    grad_clip: float = 1.0
    amp: bool = True
    early_stop_patience: int = 5
    val_ratio: float = 0.2
    pooling: str = "last"  # "last" | "mean" | "attn"
    ft_mode: str = "full"  # "frozen" | "full" | "lora"
    lora_r: int = 8
    lora_alpha: int = 16
    max_chunk_len: int = 256
    chunk_aggregate: bool = False
    log_path: str = "cls_metrics.csv"
    run_name: str = "cls_run"
