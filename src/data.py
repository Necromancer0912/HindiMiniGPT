"""Corpus ingestion, normalization, deterministic splits, and PyTorch Datasets
for both the language-modeling task and the sentiment classification task.
"""
import hashlib
import re
import unicodedata
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from tqdm import tqdm

from src.config import SPLITS_DIR

_ZERO_WIDTH_RE = re.compile(r"[​‌‍﻿]")
_WS_RE = re.compile(r"\s+")


def normalize_hindi_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", str(text))
    text = _ZERO_WIDTH_RE.sub("", text)
    text = _WS_RE.sub(" ", text).strip()
    return text


def read_corpus(corpus_dir: Path, dedup: bool = True) -> List[Dict[str, str]]:
    """Reads every *.txt file, normalizes, drops empties and exact duplicates."""
    rows = []
    seen_hashes = set()
    for file in tqdm(sorted(corpus_dir.glob("*.txt")), desc="Reading Hindi corpus"):
        try:
            raw = file.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        txt = normalize_hindi_text(raw)
        if not txt:
            continue
        if dedup:
            h = hashlib.md5(txt.encode("utf-8")).hexdigest()
            if h in seen_hashes:
                continue
            seen_hashes.add(h)
        rows.append({"id": file.stem, "text": txt})
    return rows


def split_ids(
    ids: List[str], val_ratio: float = 0.1, seed: int = 42, tag: str = "lm"
) -> Tuple[set, set]:
    """Deterministic split; persists id lists to outputs/splits/ and asserts no leakage."""
    train_ids, val_ids = train_test_split(ids, test_size=val_ratio, random_state=seed)
    train_ids, val_ids = set(train_ids), set(val_ids)
    assert len(train_ids & val_ids) == 0, "train/val leakage detected"

    (SPLITS_DIR / f"{tag}_train_ids.txt").write_text("\n".join(sorted(train_ids)), encoding="utf-8")
    (SPLITS_DIR / f"{tag}_val_ids.txt").write_text("\n".join(sorted(val_ids)), encoding="utf-8")
    return train_ids, val_ids


# ---------------------------------------------------------------------------
# Language modeling dataset
# ---------------------------------------------------------------------------
class LMDataset(Dataset):
    """Packs a list of texts into one contiguous token stream (EOS-separated) and
    serves fixed-length (x, y) blocks where y is x shifted forward by one position.
    """

    def __init__(self, texts: List[str], sp, block_size: int, eos_id: int, cache_path: Optional[Path] = None):
        self.block_size = block_size

        if cache_path is not None and cache_path.exists():
            self.tokens = torch.load(cache_path)
        else:
            token_ids: List[int] = []
            for t in tqdm(texts, desc="Tokenizing LM split"):
                token_ids.extend(sp.encode(t, out_type=int))
                token_ids.append(eos_id)
            self.tokens = torch.tensor(token_ids, dtype=torch.long)
            if cache_path is not None:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(self.tokens, cache_path)

    def __len__(self) -> int:
        return max(0, len(self.tokens) - self.block_size - 1)

    def __getitem__(self, idx: int):
        x = self.tokens[idx : idx + self.block_size]
        y = self.tokens[idx + 1 : idx + self.block_size + 1]
        return x, y


# ---------------------------------------------------------------------------
# Classification dataset
# ---------------------------------------------------------------------------
LABEL_MAP = {"negative": 0, "neutral": 1, "positive": 2}


def load_classification_df(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["text", "experience"]).copy()
    df["text"] = df["text"].astype(str).map(normalize_hindi_text)
    df = df[df["text"].str.len() > 0].copy()

    if df["experience"].dtype == object:
        df["label_id"] = df["experience"].astype(str).str.strip().str.lower().map(LABEL_MAP)
    else:
        df["label_id"] = df["experience"].astype(int)
    df = df.dropna(subset=["label_id"]).copy()
    df["label_id"] = df["label_id"].astype(int)
    return df.reset_index(drop=True)


def stratified_cls_split(df: pd.DataFrame, val_ratio: float = 0.2, seed: int = 42):
    train_df, val_df = train_test_split(
        df, test_size=val_ratio, random_state=seed, stratify=df["label_id"]
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)


class ClsDataset(Dataset):
    """Tokenizes reviews for classification.

    strategy="truncate": keep the last `max_len` tokens (causal-natural: the model's
      final hidden state then reflects the most recent context).
    strategy="chunks": split the full token sequence into <= max_len chunks; each
      __getitem__ returns the list of chunk tensors for aggregation downstream.
    """

    def __init__(self, df: pd.DataFrame, sp, max_len: int, eos_id: int, strategy: str = "truncate"):
        assert strategy in ("truncate", "chunks")
        self.strategy = strategy
        self.max_len = max_len
        self.labels = df["label_id"].tolist()
        self.encoded = [sp.encode(t, out_type=int) + [eos_id] for t in df["text"].tolist()]

    def __len__(self) -> int:
        return len(self.encoded)

    def __getitem__(self, idx: int):
        ids = self.encoded[idx]
        label = self.labels[idx]
        if self.strategy == "truncate":
            ids = ids[-self.max_len :]
            return torch.tensor(ids, dtype=torch.long), label
        # chunks: list of tensors, each <= max_len
        chunks = [ids[i : i + self.max_len] for i in range(0, len(ids), self.max_len)]
        chunks = [torch.tensor(c, dtype=torch.long) for c in chunks]
        return chunks, label


def collate_truncate(batch, pad_id: int):
    seqs, labels = zip(*batch)
    max_len = max(len(s) for s in seqs)
    padded = torch.full((len(seqs), max_len), pad_id, dtype=torch.long)
    attn_mask = torch.zeros((len(seqs), max_len), dtype=torch.bool)  # True = real token
    for i, s in enumerate(seqs):
        padded[i, : len(s)] = s
        attn_mask[i, : len(s)] = True
    return padded, attn_mask, torch.tensor(labels, dtype=torch.long)


def collate_chunks(batch, pad_id: int):
    """Flattens all chunks across the batch into one padded tensor plus an index
    mapping each chunk back to its source example, so the backbone runs once.
    """
    chunk_lists, labels = zip(*batch)
    flat_chunks = []
    owner = []
    for ex_idx, chunks in enumerate(chunk_lists):
        for c in chunks:
            flat_chunks.append(c)
            owner.append(ex_idx)
    max_len = max(len(c) for c in flat_chunks)
    padded = torch.full((len(flat_chunks), max_len), pad_id, dtype=torch.long)
    attn_mask = torch.zeros((len(flat_chunks), max_len), dtype=torch.bool)
    for i, c in enumerate(flat_chunks):
        padded[i, : len(c)] = c
        attn_mask[i, : len(c)] = True
    return padded, attn_mask, torch.tensor(owner, dtype=torch.long), torch.tensor(labels, dtype=torch.long)
