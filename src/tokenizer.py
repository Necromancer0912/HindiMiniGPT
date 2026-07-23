"""SentencePiece BPE tokenizer: train on the LM train split, load, encode/decode."""
from pathlib import Path
from typing import List

import sentencepiece as spm

from src.config import SPM_MODEL_PATH, SPM_MODEL_PREFIX

UNK_ID, BOS_ID, EOS_ID, PAD_ID = 0, 1, 2, 3


def train_spm(
    train_texts: List[str],
    vocab_size: int = 5000,
    model_type: str = "bpe",
    character_coverage: float = 0.9995,
    max_sentence_length: int = 24000,
) -> Path:
    """Trains SentencePiece on train-split text only. No-op if a model already exists.

    max_sentence_length is raised well above SentencePiece's 4192-char default: our
    corpus has document-length "sentences" (up to ~77k chars), and the default
    silently drops ~17% of the longest documents from tokenizer training.
    """
    if SPM_MODEL_PATH.exists():
        return SPM_MODEL_PATH

    corpus_file = SPM_MODEL_PREFIX.parent / "sp_train_text.txt"
    corpus_file.write_text("\n".join(train_texts), encoding="utf-8")

    spm.SentencePieceTrainer.train(
        input=str(corpus_file),
        model_prefix=str(SPM_MODEL_PREFIX),
        vocab_size=vocab_size,
        model_type=model_type,
        character_coverage=character_coverage,
        split_digits=False,
        unk_id=UNK_ID,
        bos_id=BOS_ID,
        eos_id=EOS_ID,
        pad_id=PAD_ID,
        input_sentence_size=2_000_000,
        shuffle_input_sentence=True,
        max_sentence_length=max_sentence_length,
    )
    return SPM_MODEL_PATH


def load_tokenizer(model_path: Path = SPM_MODEL_PATH) -> spm.SentencePieceProcessor:
    if not model_path.exists():
        raise FileNotFoundError(
            f"Tokenizer not found at {model_path}. Run train_spm() first (see notebook 01)."
        )
    return spm.SentencePieceProcessor(model_file=str(model_path))


def encode(sp: spm.SentencePieceProcessor, text: str) -> List[int]:
    return sp.encode(text, out_type=int)


def decode(sp: spm.SentencePieceProcessor, ids: List[int]) -> str:
    return sp.decode(ids)
