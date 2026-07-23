# CSE556 Assignment 2: Advanced Step-by-Step Workplan (No Implementation)

## 0. Reality Check From Your Dataset Scan
- Hindi corpus files found: `44,000` (`Data/hindi_corpus/train/*.txt`)
- Approx text length from a 1,000-file sample:
- Sampled files: `1,000`
- Total chars in sample: `1,865,288`
- Avg chars/file in sample: `1,865.29`
- Classification CSV: `Data/text_classification_dataset/train.csv`
- Classification rows: `718`
- Columns detected: `text, experience`
- Detected label column: `experience`
- Label distribution:
- `2`: `273`
- `0`: `240`
- `1`: `205`

Implication:
- LM dataset is large enough for meaningful pretraining at mini scale.
- Classification dataset is small; transfer learning and regularization are important.

## 1. What You Must Deliver (Ground Truth)
- Preprocessing pipeline for both tasks.
- Subword tokenizer (`vocab_size=5000`) trained on LM train split.
- Decoder-only Transformer language model (GPT-style) trained autoregressively.
- Validation perplexity report.
- 50-token Hindi generation output file.
- Classifier head on GPT backbone.
- Validation classification accuracy report.
- 5 correctly predicted movie-review samples file.
- Model checkpoints for LM and classifier.

## 2. Architecture Choice: MiniGPT vs LLaMA vs Gemini-Style
Use this decision framework:

1. `MiniGPT (assignment baseline)`
- Pros: easiest to implement and explain in demo.
- Cons: weaker training stability and efficiency if too naive.

2. `LLaMA-style decoder-only (recommended advanced path)`
- Pros: best tradeoff for small-to-mid compute; modern and strong.
- Core upgrades:
- Pre-norm residual blocks.
- `RMSNorm` instead of LayerNorm.
- `RoPE` positional encoding.
- `SwiGLU` MLP instead of plain ReLU MLP.
- Flash/efficient attention if framework supports it.
- Cons: slightly more engineering complexity.

3. `Gemini-style` (not recommended for this assignment)
- Reason: detailed architecture is not fully public and generally depends on large-scale MoE/multimodal infrastructure.
- For this assignment, using Gemini as a direct blueprint is impractical and risky.

Recommendation:
- Build a grading-compliant GPT baseline first.
- Then add LLaMA-style improvements only if they do not violate assignment constraints.

## 3. End-to-End Plan (Step by Step)

### Phase A: Project Protocol and Reproducibility
1. Freeze environment plan.
- Python, PyTorch, tokenizer library versions.
- Single `requirements` lock for all team members.
2. Set deterministic behavior.
- Global random seed for Python, NumPy, PyTorch.
- Fixed train/val splits saved to disk.
3. Define artifact directories.
- `outputs/checkpoints`, `outputs/generations`, `outputs/predictions`, `outputs/logs`.
4. Define experiment naming scheme.
- Example: `lm_bs32_ctx256_lr3e-4_run01`.

### Phase B: Data Processing (Task 1)
1. Hindi corpus ingestion.
- Read all text files.
- Remove empty/corrupt files.
- Normalize Unicode consistently.
2. Split corpus into train/val.
- Deterministic split (for reproducibility and fair perplexity).
- Save split file lists (`train_ids`, `val_ids`).
3. Train tokenizer on train split only.
- BPE/SentencePiece, `vocab=5000`.
- Save tokenizer model and vocab artifacts.
4. Create LM training pairs.
- Convert token stream into fixed-length blocks.
- `y` is `x` shifted by one token.
5. Classification preprocessing.
- Use `text` as input and `experience` as label.
- Map labels to integers `0,1,2` (already numeric in your data).
- Stratified train/val split due small dataset.
6. Quality gates for preprocessing.
- Check token coverage and OOV behavior.
- Verify label balance after split.
- Verify no leakage between train/val.

### Phase C: Language Model Build (Task 2)
1. Implement decoder-only stack.
- Token embedding + positional representation.
- Causal masked multi-head self-attention.
- MLP block.
- Residual + normalization.
- Stacked blocks.
- LM head projection to vocab.
2. Start with a stable mini configuration.
- Context length: `128-256`.
- Layers: `4-8`.
- Hidden size: `256-512`.
- Heads: `4-8`.
3. Training objective.
- Cross-entropy over next-token prediction.
4. Optimizer and schedule best practice.
- `AdamW`, warmup, cosine decay.
- Gradient clipping.
- Mixed precision (if GPU supports).
5. Monitoring.
- Track train loss, val loss, val perplexity.
- Save best checkpoint by lowest val perplexity.
6. Generation evaluation.
- Fixed prompts and temperature settings.
- Save at least one 50-token output to file.

### Phase D: Classifier on GPT Backbone (Task 3)
1. Attach classification head.
- Last-token hidden state -> linear layer -> 3 classes.
2. Transfer strategy.
- Initialize from LM-trained backbone weights.
- Option A: freeze lower blocks initially.
- Option B: full fine-tuning with lower LR.
3. Handle small-data overfitting.
- Early stopping on val accuracy.
- Weight decay and dropout tuning.
4. Metrics.
- Primary: validation accuracy.
- Secondary (recommended): macro-F1 and confusion matrix.
5. Save artifacts.
- Best classifier checkpoint.
- 5 correctly predicted validation reviews with true/pred labels.

### Phase E: Final Packaging and Demo Readiness
1. Collect all required files.
- Code notebook(s), checkpoints, output text files.
2. Create concise experiment report.
- Data split details.
- Final hyperparameters.
- Val perplexity and val accuracy.
- Error analysis snapshot.
3. Demo preparation.
- Every member explains full pipeline end-to-end.
- Keep 2 fallback checkpoints in case one run is unstable.

## 4. Resource Plan (Practical)

### Compute
- Minimum workable: single modern GPU with 8-12 GB VRAM.
- Comfortable: 16-24 GB VRAM.
- CPU-only is possible but very slow for LM training.

### Time Budget (Realistic)
1. Data pipeline + tokenizer: `0.5-1 day`
2. LM baseline training/tuning: `1-2 days`
3. Classifier fine-tuning/tuning: `0.5-1 day`
4. Packaging/report/demo prep: `0.5 day`

### Storage
- Keep `10-30 GB` free to safely store checkpoints and logs.

## 5. Best-Performance Training Strategy (Recommended Sequence)
1. Train a clean baseline first (assignment-compliant).
2. Validate perplexity and generation quickly.
3. Transfer to classifier and establish baseline accuracy.
4. Run controlled ablations, one change at a time:
- Context length.
- Model depth/width.
- LR schedule.
- Freeze vs full fine-tune for classifier.
5. Keep a strict experiment table with run ID, config, and results.

## 6. Risk Control Checklist
- Data leakage check: train/val overlap must be zero.
- Tokenizer leakage check: tokenizer trained only on train split.
- Label mapping check: `0,1,2` consistent everywhere.
- Evaluation consistency: same val split for all runs.
- Checkpoint naming and provenance tracked.

## 7. Suggested Final Folder Outputs
- `outputs/checkpoints/lm_best.pt`
- `outputs/checkpoints/classifier_best.pt`
- `outputs/generations/generated_hindi.txt`
- `outputs/predictions/5_correct_samples.txt`
- `outputs/logs/metrics.csv`
- `outputs/logs/run_summary.md`

## 8. Team Execution Mode (Equal Contribution, 3 Members)
- Do not split by isolated tasks.
- For every phase, rotate roles:
- Driver (coding)
- Navigator (review/design)
- Validator (experiments/logging)
- Rotate each session so everyone contributes to all parts.

This gives equal participation and keeps all members ready for the demo.
