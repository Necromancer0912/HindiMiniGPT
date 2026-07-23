# HindiMiniGPT

A decoder-only Transformer language model built from scratch for Hindi, trained on a
single consumer GPU, extended into a sentiment classifier, and evaluated with the
same rigor applied to model architecture, domain adaptation, cross-validation,
distillation, and quantization.

The project started as a university NLP assignment (build a GPT-style model, train
it as a language model, and use the same backbone for text classification) and was
carried well beyond that scope: an architecture ablation, a full classical-baseline
and cross-validation study, task-adaptive pretraining, a neural-lexical ensemble,
knowledge distillation, and quantization benchmarking, all constrained to run on an
8 GB GPU.

A deliberate design principle throughout this project: report what the evidence
shows, including results that complicate the initial story. The classification
section below documents a case where a simple classical model outperforms the
from-scratch Transformer once evaluated properly, and explains why, rather than
only reporting the numbers that flatter the deep model.

## Overview

Two datasets are used. A large unlabeled Hindi text corpus (news and encyclopedic
articles) is used to pretrain a decoder-only language model from scratch. A smaller
labeled dataset of Hindi movie reviews (three-way sentiment: negative, neutral,
positive) is used to fine-tune a classifier on top of that same pretrained model.

Both the language model and the classifier are implemented without relying on
`nn.TransformerEncoder`, `nn.MultiheadAttention`, or any pretrained checkpoint from
an external source. Multi-head self-attention, the causal mask, the feed-forward
block, layer normalization, positional encoding, and the classification head are
all written directly against PyTorch tensor operations, and their correctness is
verified against PyTorch's own fused attention implementation before any training
begins.

## Dataset

The language-modeling corpus consists of 38,308 raw Hindi text files. After
Unicode normalization and exact-duplicate removal, 33,520 documents remain,
totaling roughly 71 million characters. A deterministic 90/10 split produces the
training and validation sets, saved as explicit document-id lists so the split is
reproducible and leakage-free.

A SentencePiece byte-pair-encoding tokenizer with a vocabulary of 5,000 subwords is
trained on the training split only. One correctness issue was found and fixed during
tokenizer training: SentencePiece's default maximum sentence length (4,192
characters) was silently discarding roughly seventeen percent of the longest
training documents, since each document here is treated as a single "sentence" and
some run past 77,000 characters. Raising the limit to 24,000 characters resolved
this before the tokenizer was finalized.

The classification dataset consists of 718 Hindi movie reviews sourced from
professional film critics (the source corpus corresponds to the IITP / Navbharat
Times Hindi movie-review collection used in prior Hindi sentiment-analysis
research), labeled with three sentiment classes. One row was dropped for being
empty, leaving 717 usable reviews, split 573/144 with stratification to preserve
class balance. Reviews average roughly 1,850 subword tokens, and effectively all of
them exceed the model's 256-token context window, which is the reason long-document
handling strategies are a first-class part of the classification study rather than
an afterthought.

## Model architecture

The backbone is a GPT-style decoder-only Transformer, implemented in two variants
that share the same block structure so the comparison between them isolates a
single set of design choices:

- A conventional variant using learned absolute positional embeddings, standard
  layer normalization, and a two-layer feed-forward block with a ReLU activation.
- A modern variant using rotary positional embeddings, RMSNorm in place of layer
  normalization, and a gated SwiGLU feed-forward block, with the output projection
  weight tied to the input embedding matrix.

Both variants use explicit query/key/value projections, scaled dot-product
attention, and an explicit lower-triangular causal mask applied before the softmax,
so that no token can attend to a position ahead of it in the sequence. This was
verified two ways before any training: the output of the hand-written attention
module was compared against PyTorch's fused `scaled_dot_product_attention` on
identical inputs (maximum difference under 1e-4), and the causal property itself
was checked directly by perturbing tokens after a given position and confirming the
output at that position does not change.

Each residual block applies pre-normalization around both the attention and
feed-forward sub-layers. The full model stacks eight of these blocks at a hidden
size of 512 with eight attention heads, giving approximately 27.9 million
parameters, trained with a context length of 256 tokens.

The same backbone is reused, unmodified, for the classification task: a linear
head is attached on top, and three different pooling strategies for producing a
single vector from the sequence of hidden states are implemented and compared —
using the final token's hidden state (the natural choice for a causal model, since
only the last position has attended to the entire sequence), mean-pooling across
all real tokens, and a learned attention-based pooling mechanism.

## Language model training and results

The model is trained with cross-entropy loss using AdamW, a linear warmup followed
by cosine learning-rate decay, gradient clipping, label smoothing, and mixed-precision
training. Both architecture variants were trained for 15,000 steps under an
identical parameter and compute budget, differing only in the architectural choices
described above.

| Variant | Validation perplexity |
|---|---|
| Conventional (learned position embeddings, LayerNorm, ReLU) | 28.83 |
| Modern (rotary embeddings, RMSNorm, SwiGLU) | 27.59 |

The modern variant reduces perplexity by roughly 4.3 percent at identical parameter
count and training budget, consistent with the broader shift toward this style of
architecture in recent open language models.

An autoregressive text-generation function with temperature, top-k, and
nucleus (top-p) sampling was implemented and used to generate sample continuations
from both models. The output is coherent and topically consistent with the
prompt — for example, a prompt referencing Mahatma Gandhi continues correctly into
content about the civil disobedience movement, rather than producing unrelated text.

## Classification: methodology and honest results

The classification study is the part of this project that received the most
scrutiny, because an initial single-split evaluation gave a misleading picture of
model quality. What follows is the full sequence of investigation, not just the
final numbers.

### Establishing a baseline before judging the model

Before drawing any conclusion about the Transformer classifier, a majority-class
baseline and a TF-IDF plus linear-SVM baseline were computed on the same data. This
matters because the underlying dataset turns out to be a recognized, difficult
benchmark: published results using contextual string embeddings on this same family
of Hindi movie-review data top out around 62 percent accuracy, considerably lower
than sentiment benchmarks in better-resourced languages typically show, largely
because these are long professional reviews where the sentiment verdict is diluted
across mostly neutral plot description.

| Model | Evaluation | Accuracy | Macro F1 |
|---|---|---|---|
| Majority class | single split | 38.2% | 0.184 |
| TF-IDF + linear SVM | single split (144 reviews) | 50.7% | 0.492 |
| TF-IDF + linear SVM | 5-fold cross-validation (718 reviews) | 58.9% (±4.2) | 0.572 (±0.042) |

The gap between the single-split and cross-validated TF-IDF numbers (roughly eight
points of accuracy) is itself informative: a 144-example validation set is
genuinely high-variance on this data, which is exactly why the Transformer
classifier is re-evaluated with cross-validation rather than trusted on one split.

### The initial experiment grid

A grid over pooling strategy, fine-tuning mode (frozen backbone, full fine-tuning,
and a manually implemented low-rank adaptation), and long-document handling
strategy (truncating to the final 256 tokens versus splitting each review into
chunks and aggregating chunk-level features) was run on a single fixed split:

| Configuration | Accuracy | Macro F1 | Trainable parameters |
|---|---|---|---|
| Attention pooling, full fine-tune | 54.2% | 0.535 | 28.2M (100%) |
| Last-token pooling, chunk aggregation | 52.8% | 0.524 | 27.9M (100%) |
| Mean pooling, full fine-tune | 50.7% | 0.495 | 27.9M (100%) |
| Last-token pooling, low-rank adaptation | 47.2% | 0.469 | 198K (0.7%) |
| Last-token pooling, full fine-tune | 46.5% | 0.467 | 27.9M (100%) |
| Last-token pooling, frozen backbone | 34.0% | 0.339 | 1.5K (0.005%) |

Pooling strategy turned out to matter more than fine-tuning depth: attention
pooling outperforms last-token pooling by roughly seven points of accuracy. The
low-rank adaptation result is notable in its own right, recovering nearly all of
full fine-tuning's accuracy while updating well under one percent of the model's
parameters. The frozen-backbone result, at only four points above the majority
baseline, was the first sign of a deeper issue investigated next.

### Diagnosing and fixing the domain gap

The frozen-backbone result is a textbook symptom of a domain mismatch: the language
model was pretrained on encyclopedic and news-style Hindi text, not on movie-review
prose. Following the task-adaptive pretraining approach described by Gururangan et
al. (2020), the trained language model was further pretrained, without labels, on
the classification task's own review text before the classifier was attached.

| | Review-domain validation perplexity |
|---|---|
| Before adaptation | 49.06 |
| After adaptation (1,000 additional training steps) | 19.74 |

For context, the language model's perplexity on its original training domain was
27.6; a perplexity of 49 on review text confirms a real and substantial domain
gap, which the adaptation step reduces by nearly sixty percent. Repeating the
frozen-backbone classification experiment on the domain-adapted model:

| Configuration | Accuracy | Macro F1 |
|---|---|---|
| Frozen backbone, original language model | 38.2% | 0.355 |
| Frozen backbone, domain-adapted language model | 50.0% | 0.493 |
| Full fine-tune, domain-adapted, with class weighting and gradual unfreezing | 52.1% | 0.515 |

Domain adaptation improves the frozen-backbone result by close to forty percent in
relative terms — a clean, reproducible effect. Whether that benefit survives full
fine-tuning is a separate question, addressed next.

### The cross-validated comparison

Both the initial experiment grid and the domain-adaptation result above are
single-split numbers, and the baseline section already demonstrated that single
splits are unreliable on this dataset. The strongest configuration (attention
pooling, full fine-tuning) was therefore re-evaluated with proper five-fold
cross-validation, comparing the original and domain-adapted language models:

| Model | Accuracy (mean ± std) | Macro F1 (mean ± std) |
|---|---|---|
| Transformer classifier, original backbone | 50.6% (±3.0) | 0.502 (±0.024) |
| Transformer classifier, domain-adapted backbone | 51.0% (±4.0) | 0.501 (±0.038) |
| TF-IDF + linear SVM | 58.9% (±4.2) | 0.572 (±0.042) |

This comparison changes the conclusion in two ways that the single-split numbers
did not show. First, under cross-validation the classical TF-IDF model outperforms
the from-scratch Transformer by a wide margin, reversing the single-split ranking
in which the Transformer appeared stronger. With only 573 labeled training examples,
a 28-million-parameter Transformer does not have enough supervision to out-represent
a lexical model built on the same data. Second, the domain-adaptation benefit that
was large and clear for the frozen backbone (roughly forty percent relative
improvement) disappears once the backbone is fully fine-tuned — the two mean scores
are within one standard deviation of each other. This is a sensible result rather
than a contradiction: full fine-tuning already performs its own supervised
adaptation to the target domain through the labels, so the additional unsupervised
adaptation step has little marginal signal left to contribute once the whole model
is unfrozen. Neither cross-validated result approaches the roughly 62 percent
reported for contextual-embedding models on this benchmark, which reflects a
considerably larger pretraining investment than a single 8 GB GPU and a few hours
of training time allow.

### A neural and lexical ensemble

Since the Transformer and the TF-IDF model make different kinds of errors,
combining their predicted probabilities was tried as a final step. The blending
weight was selected using only out-of-fold probabilities from the 573 training
reviews (five-fold cross-validation), so the 144-review validation set is touched
exactly once, after the weight is already fixed — no tuning against the evaluation
set.

| Model | Accuracy | Macro F1 |
|---|---|---|
| Transformer only | 52.1% | 0.515 |
| TF-IDF only | 50.7% | 0.492 |
| Ensemble (weighted 15% Transformer, 85% TF-IDF) | 53.5% | 0.525 |

The ensemble outperforms either individual model on the held-out split, and the
heavily TF-IDF-weighted blend selected by the out-of-fold procedure is itself
consistent with the cross-validation finding above: the classical model is doing
most of the work here, with the Transformer contributing a smaller complementary
signal.

### Error analysis

The confusion matrix and per-class precision and recall for the best configuration
show that the neutral class is consistently the hardest to separate (precision
0.44, compared to 0.59-0.60 for the negative and positive classes), which is
expected given that neutral sentiment sits at the boundary between the other two.
Five correctly classified and five misclassified validation reviews are saved
alongside the model outputs for inspection.

## Efficiency: distillation and quantization

Beyond accuracy, the project characterizes the computational cost of deploying the
language model, which matters directly for resource-constrained or edge
environments.

A student model with roughly 3.7 million parameters (three layers, 256 hidden
dimensions — about 7.5 times smaller than the full teacher) was trained two ways
for an identical number of steps: once using a combined distillation objective
(cross-entropy on the true tokens plus a temperature-scaled KL divergence against
the teacher's output distribution), and once from scratch using cross-entropy
alone, as a control.

| Student | Validation perplexity |
|---|---|
| Distilled from the trained teacher | 79.77 |
| Trained from scratch, identical size | 84.02 |

Distillation provides a modest but real improvement, roughly five percent lower
perplexity at identical size and training budget.

Four deployment variants of the full model were then benchmarked for validation
perplexity, checkpoint size, generation latency per token, and throughput:

| Variant | Device | Perplexity | Size | Latency/token | Throughput |
|---|---|---|---|---|---|
| Full precision (FP32) | GPU | 29.30 | 116.2 MB | 4.44 ms | 225 tokens/s |
| Half precision (FP16) | GPU | 29.30 | 58.1 MB | 3.77 ms | 265 tokens/s |
| Dynamic INT8 quantization | CPU | 28.33 | 36.9 MB | 6.23 ms | 161 tokens/s |
| Distilled student | GPU | 82.80 | 19.0 MB | 1.86 ms | 538 tokens/s |

Half-precision inference is effectively free here: identical quality, half the
size, and faster execution due to tensor-core support. Dynamic INT8 quantization
(chosen over 8-bit libraries that are unreliable on Windows) reduces the model to
roughly a third of its original size for CPU-only deployment, at the cost of GPU
throughput. One measurement caveat is disclosed directly: the INT8 perplexity was
computed over fewer evaluation batches than the GPU variants because CPU inference
is slower, so its slightly better perplexity figure should be read as evaluation
noise rather than a genuine quantization benefit; the size, latency, and throughput
comparisons are unaffected by this and remain the meaningful takeaways.

## Comparison with published research

The classification dataset corresponds to a known, difficult Hindi sentiment
benchmark. The strongest published result identified for this task family uses
contextual string embeddings (Sharma and Solanki, 2021) and reaches approximately
62 percent accuracy — itself far below what sentiment benchmarks in
better-resourced languages typically achieve, confirming that the difficulty here
is inherent to the data rather than specific to any one modeling approach. This
project's cross-validated results (roughly 51 percent for the from-scratch
Transformer, 59 percent for TF-IDF) sit below that ceiling by a margin consistent
with the difference in pretraining investment: the reference result is built on
embeddings pretrained over a substantially larger corpus than the 71 million
characters and few hours of single-GPU training used here.

The task-adaptive pretraining component follows the methodology introduced by
Gururangan et al. (2020, "Don't Stop Pretraining: Adapt Language Models to Domains
and Tasks"), which demonstrated that continuing to pretrain a language model on
in-domain, unlabeled text before fine-tuning improves downstream performance,
particularly under limited supervision. That paper's experiments use models on the
order of 125 million parameters pretrained on web-scale corpora. This project
reproduces the same qualitative effect — a substantial improvement for a frozen or
lightly adapted backbone, reflecting a real, reproducible domain-gap fix — at
roughly one one-thousandth of the parameter count and training data, using a
from-scratch 28-million-parameter model and well under a million tokens of
adaptation text, in a matter of minutes on a single consumer GPU.

The distillation results are directionally consistent with the broader body of
work on knowledge distillation for language models (for example, DistilBERT-style
results showing large parameter reductions for comparatively small quality loss),
scaled down considerably here to a 3.7-million-parameter student.

The overall comparison is intentionally framed around what an 8 GB single-GPU,
few-hour compute budget can and cannot achieve, rather than presenting these
numbers as competitive with substantially larger training investments.

## Technology stack

The model, training loops, evaluation harnesses, and all supporting utilities are
implemented in Python using PyTorch (version 2.13, CUDA 12.6 build) as the deep
learning framework, running on a single NVIDIA RTX 4060 Ti with 8 GB of memory.
Training uses automatic mixed precision and TF32 matrix multiplication for
throughput.

SentencePiece provides byte-pair-encoding tokenization. scikit-learn is used for
the classical baselines and cross-validation machinery: TF-IDF vectorization,
logistic regression, calibrated linear support vector machines, stratified k-fold
splitting, and the standard classification metrics. Quantization uses PyTorch's
built-in dynamic quantization for CPU inference, chosen deliberately over
third-party 8-bit libraries that are unreliable on Windows. Low-rank adaptation is
implemented directly rather than through an external library, for the same reason.

Data handling and analysis use pandas and NumPy; all figures are produced with
Matplotlib through a shared styling module so that plots across the project are
visually consistent. Jupyter notebooks, executed end-to-end and non-interactively
via nbconvert, are used for analysis and presentation, while all training and
evaluation that takes more than a couple of minutes runs as standalone,
background-capable Python scripts, so that long-running jobs are independent of any
notebook kernel.

## Repository structure

```
src/                     shared package imported by every script and notebook
  config.py                 paths, seeding, model and training configuration
  tokenizer.py              SentencePiece training, loading, encode/decode
  data.py                   corpus ingestion, splitting, dataset classes
  model.py                  attention, feed-forward blocks, the language model
                             and classifier, both architecture variants
  train.py                  training and evaluation loops, checkpointing, logging
  generate.py               autoregressive text generation
  lora.py                   low-rank adaptation, implemented directly
  efficiency.py             distillation loss, quantization, benchmarking
  baselines.py              majority-class and TF-IDF baselines, single-split and
                             cross-validated
  cls_utils.py              shared, architecture-aware classifier construction
  cross_val.py              stratified cross-validation harness, with optional
                             per-fold domain adaptation
  plots.py                  shared plotting style
  utils.py                  attention correctness checks

scripts/                  standalone, background-runnable entry points
  train_lm.py               language model training
  train_cls.py              classifier training and fine-tuning
  distill.py                knowledge distillation
  benchmark.py              quantization and latency benchmarking
  adapt_lm.py               task-adaptive pretraining
  eval_classification.py    baselines, cross-validation, and ensemble orchestration

notebooks/                analysis and presentation, loading results the scripts produced
  01_data_and_tokenizer.ipynb
  02_language_model.ipynb
  03_classification.ipynb
  04_efficiency_edge.ipynb

outputs/                  checkpoints, figures, logs, and generated artifacts
archive/                  earlier drafts, superseded once the current pipeline was built
```

## Reproducing the results

Install a CUDA-matched PyTorch build and the remaining dependencies:

```
pip install torch --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
```

Run `notebooks/01_data_and_tokenizer.ipynb` first; it builds the tokenizer and the
cached token tensors everything else depends on. Then, in order:

```
python scripts/train_lm.py   --arch vanilla --run_name lm_vanilla --n_layer 8 --n_head 8 --n_embd 512 --batch_size 64 --max_steps 15000
python scripts/train_lm.py   --arch modern  --run_name lm_modern  --n_layer 8 --n_head 8 --n_embd 512 --batch_size 64 --max_steps 15000
python scripts/train_cls.py  --pooling last --ft_mode full --strategy truncate --run_name cls_last_full --n_layer 8 --n_head 8 --n_embd 512
python scripts/distill.py    --mode distill --run_name student_distilled --teacher_n_layer 8 --teacher_n_head 8 --teacher_n_embd 512
python scripts/benchmark.py

python scripts/adapt_lm.py --base_ckpt outputs/checkpoints/lm_modern_best.pt --run_name lm_adapted
python scripts/train_cls.py --arch modern --backbone_ckpt outputs/checkpoints/lm_modern_best.pt  --ft_mode frozen --pooling attn --run_name cls_frozen_wiki
python scripts/train_cls.py --arch modern --backbone_ckpt outputs/checkpoints/lm_adapted_best.pt --ft_mode frozen --pooling attn --run_name cls_frozen_tapt
python scripts/train_cls.py --arch modern --backbone_ckpt outputs/checkpoints/lm_adapted_best.pt --ft_mode full   --pooling attn --class_weight --freeze_epochs 2 --run_name cls_best_tapt
python scripts/eval_classification.py
```

Then run notebooks 02 through 04 to regenerate the analysis and figures. Every
script logs to `outputs/logs/*.csv`, which can be tailed for live progress, and
checkpoints on every improvement, so a run can be resumed if interrupted.

## Limitations

The language model ablation and the distillation comparison are each a single run
per configuration; those deltas are directional evidence from one seed, not
statistically averaged results. The classification comparison is the one place
this project deliberately went further and used five-fold cross-validation,
precisely because the single-split ranking there turned out to be misleading. The
validation set for classification is only 144 reviews, so class-level metrics
carry real sampling variance at that scale. The per-fold domain adaptation used
inside the cross-validation harness measures its own training progress on the same
fold's training text, since only the resulting weights are needed there; the
properly held-out, leakage-free measurement of domain-adaptation perplexity is
reported once, separately, using a genuine held-out slice of review text. The
underlying language corpus is largely news and encyclopedic Hindi, so generation
quality and perplexity reflect that domain rather than general spoken or informal
Hindi. Finally, the finding that a classical model outperforms the Transformer
classifier under cross-validation is a consequence of the small labeled dataset
size (718 examples) and should be read in that context, not as a general claim
about Transformers versus classical methods for Hindi sentiment analysis.

## Contributors

- Sayan Das (sayan20012002@gmail.com)
- Senjuti Ghosal (senjutighosal09@gmail.com)
