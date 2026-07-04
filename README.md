# Fine-Tuning Pipeline with LoRA on a Domain-Specific Dataset

An end-to-end, reproducible pipeline that builds a domain-specific dataset (legal clause classification), fine-tunes an open-source base model with LoRA/QLoRA, evaluates the adapter head-to-head against the base model on a task benchmark, checks for catastrophic forgetting, and packages the adapter for deployment.

Live demo: https://mansimengde17.github.io/Fine-Tuning-Pipeline-with-LoRA-on-a-Domain-Specific-Dataset/

## Design

Fine-tuning is where AI engineering meets ML engineering, and the pipeline is the deliverable: every run is reproducible from its config file alone, every claim about improvement is backed by the benchmark in this repository, and the same evaluation harness runs against any model.

```
 data/build_dataset.py            configs/*.yaml
   instruction-format legal          LoRA rank, alpha, dropout, lr,
   clause examples,                  epochs, quantization, seed
   80/10/10 splits, no leakage          |
        |                               v
        |                    +---------------------+
        +------------------->|  src/finetune/train  |
                             |  PEFT + TRL, QLoRA,  |
                             |  checkpoints, early  |
                             |  stopping, tracking  |
                             +---------------------+
                                        |
                              adapter (< 100 MB)
                                        |
                                        v
                             +---------------------+
                             | src/finetune/evaluate|
                             |  base vs fine-tuned  |
                             |  on the benchmark,   |
                             |  per-category deltas,|
                             |  LLM-as-judge,       |
                             |  forgetting check    |
                             +---------------------+
                                        |
                                        v
                             +---------------------+
                             |  src/finetune/serve  |
                             |  A/B comparison API  |
                             +---------------------+
```

## Quick start

Everything below runs offline in simulation mode, which exercises the entire pipeline (dataset build, training loop mechanics, checkpoint selection, evaluation, reporting) with a simulated model pair so the pipeline itself is verifiable on any machine:

```bash
pip install -r requirements.txt
python data/build_dataset.py           # builds train/val/test + benchmark
python demo.py                         # full pipeline in simulation mode
```

On a GPU machine with the ML extras installed (`pip install -r requirements-gpu.txt`), the same commands run real training:

```bash
python -m src.finetune.train --config configs/lora_r16.yaml
python -m src.finetune.train --sweep configs/sweep.yaml
python -m src.finetune.evaluate --adapter outputs/best
uvicorn src.finetune.serve:app        # POST /v1/compare for base vs tuned
```

## The dataset

Legal clause classification: given a contract clause, label it as one of `indemnification`, `limitation_of_liability`, `termination`, `confidentiality`, `governing_law`, or `payment_terms`. `data/build_dataset.py` generates instruction-format examples from curated clause templates with controlled paraphrase variation, then splits 80/10/10 with template-level separation between splits so no near-duplicate leaks from train into test. A separate 36-case handcrafted benchmark covers edge cases: mixed clauses, archaic phrasing, and near-miss vocabulary.

## Hyperparameters

The baseline config uses LoRA rank 16, alpha 32 (2x rank), dropout 0.05, targeting `q_proj` and `v_proj`, with 4-bit quantization (QLoRA) for consumer GPUs. `configs/sweep.yaml` sweeps rank (8, 16, 32), learning rate (1e-4, 2e-4, 5e-4), and epochs (1, 3, 5); every run logs config, loss curves, and validation metrics so the final choice is data-driven, not folklore.

## Evaluation protocol

1. The base model runs the full benchmark first; those scores are the denominator of every claim.
2. The fine-tuned adapter runs the identical benchmark with identical scoring.
3. Per-category accuracy deltas plus the specific examples where fine-tuning helped or hurt.
4. LLM-as-judge blind comparison on each test example (1 to 5 quality scale).
5. Catastrophic forgetting check: a general-capability probe set (common sense QA, instruction following) runs on both models; a significant drop fails the run.

## Repository layout

```
data/build_dataset.py        dataset generation, splits, benchmark
configs/                     training configs and sweep definition
src/finetune/train.py        LoRA/QLoRA training with tracking
src/finetune/evaluate.py     benchmark harness, judge, forgetting check
src/finetune/serve.py        A/B comparison API
demo.py                      full pipeline in simulation mode
tests/
```
