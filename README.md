<div align="center">

# 🧠 Hybrid Neuro-Symbolic Visual Commonsense Reasoning

**YOLO Perception × Qwen2.5-VL Cognition — Token-Level Logit Probability Scoring**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-ee4c2c.svg)](https://pytorch.org/)
[![Apple Silicon](https://img.shields.io/badge/Apple_Silicon-MPS-black.svg)](https://developer.apple.com/metal/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

</div>

---

## Key Features

- **🍎 Apple Silicon Native** — First-class MPS backend support for M-series hardware acceleration with float16 precision
- **🔬 Dual-Stage Reasoning** — Two-stage evaluation pipeline: Answer Selection ($A^*$) → Rationale Selection ($R^*$) with joint $Q \rightarrow AR$ accuracy
- **📐 Mathematical Logit Scoring** — Deterministic token-level log-likelihood computation instead of stochastic text generation
- **🎯 YOLO Perception Layer** — Focused object-region crops provide grounded visual context for entity-specific reasoning
- **🧩 Modular Architecture** — Clean separation of concerns across `dataset`, `perception`, `cognition`, and `evaluate` modules

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        VCR Dataset (JSONL)                       │
│              question + answer_choices + rationale_choices        │
└──────────────────────────────┬──────────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────────┐
│                     PERCEPTION LAYER (YOLO)                      │
│                                                                  │
│   Scene Image ──► Bounding Boxes (from metadata) ──► PIL Crops   │
│                                                                  │
│   Output: { 0: crop_person1, 1: crop_person2, 2: crop_dog1 }    │
└──────────────────────────────┬───────────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────────┐
│                COGNITION LAYER (Qwen2.5-VL-7B)                   │
│                                                                  │
│   Interleaved Input:                                             │
│   [Scene Image] + [Crop₁] + [Crop₂] + [Text Prompt + Choice]    │
│                                                                  │
│   ┌──────────────────────────────────────────────────────────┐   │
│   │  STAGE 1: Answer Selection                               │   │
│   │  A* = argmax  Σᵢ log P(cᵢ | prompt, c₁...cᵢ₋₁)        │   │
│   │       A∈{A,B,C,D}                                       │   │
│   └──────────────────────────┬───────────────────────────────┘   │
│                              │                                   │
│                              ▼                                   │
│   ┌──────────────────────────────────────────────────────────┐   │
│   │  STAGE 2: Rationale Selection (conditioned on A*)        │   │
│   │  R* = argmax  Σᵢ log P(cᵢ | prompt, A*, c₁...cᵢ₋₁)    │   │
│   │       R∈{A,B,C,D}                                       │   │
│   └──────────────────────────┬───────────────────────────────┘   │
│                              │                                   │
└──────────────────────────────┼───────────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────────┐
│                       EVALUATION METRICS                         │
│                                                                  │
│   Q→A  Accuracy:  correct answers / total                        │
│   Q→R  Accuracy:  correct rationales / total                     │
│   Q→AR Accuracy:  both correct / total  ← PRIMARY METRIC        │
└──────────────────────────────────────────────────────────────────┘
```

---

## Repository Structure

```
Project AIMS/
├── src/
│   ├── __init__.py
│   ├── dataset.py          # BoundingBox, VCRSample, EvalResult, data loaders
│   ├── perception.py       # YOLO-backed region cropping layer
│   ├── cognition.py        # Qwen2.5-VL logit scoring engine
│   └── evaluate.py         # Main entry point, CLI, metrics, eval loop
├── requirements.txt
├── REPORT.md               # Technical report (4-page format)
├── README.md
└── vcr_eval.py             # Legacy monolithic script (deprecated)
```

---

## Installation & Setup

### Prerequisites

- Python 3.10+
- macOS with Apple Silicon (M1/M2/M3/M4/M5) for MPS acceleration
- Git

### Step 1: Clone the Repository

```bash
git clone https://github.com/your-username/project-aims.git
cd project-aims
```

### Step 2: Create Virtual Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Prepare the VCR Dataset

Download the VCR dataset and organize it as:

```
vcr1/
├── vcr1images/
│   └── [movie_name]/
│       ├── *.jpg
│       └── *.json
├── train.jsonl
└── val.jsonl
```

### Step 5: Verify MPS Backend

```bash
python3 -c "import torch; print('MPS available:', torch.backends.mps.is_available())"
```

---

## Usage

### Run Full Validation

```bash
python -m src.evaluate --vcr-root ./vcr1 --split val
```

### Debug on Mini-Sample

```bash
python -m src.evaluate --vcr-root ./vcr1 --split val --num-samples 5
```

### Custom Models & Output

```bash
python -m src.evaluate \
    --vcr-root ./vcr1 \
    --split val \
    --qwen-model Qwen/Qwen2.5-VL-7B-Instruct \
    --yolo-model yolo11n.pt \
    --output results/val_results.json
```

### Disable Length Normalization

```bash
python -m src.evaluate --vcr-root ./vcr1 --split val --no-length-norm
```

### CLI Reference

| Flag | Default | Description |
|------|---------|-------------|
| `--vcr-root` | *required* | Path to VCR dataset root directory |
| `--split` | `val` | Dataset split (`train` or `val`) |
| `--num-samples` | all | Limit evaluation to first N samples |
| `--qwen-model` | `Qwen/Qwen2.5-VL-7B-Instruct` | Hugging Face model identifier |
| `--yolo-model` | `yolo11n.pt` | YOLO weights file |
| `--no-length-norm` | `false` | Disable per-token length normalization |
| `--output` | `vcr_results_<split>.json` | Output JSON file path |

---

## Evaluation Metrics

| Metric | Formula | Description |
|--------|---------|-------------|
| **$Q \rightarrow A$** | $\frac{\text{correct answers}}{N}$ | Answer selection accuracy |
| **$Q \rightarrow R$** | $\frac{\text{correct rationales}}{N}$ | Rationale selection accuracy |
| **$Q \rightarrow AR$** | $\frac{\text{both correct}}{N}$ | **Primary ranking metric.** Joint accuracy — the model must select the correct answer AND the correct rationale for the same sample. |

The output JSON file contains per-sample predictions, ground-truth labels, correctness flags, and the raw log-likelihood scores for all candidates.

---

## Deliverables Checksheet

- [x] **Deliverable 1** — Modular source code (`src/dataset.py`, `src/perception.py`, `src/cognition.py`, `src/evaluate.py`)
- [x] **Deliverable 2** — `requirements.txt` with all dependencies
- [x] **Deliverable 3** — `README.md` with architecture diagram, setup, usage, and metrics
- [x] **Deliverable 4** — `REPORT.md` technical report (abstract, architecture, math, scorecard, trade-offs)
- [x] **Deliverable 5** — Apple Silicon MPS hardware acceleration support
- [x] **Deliverable 6** — Two-stage evaluation pipeline ($Q \rightarrow A$ → $Q \rightarrow R$ → $Q \rightarrow AR$)
- [x] **Deliverable 7** — Token-level logit probability scoring (no `model.generate()`)
- [x] **Deliverable 8** — JSON results output with per-sample log-likelihoods

---

## License

MIT

---

<div align="center">

**Project AIMS** — Hybrid Neuro-Symbolic Visual Commonsense Reasoning

</div>
