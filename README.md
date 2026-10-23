# ATTUNE: Context-Aware Preference Gating Engine for Technical Explanations

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Accuracy](https://img.shields.io/badge/Accuracy-98.5%25-brightgreen)](https://github.com/NITISH-027/attune)
[![Negative Transfer](https://img.shields.io/badge/Negative%20Transfer%20(NTR)-0.0%25-emerald)](https://github.com/NITISH-027/attune)
[![CPU Latency](https://img.shields.io/badge/CPU%20Inference-~25ms-blue)](https://github.com/NITISH-027/attune)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **ATTUNE** is a lightweight, calibrated preference gating engine that dynamically decides whether a user's prior stylistic, pedagogical, or structural preferences should be applied to an incoming technical query, completely suppressing inappropriate personalization across context shifts and explicit reversals.

---

## Executive Summary

State-of-the-art conversational AI assistants frequently suffer from **Negative Transfer (Over-Personalization)**: when a user expresses a preference in one domain (e.g., *"give me concise bullet points for algorithm problems"*), existing architectures blindly propagate that preference to subsequent, unrelated tasks (e.g., debugging a production database outage or learning a new language syntax).

**ATTUNE** solves this problem by treating preference application as an explicit **2-tier context-aware gating classification task**:
1. **Tier 1 (Golden Priority Rule)**: Catches immediate query-level explicit reversals and overrides, eliminating repeat correction errors with **0-turn adaptation latency** ($<0.1\text{ ms}$).
2. **Tier 2 (Calibrated Statistical Gate)**: Evaluates a structured tuple `(Current Request, Candidate Preference, Scope, Supporting Evidence, Override)` using a fine-tuned, temperature-calibrated encoder (`all-MiniLM-L6-v2`) against an empirically optimal decision threshold ($\tau^* = 0.9000$).

On a strictly partitioned, zero-leakage test benchmark ($N = 200$), ATTUNE achieves **98.5% classification accuracy** and **0.0% Negative Transfer Rate (NTR)**, retaining **96.25% of valid personalization** while running in **$\sim 24.7\text{ ms}$ on standard CPU hardware**.

---

## The Core Problem: Why Similarity $\ne$ Applicability

Traditional conversational memory systems fall into two pathological traps:

```
+-----------------------------------------------------------------------------------------------+
| TRADITIONAL ARCHITECTURES                                                                     |
+------------------------------------+----------------------------------------------------------+
| 1. Fixed Profiles (Custom GPTs)    | Permanently appends preference strings to system prompts.|
|                                    | Result: 100% Negative Transfer on context switches and    |
|                                    | 100% Repeated-Correction Rate (RCR).                     |
+------------------------------------+----------------------------------------------------------+
| 2. Vector History RAG              | Top-k semantic retrieval over past turns embeddings.     |
|                                    | Result: Keyword confounding! Technical overlap (e.g.,    |
|                                    | 'tree', 'complexity') triggers retrieval even during     |
|                                    | radical context shifts.                                  |
+------------------------------------+----------------------------------------------------------+
| ATTUNE (Context-Aware Gating)      | Evaluates applicability probability p_hat against tau*.   |
|                                    | Injects preference ONLY when contextually eligible.       |
|                                    | Result: 0.0% NTR, 0.0% RCR, 98.5% accuracy.              |
+------------------------------------+----------------------------------------------------------+
```

---

## System Architecture

ATTUNE employs a **2-Tier Decision Pipeline** designed for high throughput and zero latency overhead:

```
                            Incoming User Query
                                    |
                                    v
            +-----------------------------------------------+
            |  Tier 1: The Golden Priority Rule             |
            |  (Query-Level Explicit Reversal Pre-Filter)   |
            +-----------------------------------------------+
                     /                             \
     [Override Detected]                   [No Active Override]
           /                                         \
          v                                           v
+-----------------------+              +-------------------------------+
|  IMMEDIATE SUPPRESS   |              |  Tier 2: Preference Gating    |
|  Decision = 0         |              |  - Context Tuple Serializer   |
|  p_hat = 0.0          |              |  - all-MiniLM-L6-v2 Backbone  |
|  Latency < 0.1 ms     |              |  - Temperature Scaling (T*)   |
+-----------------------+              +-------------------------------+
          |                                           |
          |                                   Compute p_hat
          |                                           |
          |                       +-------------------+-------------------+
          |                      /                                         \
          |              [p_hat >= tau* (0.900)]                     [p_hat < tau*]
          |                     /                                           \
          |                    v                                             v
          |     +-----------------------------+               +-----------------------------+
          |     |    GATE OPEN: APPLY         |               |   GATE CLOSED: SUPPRESS     |
          |     |    Inject Style Directive   |               |   Prevent Negative Transfer |
          |     +-----------------------------+               +-----------------------------+
          \                    |                                             /
           \                   v                                            /
            \---> +-------------------------------------------------------+ <---/
                  |       Dynamic Technical CS Response Generator        |
                  |     (Gemini API / Domain-Grounded CS Generator)       |
                  +-------------------------------------------------------+
                                              |
                                              v
                              Assistant Output + Telemetry
```

---

## Empirical Benchmark Results

### 1. Master 5-Way Comparative Scorecard (`data/splits/test.jsonl`, $N = 200$)

Evaluated under strict zero-leakage grouped partitioning (no scenario family or episode ID overlap between train and test):

| System Architecture | Accuracy | F1 Score | NTR *(Over-Pers.)* | FSR *(Under-Pers.)* | RCR *(Correction)* | Latency / Turn |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **System A: Zero-Shot Base** | 0.600 | 0.000 | 0.000 | 1.000 | 0.000 | 0.00 ms |
| **System B: Fixed Profile** | 0.400 | 0.571 | 1.000 | 0.000 | 1.000 | 0.00 ms |
| **System C: History Retrieval (RAG)** | 0.600 | 0.000 | 0.000 | 1.000 | 0.000 | 22.35 ms |
| **System D: Deterministic Rule Gate** | 0.900 | 0.889 | 0.167 | 0.000 | 0.000 | 0.01 ms |
| **System E: ATTUNE Gating Engine** | **0.985** | **0.981** | **0.000** | **0.038** | **0.000** | **24.73 ms** |

*Definitions:*
- **NTR (Negative Transfer Rate)**: $\frac{\text{False Positives}}{\text{Total True Negatives}}$ (Target: $\le 0.10$). ATTUNE achieves **0.000**.
- **FSR (False Suppression Rate)**: $\frac{\text{False Negatives}}{\text{Total True Positives}}$ (Target: $\le 0.10$). ATTUNE achieves **0.038**.
- **RCR (Repeated-Correction Rate)**: Rate of repeating an error after an explicit override. ATTUNE achieves **0.000**.

---

### 2. Zero-Shot External Transfer Generalization

To test generalization beyond synthetic technical CS distributions, System E was evaluated zero-shot (with frozen weights and threshold $\tau^* = 0.900$) against external held-out benchmarks:

| External Benchmark | Domain & Type | $N$ | Accuracy | Precision | Recall | NTR | FSR | Mean Latency |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **PrefEval** | CS Technical Explanations | 150 | **1.000** | **1.000** | **1.000** | **0.000** | **0.000** | 24.50 ms |
| **CUPID** | Open-Domain Dialogue | 756 | 0.672 | 0.526 | 0.163 | **0.073** | 0.837 | 20.67 ms |

- On **PrefEval** (in-domain technical reasoning), ATTUNE achieves **100% accuracy** with zero negative transfer.
- On **CUPID** (open-domain chat), ATTUNE operates conservatively, actively suppressing **92.7% of negative transfer** ($\text{NTR} = 7.3\%$) despite domain shift.

---

### 3. Probability Calibration & Computational Efficiency

- **Post-Hoc Calibration**: Fitted on validation logit distribution using L-BFGS temperature scaling:
  - Optimal Temperature: $T^* = \mathbf{0.8453}$
  - Expected Calibration Error (ECE): Reduced from $0.0242 \to \mathbf{0.0125}$ (48.3% improvement).
- **Decision Threshold Optimization**: $\tau^* = \mathbf{0.9000}$ chosen to guarantee the specification gate $\text{NTR} \le 0.05$.
- **Latency & Footprint**:
  - Backbone: `all-MiniLM-L6-v2` (22.7M parameters, 88 MB checkpoint).
  - Mean CPU Latency: **$24.73\text{ ms}$** per query.
  - P99 CPU Latency: **$34.72\text{ ms}$** (exceeds the $\le 35\text{ ms}$ graduation target).

---

## Diagnostics & Failure Analysis

A comprehensive failure analysis was conducted over the held-out test split (`eval/failure_analysis.md`):

1. **Confusion Matrix Summary**:
   - $\text{True Positives} = 77 \quad | \quad \text{True Negatives} = 120$
   - $\text{False Positives} = 0 \quad (\text{NTR} = 0.000\%)$
   - $\text{False Negatives} = 3 \quad (\text{FSR} = 3.750\%)$
2. **Empirical Safety Margin**:
   - Mean predicted probability for true positives: $\mu_+ = 0.9649$.
   - Mean predicted probability for true negatives: $\mu_- = 0.0062$.
   - Separation margin: $\Delta p = \mathbf{0.9587}$, demonstrating extreme bimodality.
3. **Root Cause for 3 False Negatives**:
   - All 3 false negatives belonged to the continuous prose dimension (`STRUCTURE:CONTINUOUS`) combined with dual-aspect prompts (*"Analyze time and space complexity trade-offs... in continuous prose"*). The gate assigned $\hat{p} \in [0.3757, 0.7026]$, falling below the conservative threshold $\tau^* = 0.9000$. The resulting failure mode is safe: the system defaulted to neutral baseline explanations without corrupting correctness.

---

## Quickstart Guide

### 1. Setup Environment

```powershell
# Clone the repository
git clone https://github.com/NITISH-027/attune.git
cd attune

# Install dependencies
pip install -r requirements.txt
```

### 2. Launch Interactive Demonstration & Policy Inspector

Launch the FastAPI application with live Policy Inspector UI:

```powershell
# Using the PowerShell launcher
.\run_demo.ps1

# Or run directly via Python
python app/server.py
```

Then open your browser to: **`http://127.0.0.1:8000`**

- Test **Preset A (Context Shift)**: Watch the gate suppress algorithm preferences when asking about Git CLI.
- Test **Preset B (Same-Domain)**: Watch the gate inject concise styling when asking about AVL Trees.
- Test **Preset C (Explicit Override)**: Watch Tier 1 Golden Rule trigger instant zero-latency reversal.

### 3. Reproduce the Full Benchmark Suite

Run the master evaluation script across Systems A through E and external benchmarks:

```powershell
python eval/evaluate_all.py
```

Run the failure diagnostic engine:

```powershell
python eval/failure_analysis.py
```

---

## Repository Structure

```
attune/
├── app/
│   ├── engine.py                 # Multi-session runtime coordinator & generator
│   ├── server.py                 # FastAPI server (endpoints: /api/chat, /api/status, /api/reset)
│   └── static/
│       └── index.html            # Dark-mode dashboard with live Policy Inspector drawer
├── baselines/
│   ├── zero_shot.py              # System A: Zero-shot baseline (always 0)
│   ├── fixed_profile.py          # System B: Fixed user profile (always 1)
│   ├── history_rag.py            # System C: Dense embedding similarity RAG gate
│   └── rule_based.py             # System D: Deterministic regex/keyword baseline gate
├── data/
│   ├── synthetic/
│   │   ├── generate_data.py      # Stratified dataset generator (2,000 tuples across 50 families)
│   │   └── attune_raw_synthetic.jsonl
│   ├── splits/
│   │   ├── train.jsonl           # 80% (1,600 tuples, 40 families)
│   │   ├── val.jsonl             # 10% (200 tuples, 5 families)
│   │   └── test.jsonl            # 10% (200 tuples, 5 families)
│   ├── external_benchmarks/
│   │   ├── download_benchmarks.py
│   │   ├── cupid_test.jsonl      # Held-out open-domain benchmark (N = 756)
│   │   └── prefeval_test.jsonl   # Held-out technical benchmark (N = 150)
│   └── split_and_verify.py       # Zero-leakage verification suite
├── docs/
│   ├── spec.md                   # Complete architectural and statistical specification
│   └── demo_script.md            # 90-second recruiter presentation guide & interview Q&A
├── eval/
│   ├── metrics.py                # Mathematical metrics (NTR, FSR, RCR, ECE, Accuracy, F1)
│   ├── run_benchmark.py          # Baseline comparison runner
│   ├── evaluate_all.py           # Master 5-system comparative evaluation script
│   ├── failure_analysis.py       # Diagnostic error inspection engine
│   ├── final_scorecard.json      # Comprehensive evaluation metrics artifact
│   └── failure_analysis.md       # Diagnostic report & trade-off analysis
├── models/
│   ├── attune_gate_best.pt       # Fine-tuned PyTorch checkpoint (88 MB)
│   └── config.json               # Model hyperparameters, T*, and tau*
├── src/
│   ├── dataset.py                # PyTorch Dataset & tuple serialization
│   ├── model.py                  # AttuneGatingModel architecture
│   ├── calibration.py            # Temperature scaling & optimal threshold selection
│   ├── train.py                  # PyTorch CPU training loop
│   └── predict.py                # Production 2-tier inference engine
├── requirements.txt              # Core project dependencies
├── run_demo.ps1                  # PowerShell one-click demo launcher
├── run_demo.bat                  # Windows batch one-click demo launcher
└── README.md                     # Portfolio documentation & architectural overview
```

---

## Citation & License

This project is licensed under the **MIT License**.

```bibtex
@software{attune2026,
  author = {ATTUNE Research Team},
  title = {ATTUNE: A Context-Aware Preference Gating Engine for Technical Explanations in Computer Science},
  year = {2026},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/your-username/attune}}
}
```
