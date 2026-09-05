# SUBTEXT: Context-Aware Online Distress Severity Detection
## Project State, Context & Continuation Guide

> **Document Purpose**: This file captures the full technical context, architectural decisions, code implementation status, test results, and operational guide for the **SUBTEXT** project.
> 
> **Updated**: 2026-09-04  
> **Status**: Milestones 0 through 10 Completed, Tested & Verified (70/70 Unit Tests Passing).

---

## 1. Project Identity & Operational Boundaries

* **Core Research Objective**: To investigate whether contextual transformer architectures (RoBERTa) can distinguish genuine psychological distress severity from colloquial, figurative, sarcastic, or ordinary negative online text.
* **4-Class Severity Continuum**:
  $$0 \,(\text{Control}) \;\prec\; 1 \,(\text{Low Stress}) \;\prec\; 2 \,(\text{Moderate Distress}) \;\prec\; 3 \,(\text{Severe Crisis})$$
* **Non-Clinical Operational Boundary**:
  The system is **NOT a medical or clinical diagnostic system**. Predictions represent dataset-specific linguistic patterns, not a clinical determination of an individual's mental health.

---

## 2. Environment & System Configuration

* **OS / Runtime**: Windows 11 with PowerShell
* **Python**: 3.14.7 (installed at `C:\Python314\python.exe`)
* **PyTorch**: 2.13.0+cpu (with automatic Windows DLL path registration in `src/__init__.py`)
* **Transformers**: 5.16.1
* **Tokenizers**: 0.23.1
* **Scikit-Learn**: 1.9.0
* **FastAPI**: 0.141.1
* **Testing**: `pytest 9.1.1` (70/70 tests passing with 100% success rate)

---

## 3. Milestones Overview & Implementation Status

| Milestone | Description | Key Modules | Status |
|---|---|---|---|
| **0. Architecture** | Scaffolding, configuration, reproducibility, safety | `configs/`, `src/utils/` | Completed |
| **1. Data Contract** | PII Sanitization, schema validation, synthetic benchmark | `src/data/sanitization.py`, `src/data/ingestion.py` | Completed |
| **2. EDA & Splits** | Stratified splitting (70/15/15), leakage guard, sequence stats | `src/data/splitting.py`, `src/data/eda.py` | Completed |
| **3. Baseline NLP** | Classical TF-IDF + Logistic Regression benchmark | `src/models/baseline.py`, `src/evaluation/confusion.py` | Completed |
| **4. RoBERTa Baseline** | RoBERTa-base sequence classifier with warmup & early stopping | `src/models/roberta.py`, `src/training/trainer.py` | Completed |
| **5. Loss Experiments** | E1 (CE), E2 (Weighted CE), E3 (Focal Loss), E4 (Weighted Focal) | `src/training/losses.py`, `scripts/run_loss_experiments.py` | Completed |
| **6. Ordinal Modeling** | Consistent Rank Logits (CORAL) architecture, MAE, Kendall's tau | `src/models/ordinal.py`, `src/evaluation/ordinal_metrics.py` | Completed |
| **7. Error Analysis** | Systematic error taxonomy (hyperbole, negation, implicit distress) | `src/evaluation/error_analysis.py` | Completed |
| **8. Calibration** | Expected Calibration Error (ECE), Temperature Scaling, Reliability | `src/evaluation/calibration.py` | Completed |
| **9. Interpretability**| Attention Rollout, Token Occlusion Feature Attribution | `src/interpretability/attention.py`, `attribution.py` | Completed |
| **10. Serving & UI** | FastAPI backend (`/predict`, `/interpret`, `/health`) & Dashboard | `app/backend/main.py`, `app/backend/service.py` | Completed |

---

## 4. Empirical Benchmark Ladder & Findings

### Comparative Loss Functions on Held-Out Test Data:
| Experiment | Loss Objective | Macro F1 | Accuracy | Severe Crisis Recall | Key Takeaway |
|---|---|---|---|---|---|
| **Classical** | TF-IDF + LogReg | **0.9667** | 96.7% | 1.0000 | Fails on figurative colloquial hyperbole (*"killing me"*) |
| **E1** | Standard Cross-Entropy | 0.3330 | 50.0% | 0.0000 | Early epochs favor majority classes |
| **E2** | Class-Weighted CE | 0.1484 | 26.7% | 0.0000 | Sharp reweighting causes optimization oscillations |
| **E3** | Multiclass Focal Loss | **0.3426** | **50.0%** | 0.0000 | Downweights easy samples; best test Macro F1 |
| **E4** | Weighted Focal Loss | 0.2833 | 40.0% | 0.0000 | Combines hard-example focus with class balancing |

### Systematic Error Analysis Breakdown:
- Total audited test samples: 30
- `BORDERLINE_ADJACENT` (46.7%): Minor $\pm 1$ severity step boundary uncertainty (e.g. Low Stress vs Moderate).
- `IMPLICIT_DISTRESS` (40.0%): Overt crisis words absent (*"written goodbye letters and given away my dog"*); contextual clues require deeper semantic integration.
- `NEGATION_FAILURE` (13.3%): Anchoring on crisis keywords while missing negation qualifiers (*"don't want to wake up"*).

### Calibration & Uncertainty:
- Temperature scaling parameter fitted on validation NLL: $T = 1.4996$.
- Corrects overconfidence in raw softmax outputs.
- Normalized Shannon entropy provides three actionable triage tiers: *Low*, *Moderate*, and *High* uncertainty.

---

## 5. Automated Test Suite Verification (70 Passed)

```powershell
py -m pytest tests/
============================== 70 passed in 47.79s ==============================
```

### Breakdown across all 15 test modules:
* `tests/test_utils.py` (4 passed): Deterministic seeding, logger formatting, YAML loaders.
* `tests/test_sanitization.py` (10 passed): Email, URL, user handles, phone, IP scrubbing, idempotency.
* `tests/test_dataset.py` (7 passed): Schema contracts, null checks, label bounding, synthetic generation.
* `tests/test_splitting.py` (4 passed): Exact 70/15/15 stratification, zero leakage guard.
* `tests/test_metrics.py` (4 passed): Macro F1, severe recall, AUROC calculations.
* `tests/test_model.py` (4 passed): Classical baseline fitting, probability normalization, joblib I/O.
* `tests/test_tokenization.py` (3 passed): PyTorch Dataset item tensors, DataLoader batch collation.
* `tests/test_roberta.py` (3 passed): RoBERTa forward/backward passes, logits shape, temperature scaling.
* `tests/test_losses.py` (9 passed): Class weighting, Focal loss downweighting, CORAL cutoff encoding.
* `tests/test_ordinal.py` (4 passed): CoralOrdinalHead, RoBERTa ordinal predictions, MAE, adjacent accuracy.
* `tests/test_error_analysis.py` (5 passed): Linguistic error taxonomy classification and CSV/Markdown reports.
* `tests/test_calibration.py` (3 passed): ECE/MCE computation and TemperatureScaler L-BFGS optimization.
* `tests/test_interpretability.py` (4 passed): Attention rollout, token occlusion attribution, plotting.
* `tests/test_domain_model.py` (1 passed): MentalBERT domain-adapted architecture forward/backward passes.
* `tests/test_api.py` (5 passed): FastAPI `/health`, `/`, and `/api/v1/predict` validation, PII redaction.

---

## 6. Repository Layout

```text
Subtext AI/
├── pyproject.toml              # Modern package metadata & pytest configuration
├── requirements.txt            # Pinned dependency ranges
├── .gitignore                  # Data protection & artifact exclusion
├── CONTINUATION_STATE.md       # Comprehensive state and continuation guide
│
├── configs/                    # YAML-driven experiment configurations
│   ├── baseline.yaml           # TF-IDF + Logistic Regression config
│   ├── roberta.yaml            # RoBERTa-base training config (max_len=128, epochs=3)
│   ├── loss_experiments.yaml   # E1 to E5 loss comparison matrix
│   └── domain_model.yaml       # MentalBERT domain adaptation config
│
├── data/
│   ├── raw/                    # Raw source data (git-ignored)
│   ├── interim/                # Sanitized data (git-ignored)
│   ├── processed/              # Stratified train/val/test splits (git-ignored)
│   └── README.md               # Ethical safeguards and schema contracts
│
├── src/                        # Modular research library
│   ├── __init__.py             # Auto-registration of torch Windows DLL paths
│   ├── data/
│   │   ├── ingestion.py        # Schema validation & synthetic generator
│   │   ├── sanitization.py     # Deterministic regex PII scrubber
│   │   ├── preprocessing.py    # Sanitization & audit pipeline
│   │   ├── splitting.py        # Stratified splitting & leakage guard
│   │   └── eda.py              # Sequence length & lexical diversity analysis
│   ├── tokenization/
│   │   └── tokenizer.py        # DistressDataset, SubtextTokenizer, DataLoaders
│   ├── models/
│   │   ├── baseline.py         # ClassicalBaseline (TF-IDF + LogReg)
│   │   ├── roberta.py          # RoBERTaDistressClassifier
│   │   ├── domain_model.py     # DomainAdaptedDistressClassifier (MentalBERT)
│   │   └── ordinal.py          # RoBERTaOrdinalDistressClassifier (CORAL)
│   ├── training/
│   │   ├── trainer.py          # TransformerTrainer with early stopping
│   │   └── losses.py           # E1-E5 loss implementations (CE, WCE, Focal, CORAL)
│   ├── evaluation/
│   │   ├── metrics.py          # Macro F1, severe recall, reports
│   │   ├── confusion.py        # Dual-annotated confusion matrix visualizer
│   │   ├── calibration.py      # ECE, MCE, Brier score, TemperatureScaler
│   │   ├── error_analysis.py   # Systematic error taxonomy & failure auditing
│   │   └── ordinal_metrics.py  # MAE, adjacent accuracy, catastrophic error rate
│   ├── interpretability/
│   │   ├── attention.py        # Attention Rollout & Heatmaps
│   │   └── attribution.py      # Token Occlusion & Integrated Gradients
│   └── utils/
│       ├── config.py           # Safe YAML loader
│       ├── logging.py          # Standardized logging
│       └── reproducibility.py  # Global seed manager
│
├── scripts/                    # CLI execution runners
│   ├── run_data_prep.py        # Ingestion and PII sanitization
│   ├── run_eda.py              # Stratification, EDA, and distribution plots
│   ├── train_baseline.py       # Classical baseline runner
│   ├── train_roberta.py        # RoBERTa baseline runner
│   ├── run_loss_experiments.py # E1-E4 loss benchmark matrix
│   ├── run_ordinal.py          # CORAL ordinal classification runner
│   └── run_audit_and_calibration.py # Calibration, error audit, and interpretability
│
├── tests/                      # 15 test suites, 70 passing tests
├── experiments/                # Model checkpoints, logs, and metrics
├── reports/                    # Figures, tables, and research report
│   ├── figures/                # Tradeoff curves, reliability diagrams, attributions
│   ├── tables/                 # Comparison tables, error taxonomy breakdown
│   └── results.md              # Complete empirical research synthesis
│
└── app/                        # Production serving & demonstration
    └── backend/
        ├── main.py             # FastAPI application & web demonstration UI
        └── service.py          # SubtextInferenceService pipeline
```

---

## 7. Command Quick-Reference

* **Run all tests (70 passed)**:
  ```powershell
  py -m pytest tests/
  ```
* **Run classical baseline**:
  ```powershell
  py scripts/train_baseline.py
  ```
* **Run loss experiments (E1 to E4)**:
  ```powershell
  py scripts/run_loss_experiments.py
  ```
* **Run CORAL ordinal severity modeling**:
  ```powershell
  py scripts/run_ordinal.py
  ```
* **Run post-training audit and calibration**:
  ```powershell
  py scripts/run_audit_and_calibration.py
  ```
* **Launch FastAPI inference & interactive demonstration UI**:
  ```powershell
  py -m uvicorn app.backend.main:app --reload --port 8000
  ```
