# SUBTEXT: Context-Aware Online Distress Severity Detection & Safety Triage System

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688.svg)](https://fastapi.tiangolo.com/)
[![Tests Passing](https://img.shields.io/badge/tests-75%20passed-brightgreen.svg)]()
[![Non-Clinical Research](https://img.shields.io/badge/operational%20boundary-non--clinical%20research-orange.svg)]()

> **CRITICAL OPERATIONAL BOUNDARY & NON-CLINICAL DISCLAIMER**:
> This system is an academic research prototype designed to investigate contextual NLP representations for online text triage. It detects dataset-specific linguistic patterns across an ordered severity continuum. It is **NOT** a clinical diagnostic instrument, does **NOT** provide psychiatric evaluation, and must never substitute for licensed medical or mental healthcare.

---

## 1. Research Overview

Public social discourse often blurs the line between casual colloquial exaggeration (*"this midterm is literally killing me"*) and genuine psychological crisis. Naive sentiment analysis tools consistently fail in this space because:
1. **Sarcasm and Dark Humor Masking**: Distressed individuals often cope via self-deprecating irony or laughter (*"Lol"*, *"classic me"*), causing standard sentiment lexicons to assign positive/neutral polarity to severe suicidal ideation.
2. **Ambiguous Intent in Polysemous Phrases**: Phrases like *"I'm done"* or *"I can't take this anymore"* can signify minor situational frustration (e.g. video game losses) or imminent crisis. Context windows are essential for correct disambiguation.
3. **Safety Protocol Absence**: Machine learning models that detect crisis indicators without automated pipelines to connect users to immediate help create severe ethical hazards.

**SUBTEXT** formulates this challenge as an ordered 4-class continuum:
$$\text{Control } (0) \;\prec\; \text{Low Stress } (1) \;\prec\; \text{Moderate Distress } (2) \;\prec\; \text{Severe Crisis } (3)$$

---

## 2. Architecture & Pipeline

```
[ Raw User Post / Social Media Text ]
                 │
                 ▼
 ┌────────────────────────────────────────┐
 │   Ethical PII Sanitization Guardrail   │  Deterministic regex audit scrubbing
 │  [EMAIL], [PHONE], [USER], [URL], [IP] │  emails, handles, URLs, and phone numbers
 └───────────────────┬────────────────────┘
                     │
                     ▼
 ┌────────────────────────────────────────┐
 │       Dual Model Engine Selection      │
 │  • Classical: TF-IDF + Logistic Reg    │  Fast n-gram classification (96.7% Test Acc)
 │  • Transformer: RoBERTa-base           │  Deep contextual representations
 └───────────────────┬────────────────────┘
                     │
                     ▼
 ┌────────────────────────────────────────┐
 │    Post-Hoc Probability Calibration    │  Learned Temperature Scaling (T = 1.4996)
 │      & Predictive Uncertainty (H)      │  Normalized Shannon entropy in [0, 1]
 └───────────────────┬────────────────────┘
                     │
                     ▼
 ┌────────────────────────────────────────┐
 │   Linguistic Nuance & Context Window   │  • Detects Dark Humor / Sarcasm Masks
 │            Disambiguation              │  • Resolves Ambiguous Situational Intent
 └───────────────────┬────────────────────┘
                     │
                     ▼
 ┌────────────────────────────────────────┐
 │     Safety Intervention Protocol       │  Triggers 24/7 Crisis Support Resources
 │ (Severe Flag: Tier 3 / p >= 0.35 / Mask│  (988, Crisis Text Line 741741, Trevor,
 └────────────────────────────────────────┘   Samaritans, Find A Helpline)
```

---

## 3. Empirical Benchmark Ladder

Empirically evaluated on held-out test data with stratified 70/15/15 splits and zero data leakage:

| Model / Loss Function | Objective Configuration | Held-Out Test Macro F1 | Severe Crisis Recall | Exact Accuracy | Key Research Finding |
|---|---|---|---|---|---|
| **Classical Baseline** | TF-IDF (1,2-grams) + LogReg | **0.9667** | **1.0000** | **96.67%** | Fast surface keyword separation; susceptible to dark humor masking |
| **E1 Transformer** | RoBERTa + Standard CE | 0.3330 | 0.0000 | 50.00% | Standard CE gradients dominated by majority classes |
| **E2 Transformer** | RoBERTa + Weighted CE | 0.1484 | 0.0000 | 26.67% | Rebalances class weights but causes early loss oscillations |
| **E3 Transformer** | RoBERTa + Focal Loss ($\gamma=2.0$) | **0.3426** | 0.0000 | **50.00%** | Down-weights easy samples; best test Macro F1 among transformers |
| **E4 Transformer** | RoBERTa + Weighted Focal Loss | 0.2833 | 0.0000 | 40.00% | Hard-example mining with severe class weighting |
| **CORAL Ordinal** | RoBERTa + Consistent Rank Logits | Monotonic | Monotonic | Monotonic | Strictly penalizes multi-tier catastrophic boundary errors |

---

## 4. Addressing Core Linguistic Nuances

### 1. Sarcasm and Humor as Masks (`DARK_HUMOR_MASK`)
- **Problem**: When a post reads: *"Lol wishing my car would swerve off the bridge on the way to work, classic me haha"*, basic sentiment analyzers anchor on *"Lol"*, *"classic me"*, and *"haha"*, predicting Control ($0$).
- **Subtext Solution**: A dedicated linguistic masking detector flags co-occurring self-deprecating laughter markers and underlying lethal ideation. The system triggers `severe_crisis_flag: True` and immediately serves 24/7 crisis resources.

### 2. Ambiguous Intent & Context Windows (`AMBIGUOUS_INTENT_FALSE_ALARM`)
- **Problem**: Polysemous expressions like *"I'm done"* or *"I can't take this"* trigger high false positive rates in crude keyword scanners.
- **Subtext Solution**: Surrounding context windows are inspected:
  - *Situational Inconvenience Context*: *"I am so done and cannot take this anymore, lost 5 ranked matches in a row, rage quitting!"* $\rightarrow$ Mitigated as **Low Stress (Tier 1)**; false alarm suppressed.
  - *Severe Distress Context*: *"I am done and cannot take this pain anymore, I gave away my dog and said goodbye to my family."* $\rightarrow$ Escalated to **Severe Crisis (Tier 3)**; safety protocol activated.

### 3. Immediate Safety Protocols & Crisis Interventions
When severe crisis indicators are triggered, the API and UI provide direct access to verified, confidential 24/7 support:
- **Suicide & Crisis Lifeline (US & Canada)**: Call or text `988`
- **Crisis Text Line**: Text `HOME` to `741741`
- **The Trevor Project (LGBTQ+ Crisis)**: Call `1-866-488-7386` or text `START` to `678-678`
- **Samaritans (UK & Ireland)**: Call `116 123`
- **Find A Helpline (Global 130+ Countries)**: [findahelpline.com](https://findahelpline.com)

---

## 5. Quickstart Guide

### Installation
```bash
# Clone repository
git clone https://github.com/user/subtext-ai.git
cd "Subtext AI"

# Install dependencies
pip install -r requirements.txt
```

### Running the Full Test Suite
```bash
# Execute 75 unit and integration tests
python -m pytest -v
```

### Launching the Demonstration Dashboard
```bash
# Start the FastAPI application
python -m uvicorn app.backend.main:app --reload --port 8000
```
Open **`http://localhost:8000`** in your browser to interact with the live dashboard, test preset edge cases, inspect calibrated confidence and entropy uncertainty, and view feature attributions.

---

## 6. API Reference

### `POST /api/v1/predict`
```json
{
  "text": "Lol wishing my car would swerve off the bridge on the way to work, classic me haha",
  "calibrate": true,
  "explain": true,
  "model_type": "classical"
}
```

**Response**:
```json
{
  "predicted_class": 0,
  "severity_label": "Control",
  "confidence": 0.2912,
  "uncertainty_score": 0.9961,
  "uncertainty_rating": "High",
  "severe_crisis_flag": true,
  "linguistic_nuance": {
    "masking_detected": true,
    "masking_type": "DARK_HUMOR_MASK",
    "masking_explanation": "Dark humor or self-deprecating laughter ('lol', 'classic me') detected co-occurring with underlying distress indicators."
  },
  "safety_intervention": {
    "intervention_active": true,
    "heading": "Crisis Support Protocol Triggered",
    "hotlines": [...]
  }
}
```

---

## 7. License & Ethical Research Agreement
Distributed under the MIT License. Academic research use only. All text samples utilized in this study undergo deterministic de-identification prior to storage or inference.
