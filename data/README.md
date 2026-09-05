# Data Contract, Ethics, and Governance Specification

## 1. Ethical Governance & Non-Clinical Framing
This dataset directory stores text artifacts for the **SUBTEXT** research NLP project:
> **Core Purpose**: A linguistic research system evaluating contextual transformer representations for distinguishing genuine distress severity from colloquial, figurative, sarcastic, or ordinary negative online text.
>
> **Strict Operational Boundary**: This project is **NOT a medical or clinical diagnostic system**. Predictions represent dataset-specific linguistic patterns, not a psychological evaluation or diagnostic assessment of any individual.

## 2. Privacy & De-Identification Protocol
Raw social media and online platform text frequently contains Personally Identifiable Information (PII). Under our data protection contract:
- **Zero Raw Data in Version Control**: All files in `data/raw/` and sensitive intermediate data in `data/interim/` are excluded via `.gitignore`.
- **Pre-Tokenization Sanitization**: Text must pass through the deterministic PII sanitization pipeline before tokenization, representation extraction, or model training.
- **Redaction Targets**:
  - Usernames / handles (`@handle`, `u/user`, `r/subreddit`)
  - URLs and hyperlinks (`http://...`, `https://...`)
  - Email addresses
  - Phone numbers and phone-like numeric sequences
  - IP addresses
  - Explicit named entity identifiers (names, specific locations)

## 3. Standard Conceptual Schema
All datasets ingested into the pipeline must conform to the following schema:

| Field Name | Type | Description | Allowed Values / Format |
|---|---|---|---|
| `id` | `string` | Unique deterministic identifier (UUID or salted SHA-256 hash) | Non-reversible unique string |
| `text` | `string` | The sanitized text content | UTF-8 string |
| `label` | `int` | Ordered distress severity label | `0`, `1`, `2`, `3` |
| `source` | `string` | Provenance / corpus name | e.g. `clpsych_2015`, `dreaddit`, `synthetic_benchmark` |
| `split` | `string` | Dataset partition | `train`, `val`, `test` |

### Label Hierarchy Definition
* **`0 — Control`**: Neutral, positive, or ordinary negative chatter with no meaningful distress signal (includes casual venting, banter, everyday frustration, sarcasm without personal despair).
* **`1 — Low Stress`**: Everyday acute or situational stress, academic/work burnout, low-level anxiety without profound hopelessness or self-harm ideation.
* **`2 — Moderate Distress`**: Substantial depressive symptoms, pervasive feelings of worthlessness, chronic isolation, or overwhelming emotional suffering without acute emergency risk.
* **`3 — Severe Crisis`**: Acute crisis, explicit suicidal intent, active self-harm ideation, farewell statements, or urgent high-risk distress signals.

## 4. Directory Structure
```text
data/
├── raw/        <- Immutable original data (Strictly ignored by git)
├── interim/    <- Sanitized, validated, and normalized text prior to final split
├── processed/  <- Final train/val/test splits (Parquet/Arrow format)
└── README.md   <- This document
```
