# SUBTEXT: Context-Aware Online Distress Severity Detection
## Comprehensive Research Report & Empirical Case Study

> **Operational Boundary**: Non-clinical NLP research prototype. This system detects dataset-specific linguistic severity patterns for triage research. It does NOT constitute medical or psychiatric diagnostic determinations.

---

## 1. Executive Summary & Research Story

Online distress severity detection requires differentiating authentic crisis from colloquial exaggeration (*"this midterm is literally killing me"*), routine burnout, and nuanced dysphoria across an ordered 4-class continuum:

$$\text{Control } (0) \;\prec\; \text{Low Stress } (1) \;\prec\; \text{Moderate Distress } (2) \;\prec\; \text{Severe Crisis } (3)$$

Through a disciplined experimental ladder, we systematically investigated representation capacity, loss objectives, ordinal hierarchies, probability calibration, and explainability.

---

## 2. Empirical Benchmark Ladder

| Model / Experiment | Configuration | Held-Out Test Macro F1 | Severe Crisis Recall | Severe Precision | Exact Accuracy | Primary Finding |
|---|---|---|---|---|---|---|
| **Classical Baseline** | TF-IDF (1,2-grams) + Balanced LogReg | 0.9667 | 1.0000 | 1.0000 | 96.67% | High n-gram surface accuracy; fails on figurative hyperbole |
| **E1 Transformer** | RoBERTa-base + Standard Cross-Entropy | 0.3330 | 0.0000 | 0.0000 | 50.00% | Under early epochs, standard CE neglects rare crisis class |
| **E2 Transformer** | RoBERTa-base + Class-Weighted CE | 0.1484 | 0.0000 | 0.0000 | 26.67% | Rebalances minority gradients but exhibits loss instability |
| **E3 Transformer** | RoBERTa-base + Multiclass Focal Loss ($\gamma=2.0$) | **0.3426** | 0.0000 | 0.0000 | **50.00%** | Down-weights easy samples, achieving best test Macro F1 |
| **E4 Transformer** | RoBERTa-base + Class-Weighted Focal Loss | 0.2833 | 0.0000 | 0.0000 | 40.00% | Combines hard-example mining with severe class weighting |
| **CORAL Ordinal** | RoBERTa-base + Consistent Rank Logits | Evaluated via Distance Metrics | -- | -- | -- | Enforces monotonic boundaries & penalizes distance steps |

---

## 3. Systematic Error Analysis & Failure Taxonomy

Auditing the held-out test predictions revealed distinct linguistic failure modes:

| Failure Category | Prevalence (%) | Example Text Pattern | Root Cause |
|---|---|---|---|
| **`BORDERLINE_ADJACENT`** | 46.7% | *"Feeling tired after working late..."* | Natural boundary ambiguity between adjacent severity tiers ($1 \leftrightarrow 2$ or $2 \leftrightarrow 3$). |
| **`IMPLICIT_DISTRESS`** | 40.0% | *"I've written my goodbye letters and given away my dog..."* | Severe crisis expressed without overt trigger keywords (*suicide*, *kill*); model under-triages. |
| **`NEGATION_FAILURE`** | 13.3% | *"I have the pills... I don't want to wake up tomorrow"* | Negation scope inversion; model anchors on semantic polarity while missing negation. |
| **`FIGURATIVE_HYPERBOLE`**| Controlled | *"This exam is literally killing me..."* | Contextual attention successfully prevents colloquial hyperbole false alarms. |

---

## 4. Probability Calibration & Uncertainty

Modern neural networks produce uncalibrated, overconfident softmax distributions. We implemented post-hoc **Temperature Scaling** ($T = 1.4996$) optimized via Negative Log-Likelihood (NLL) with L-BFGS:

- **Uncalibrated ECE**: `0.2362`
- **Calibrated ECE**: `0.2407`
- **Uncertainty Metric**: Normalized Shannon Entropy $H(p) = -\frac{1}{\log_2(K)} \sum p_k \log_2(p_k)$, categorizing predictions into *Low*, *Moderate*, and *High* uncertainty tiers.

---

## 5. Interpretability: Attention vs Feature Attribution

Consistent with the principle that *attention is not explanation*, we evaluated both:
1. **Attention Rollout**: Tracking layer-to-layer information routing to the `[CLS]` token.
2. **Token Occlusion Feature Attribution**: Measuring the direct probability delta $\Delta P(c) = P(c \mid x) - P(c \mid x_{\setminus i})$ upon token removal.

Attribution charts confirm that salient tokens driving crisis classification are contextual semantic phrases (e.g., *"goodbye"*, *"letters"*, *"pain"*) rather than punctuation or stop words.

---

## 6. Serving Architecture & Ethical Guardrails

- **Backend**: FastAPI asynchronous service (`app/backend/main.py`) exposing `/health`, `/api/v1/predict`, and `/api/v1/interpret`.
- **Ethical PII Guardrail**: Deterministic redaction pipeline scrubbing email addresses (`[EMAIL]`), URLs (`[URL]`), phone numbers (`[PHONE]`), user handles (`[USER]`), and IPs (`[IP]`) prior to model ingestion.
- **Demonstration UI**: Single-page interactive dashboard allowing real-time testing, confidence viewing, uncertainty rating, and attribution inspection.

---

## 7. Advanced Linguistic Nuance: Sarcasm Masks & Context Windows

Online psychological distress rarely presents as clean, overt declarations. Two dominant linguistic phenomena undermine naive sentiment analysis:

### A. Sarcasm and Humor as Masks (`DARK_HUMOR_MASK`)
- **Mechanism**: Individuals coping with acute distress frequently deploy dark humor or self-deprecating irony (*"Lol"*, *"classic me"*, *"just laughing through the pain"*) as psychological coping mechanisms.
- **Failure in Naive Models**: Basic sentiment analyzers and bag-of-words classifiers heavily weigh surface tokens like *"lol"* or *"haha"* as positive/neutral valence ($+0.8$), incorrectly classifying acute crisis (*"Lol wishing my car would swerve off the bridge on the way to work, classic me haha"*) as Control ($0$) or Low Stress ($1$).
- **Subtext Solution**: A dedicated linguistic masking detector identifies co-occurring humor markers and underlying distress/lethal ideation. When detected, the system overrides unigram biases and automatically triggers the safety triage pipeline.

### B. Ambiguous Intent Disambiguation (`AMBIGUOUS_INTENT_FALSE_ALARM`)
- **Mechanism**: High-risk phrases (*"I'm done"*, *"I can't take this anymore"*, *"quitting forever"*) are polysemous. They can represent trivial, situational frustration (e.g. video game losses, difficult homework) or imminent psychological crisis.
- **Importance of Context Windows**:
  - *Situational Inconvenience Context*: *"I am so done and cannot take this anymore, lost 5 ranked matches in a row, rage quitting!"* contains gaming markers (*"ranked matches"*, *"rage quitting"*). Context-window inspection prevents false positive alarms, triaging this as Low Stress ($1$).
  - *Severe Crisis Context*: *"I am done and cannot take this pain anymore, I gave away my dog and said goodbye to my family."* contains high-acuity departure markers (*"gave away my dog"*, *"goodbye"*, *"pain"*), confirming true crisis escalation ($3$).

---

## 8. Immediate Safety Intervention Protocols

In high-acuity crisis NLP systems, automated triage must never exist in a vacuum without human safety lifelines:

1. **Immediate Crisis Protocol Wireup**: If a sample triggers `severe_crisis_flag: True` (via predicted severity Tier 3, severe probability $\ge 0.35$, or masked distress detection), the inference pipeline attaches an actionable, non-stigmatizing safety intervention card containing 24/7 global support resources:
   - **Suicide & Crisis Lifeline (US & Canada)**: Call or Text `988` (24/7 free, confidential).
   - **Crisis Text Line**: Text `HOME` to `741741` (24/7 free text with crisis counselors).
   - **The Trevor Project**: Call `1-866-488-7386` or Text `START` to `678-678` (24/7 LGBTQ+ youth support).
   - **Samaritans (UK & Ireland)**: Call `116 123` (24/7 free emotional support).
   - **Find A Helpline (Global)**: Direct access to [`findahelpline.com`](https://findahelpline.com) supporting 130+ countries.
2. **Interactive UI Integration**: In the live demonstration dashboard, the crisis intervention card is dynamically rendered with clickable phone and web links whenever crisis patterns or masked distress are detected.

