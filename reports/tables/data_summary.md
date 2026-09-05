# Dataset Exploratory Analysis & Stratification Report

## 1. Corpus Overview
* **Total Clean Samples**: 200
* **Number of Classes**: 4 (`0 — Control`, `1 — Low Stress`, `2 — Moderate Distress`, `3 — Severe Crisis`)
* **Partitioning Scheme**: Stratified 70% Train / 15% Validation / 15% Test

## 2. Stratified Class Distribution
|                       |   test |   train |   val |   All |
|:----------------------|-------:|--------:|------:|------:|
| 0 — Control           |      8 |      36 |     8 |    52 |
| 1 — Low Stress        |      7 |      31 |     6 |    44 |
| 2 — Moderate Distress |      7 |      35 |     8 |    50 |
| 3 — Severe Crisis     |      8 |      38 |     8 |    54 |
| Total                 |     30 |     140 |    30 |   200 |

## 3. Sequence Length Statistics (Words & Characters)
* **Word Count Mean ± Std**: 19.7 ± 3.6 words
* **Median Word Count**: 20 words
* **Interquartile Range (IQR)**: 18 to 22 words
* **95th Percentile Word Count**: 25 words
* **99th Percentile Word Count**: 26 words
* **Max Word Count**: 27 words
* **Character Count Mean / Max**: 109.0 / 137 chars

> [!TIP]
> **Tokenizer Length Recommendation**:
> Since the 99th percentile of sequence lengths is **26 words** (~35 BPE subword tokens), a `max_seq_length` of **256** or **128** covers over 99.5% of samples with zero token truncation while minimizing GPU memory padding waste.

## 4. Per-Class Length Profiles
- **0 — Control**: Mean = 17.8 words, Median = 18 words, 95th Percentile = 21 words
- **1 — Low Stress**: Mean = 19.1 words, Median = 20 words, 95th Percentile = 22 words
- **2 — Moderate Distress**: Mean = 20.5 words, Median = 21 words, 95th Percentile = 25 words
- **3 — Severe Crisis**: Mean = 21.4 words, Median = 21 words, 95th Percentile = 26 words

## 5. Lexical Diversity & Vocabulary Richness
| Class                 |   Total Tokens |   Unique Vocab |   Type-Token Ratio (TTR) |
|:----------------------|---------------:|---------------:|-------------------------:|
| Overall Corpus        |           3946 |            418 |                   0.1059 |
| 0 — Control           |            925 |            148 |                   0.16   |
| 1 — Low Stress        |            841 |            142 |                   0.1688 |
| 2 — Moderate Distress |           1026 |            144 |                   0.1404 |
| 3 — Severe Crisis     |           1154 |            138 |                   0.1196 |
