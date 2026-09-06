# Dataset Exploratory Analysis & Stratification Report

## 1. Corpus Overview
* **Total Clean Samples**: 301
* **Number of Classes**: 4 (`0 — Control`, `1 — Low Stress`, `2 — Moderate Distress`, `3 — Severe Crisis`)
* **Partitioning Scheme**: Stratified 70% Train / 15% Validation / 15% Test

## 2. Stratified Class Distribution
|                       |   test |   train |   val |   All |
|:----------------------|-------:|--------:|------:|------:|
| 0 — Control           |     13 |      58 |    12 |    83 |
| 1 — Low Stress        |      9 |      40 |     9 |    58 |
| 2 — Moderate Distress |     10 |      49 |    11 |    70 |
| 3 — Severe Crisis     |     14 |      63 |    13 |    90 |
| Total                 |     46 |     210 |    45 |   301 |

## 3. Sequence Length Statistics (Words & Characters)
* **Word Count Mean ± Std**: 18.1 ± 5.0 words
* **Median Word Count**: 17 words
* **Interquartile Range (IQR)**: 14 to 22 words
* **95th Percentile Word Count**: 27 words
* **99th Percentile Word Count**: 30 words
* **Max Word Count**: 31 words
* **Character Count Mean / Max**: 104.5 / 172 chars

> [!TIP]
> **Tokenizer Length Recommendation**:
> Since the 99th percentile of sequence lengths is **30 words** (~40 BPE subword tokens), a `max_seq_length` of **256** or **128** covers over 99.5% of samples with zero token truncation while minimizing GPU memory padding waste.

## 4. Per-Class Length Profiles
- **0 — Control**: Mean = 13.2 words, Median = 13 words, 95th Percentile = 17 words
- **1 — Low Stress**: Mean = 17.2 words, Median = 16 words, 95th Percentile = 25 words
- **2 — Moderate Distress**: Mean = 21.0 words, Median = 19 words, 95th Percentile = 28 words
- **3 — Severe Crisis**: Mean = 21.1 words, Median = 21 words, 95th Percentile = 27 words

## 5. Lexical Diversity & Vocabulary Richness
| Class                 |   Total Tokens |   Unique Vocab |   Type-Token Ratio (TTR) |
|:----------------------|---------------:|---------------:|-------------------------:|
| Overall Corpus        |           5459 |           1244 |                   0.2279 |
| 0 — Control           |           1097 |            507 |                   0.4622 |
| 1 — Low Stress        |            997 |            386 |                   0.3872 |
| 2 — Moderate Distress |           1467 |            395 |                   0.2693 |
| 3 — Severe Crisis     |           1898 |            292 |                   0.1538 |
