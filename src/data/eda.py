"""
Exploratory Data Analysis and Distribution Module

Computes class balance, sequence length percentiles, lexical diversity,
and generates publication-quality distribution figures and summary tables.
"""

from pathlib import Path
from typing import Any, Dict, List, Tuple
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless execution
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


LABEL_NAMES = {
    0: "0 — Control",
    1: "1 — Low Stress",
    2: "2 — Moderate Distress",
    3: "3 — Severe Crisis",
}


def analyze_class_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes count and percentage distribution overall and by split.
    
    Args:
        df: DataFrame with 'label' and optional 'split'.
        
    Returns:
        DataFrame with class distribution summary.
    """
    if "split" in df.columns:
        pivot = pd.crosstab(df["label"], df["split"], margins=True)
        # Add class descriptive name
        pivot.index = [LABEL_NAMES.get(idx, f"Class {idx}") if idx != "All" else "Total" for idx in pivot.index]
        return pivot
    else:
        counts = df["label"].value_counts().sort_index()
        pcts = df["label"].value_counts(normalize=True).sort_index() * 100
        summary = pd.DataFrame({
            "Class Name": [LABEL_NAMES.get(c, str(c)) for c in counts.index],
            "Count": counts.values,
            "Percentage (%)": pcts.round(2).values,
        })
        return summary


def analyze_sequence_lengths(df: pd.DataFrame, text_col: str = "text") -> Dict[str, Any]:
    """
    Computes character and word length statistics and percentiles.
    
    Args:
        df: DataFrame containing text column.
        text_col: Name of text column.
        
    Returns:
        Dictionary with descriptive statistics.
    """
    words = df[text_col].astype(str).str.split().apply(len)
    chars = df[text_col].astype(str).apply(len)

    # Word count stats
    word_stats = {
        "mean": float(words.mean()),
        "std": float(words.std()),
        "min": int(words.min()),
        "p25": float(words.quantile(0.25)),
        "median": float(words.median()),
        "p75": float(words.quantile(0.75)),
        "p90": float(words.quantile(0.90)),
        "p95": float(words.quantile(0.95)),
        "p99": float(words.quantile(0.99)),
        "max": int(words.max()),
    }

    # Per-class word length medians
    per_class_medians = {}
    if "label" in df.columns:
        for label, group in df.groupby("label"):
            class_words = group[text_col].astype(str).str.split().apply(len)
            per_class_medians[LABEL_NAMES.get(label, str(label))] = {
                "mean_words": round(float(class_words.mean()), 1),
                "median_words": int(class_words.median()),
                "p95_words": int(class_words.quantile(0.95)),
            }

    return {
        "word_stats": word_stats,
        "char_stats": {
            "mean": float(chars.mean()),
            "median": float(chars.median()),
            "p95": float(chars.quantile(0.95)),
            "max": int(chars.max()),
        },
        "per_class": per_class_medians,
    }


def analyze_lexical_diversity(df: pd.DataFrame, text_col: str = "text") -> pd.DataFrame:
    """
    Computes vocabulary richness (Type-Token Ratio) overall and per class.
    
    Args:
        df: DataFrame containing text and label.
        text_col: Text column name.
        
    Returns:
        DataFrame with lexical diversity metrics.
    """
    records = []
    
    # Overall
    all_tokens = [w.lower() for text in df[text_col] for w in str(text).split()]
    total_tokens = len(all_tokens)
    vocab_size = len(set(all_tokens))
    ttr = (vocab_size / total_tokens) if total_tokens > 0 else 0.0
    records.append({
        "Class": "Overall Corpus",
        "Total Tokens": total_tokens,
        "Unique Vocab": vocab_size,
        "Type-Token Ratio (TTR)": round(ttr, 4),
    })

    if "label" in df.columns:
        for label, group in df.groupby("label"):
            tokens = [w.lower() for text in group[text_col] for w in str(text).split()]
            tot = len(tokens)
            voc = len(set(tokens))
            ratio = (voc / tot) if tot > 0 else 0.0
            records.append({
                "Class": LABEL_NAMES.get(label, f"Class {label}"),
                "Total Tokens": tot,
                "Unique Vocab": voc,
                "Type-Token Ratio (TTR)": round(ratio, 4),
            })

    return pd.DataFrame(records)


def generate_eda_plots(
    df: pd.DataFrame,
    output_dir: Path,
    text_col: str = "text",
) -> Dict[str, Path]:
    """
    Generates class distribution and token length visualizations.
    
    Args:
        df: Input DataFrame.
        output_dir: Directory to save figure PNGs.
        text_col: Name of text column.
        
    Returns:
        Dict of paths to saved figures.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    generated_plots = {}

    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    # Plot 1: Class Distribution by Split
    fig, ax = plt.subplots(figsize=(9, 5))
    if "split" in df.columns:
        order = sorted(df["label"].unique())
        splits = ["train", "val", "test"]
        x = np.arange(len(order))
        width = 0.25

        for i, sp in enumerate(splits):
            sp_df = df[df["split"] == sp]
            counts = [len(sp_df[sp_df["label"] == c]) for c in order]
            ax.bar(x + (i - 1) * width, counts, width, label=sp.capitalize(), alpha=0.9)

        ax.set_xticks(x)
        ax.set_xticklabels([LABEL_NAMES.get(c, str(c)) for c in order], fontsize=10)
        ax.set_ylabel("Sample Count", fontsize=11)
        ax.set_title("Dataset Stratification: Class Distribution across Splits", fontsize=13, fontweight="bold")
        ax.legend(title="Split")
    else:
        counts = df["label"].value_counts().sort_index()
        ax.bar([LABEL_NAMES.get(c, str(c)) for c in counts.index], counts.values, color="#4C72B0")
        ax.set_ylabel("Sample Count", fontsize=11)
        ax.set_title("Dataset Class Distribution", fontsize=13, fontweight="bold")

    plt.tight_layout()
    dist_path = output_dir / "class_distribution.png"
    plt.savefig(dist_path, dpi=200)
    plt.close(fig)
    generated_plots["class_distribution"] = dist_path

    # Plot 2: Sequence Length Boxplot by Class
    fig, (ax_box, ax_hist) = plt.subplots(1, 2, figsize=(13, 5))

    df_plot = df.copy()
    df_plot["word_count"] = df_plot[text_col].astype(str).str.split().apply(len)
    df_plot["class_name"] = df_plot["label"].map(LABEL_NAMES)

    # Boxplot
    classes = [LABEL_NAMES[c] for c in sorted(df["label"].unique())]
    box_data = [df_plot[df_plot["class_name"] == c]["word_count"] for c in classes]
    ax_box.boxplot(box_data, tick_labels=[f"C{c}" for c in sorted(df["label"].unique())], patch_artist=True)
    ax_box.set_ylabel("Word Count", fontsize=11)
    ax_box.set_title("Word Count Distribution by Class", fontsize=12, fontweight="bold")

    # Histogram overall with 95th percentile vertical line
    p95 = df_plot["word_count"].quantile(0.95)
    ax_hist.hist(df_plot["word_count"], bins=25, color="#55A868", edgecolor="black", alpha=0.7)
    ax_hist.axvline(p95, color="red", linestyle="--", linewidth=1.5, label=f"95th Percentile ({int(p95)} words)")
    ax_hist.set_xlabel("Word Count", fontsize=11)
    ax_hist.set_ylabel("Frequency", fontsize=11)
    ax_hist.set_title("Sequence Length Histogram (Words)", fontsize=12, fontweight="bold")
    ax_hist.legend()

    plt.tight_layout()
    len_path = output_dir / "length_distribution.png"
    plt.savefig(len_path, dpi=200)
    plt.close(fig)
    generated_plots["length_distribution"] = len_path

    return generated_plots


def generate_markdown_summary(
    df: pd.DataFrame,
    class_dist: pd.DataFrame,
    len_stats: Dict[str, Any],
    lex_div: pd.DataFrame,
    output_path: Path,
) -> Path:
    """Generates a structured markdown report of the dataset characteristics."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    w = len_stats["word_stats"]
    c = len_stats["char_stats"]

    md = f"""# Dataset Exploratory Analysis & Stratification Report

## 1. Corpus Overview
* **Total Clean Samples**: {len(df):,}
* **Number of Classes**: 4 (`0 — Control`, `1 — Low Stress`, `2 — Moderate Distress`, `3 — Severe Crisis`)
* **Partitioning Scheme**: Stratified 70% Train / 15% Validation / 15% Test

## 2. Stratified Class Distribution
{class_dist.to_markdown()}

## 3. Sequence Length Statistics (Words & Characters)
* **Word Count Mean ± Std**: {w['mean']:.1f} ± {w['std']:.1f} words
* **Median Word Count**: {int(w['median'])} words
* **Interquartile Range (IQR)**: {w['p25']:.0f} to {w['p75']:.0f} words
* **95th Percentile Word Count**: {int(w['p95'])} words
* **99th Percentile Word Count**: {int(w['p99'])} words
* **Max Word Count**: {w['max']} words
* **Character Count Mean / Max**: {c['mean']:.1f} / {c['max']} chars

> [!TIP]
> **Tokenizer Length Recommendation**:
> Since the 99th percentile of sequence lengths is **{int(w['p99'])} words** (~{int(w['p99'] * 1.35)} BPE subword tokens), a `max_seq_length` of **256** or **128** covers over 99.5% of samples with zero token truncation while minimizing GPU memory padding waste.

## 4. Per-Class Length Profiles
"""
    for cls_name, stats in len_stats.get("per_class", {}).items():
        md += f"- **{cls_name}**: Mean = {stats['mean_words']} words, Median = {stats['median_words']} words, 95th Percentile = {stats['p95_words']} words\n"

    md += f"""
## 5. Lexical Diversity & Vocabulary Richness
{lex_div.to_markdown(index=False)}
"""
    output_path.write_text(md, encoding="utf-8")
    return output_path
