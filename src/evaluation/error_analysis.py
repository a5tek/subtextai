"""
Systematic Error Analysis and Failure Mode Categorization

Identifies, prioritizes, and classifies prediction discrepancies across the 4 distress severity levels:
  Control (0) < Low Stress (1) < Moderate Distress (2) < Severe Crisis (3)

Failure Mode Taxonomy:
  1. FIGURATIVE_HYPERBOLE: Colloquial exaggerations ('killing me', 'literally dying') misclassified as distress.
  2. NEGATION_FAILURE: Anchoring on distress keywords while missing negation scope ('not suicidal anymore').
  3. IMPLICIT_DISTRESS: Severe distress expressed through subtle feelings (numbness, emptiness) without crisis words.
  4. LEXICAL_OVERFIT: Neutral discussion of crisis topics (news, research, hotlines) triggering false alarms.
  5. DARK_HUMOR_MASK: Genuine distress or suicidal ideation masked by laughter or self-deprecating jokes ('lol', 'classic me').
  6. AMBIGUOUS_INTENT_FALSE_ALARM: Colloquial frustration ('I'm done', 'can't take this') misinterpreted as acute crisis without considering context windows.
  7. BORDERLINE_ADJACENT: Subtle boundary disagreements (|y - y_hat| = 1) between adjacent severity levels.
  8. CATASTROPHIC_ERROR: High-risk severe confusion (|y - y_hat| >= 2).
"""

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd


# Linguistic pattern registries
HYPERBOLE_PATTERNS = [
    r"\bkilling me\b",
    r"\bliterally dying\b",
    r"\bdying of\b",
    r"\bstarving to death\b",
    r"\bdead on my feet\b",
    r"\bbored to death\b",
    r"\bhead is exploding\b",
]

NEGATION_PATTERNS = [
    r"\bnot\b",
    r"\bno longer\b",
    r"\bnever\b",
    r"\bstopped\b",
    r"\bdon't\b",
    r"\bdid not\b",
    r"\bwon't\b",
    r"\bwithout\b",
]

CRISIS_KEYWORDS = [
    # Explicit suicide / lethal self-harm
    r"\bkill myself\b",
    r"\bsuicide\b",
    r"\bsuicidal\b",
    r"\bend it all\b",
    r"\bend my life\b",
    r"\bwant to die\b",
    r"\btake my life\b",
    r"\bgoodbye world\b",
    r"\boverdose\b",
    r"\bhanging\b",
    r"\bhang myself\b",
    r"\bslit my wrists\b",
    r"\bcut my wrists\b",
    r"\bswallow pills\b",
    r"\blethal dose\b",
    r"\bjump off a bridge\b",
    r"\bjump off the bridge\b",
    # Negated desire to live / exist
    r"\bnot want to be alive\b",
    r"\bdo not want to be alive\b",
    r"\bdon'?t want to be alive\b",
    r"\bnot want to live\b",
    r"\bdo not want to live\b",
    r"\bdon'?t want to live\b",
    r"\bnot wanting to live\b",
    r"\bnot wanting to be alive\b",
    r"\bnot want to exist\b",
    r"\bdo not want to exist\b",
    r"\bdon'?t want to exist\b",
    r"\btired of living\b",
    r"\btired of being alive\b",
    r"\bdone with living\b",
    r"\bdone with life\b",
    r"\bstop living\b",
    r"\bno reason to live\b",
    r"\bno point in living\b",
    r"\bno point living\b",
    r"\bno will to live\b",
    r"\bno desire to live\b",
    r"\blost the will to live\b",
    r"\blost all will to live\b",
    # Wish to be dead
    r"\bwish i was dead\b",
    r"\bwish i were dead\b",
    r"\brather be dead\b",
    r"\bbetter off dead\b",
    r"\bbetter off without me\b",
    r"\beveryone would be better off without me\b",
    # Finality / permanent disappearance
    r"\bdisappear forever\b",
    r"\bsleep forever\b",
    r"\bnever wake up\b",
    r"\bdon'?t want to wake up\b",
    r"\bdo not want to wake up\b",
]

DARK_HUMOR_PATTERNS = [
    r"\blol\b",
    r"\blmao\b",
    r"\bclassic me\b",
    r"\bhaha\b",
    r"\bjoking\b",
    r"\bjk\b",
    r"\bjust laughing through\b",
    r"\bdead inside lol\b",
]

AMBIGUOUS_INTENT_PATTERNS = [
    r"\bi'm done\b",
    r"\bi am done\b",
    r"\bcan't take this\b",
    r"\bcannot take this\b",
    r"\bquitting forever\b",
    r"\blogging off\b",
]


def classify_error_taxonomy(text: str, y_true: int, y_pred: int) -> str:
    """
    Categorizes the linguistic or structural reason for a prediction failure.
    """
    text_lower = text.lower()
    step_diff = abs(y_true - y_pred)

    # 1. Check for dark humor masking (genuine distress masked by laughter/jokes)
    if y_true >= 2 and y_pred <= 1:
        if any(re.search(pat, text_lower) for pat in DARK_HUMOR_PATTERNS):
            return "DARK_HUMOR_MASK"

    # 2. Check for ambiguous intent false alarm (colloquial frustration misinterpreted)
    if y_true <= 1 and y_pred >= 2:
        if any(re.search(pat, text_lower) for pat in AMBIGUOUS_INTENT_PATTERNS):
            return "AMBIGUOUS_INTENT_FALSE_ALARM"

    # 3. Check for colloquial hyperbole false positive
    if y_true in [0, 1] and y_pred >= 2:
        for pat in HYPERBOLE_PATTERNS:
            if re.search(pat, text_lower):
                return "FIGURATIVE_HYPERBOLE"

    # 4. Check for negation scope failure
    if any(re.search(n_pat, text_lower) for n_pat in NEGATION_PATTERNS):
        if any(re.search(c_pat, text_lower) for c_pat in CRISIS_KEYWORDS):
            return "NEGATION_FAILURE"

    # 5. Check for implicit distress false negative (Severe true label missed)
    if y_true == 3 and y_pred < 3:
        has_explicit_keyword = any(re.search(c_pat, text_lower) for c_pat in CRISIS_KEYWORDS)
        if not has_explicit_keyword:
            return "IMPLICIT_DISTRESS"

    # 6. Check for high-risk catastrophic errors
    if step_diff >= 2:
        return "CATASTROPHIC_ERROR"

    # 7. Borderline adjacent error (|y - y_hat| == 1)
    if step_diff == 1:
        return "BORDERLINE_ADJACENT"

    return "OTHER_DISCREPANCY"


class DistressErrorAnalyzer:
    """
    Audits model predictions, isolates failure cases, and aggregates error statistics.
    """

    def __init__(
        self,
        texts: List[str],
        y_true: Union[List[int], np.ndarray],
        y_pred: Union[List[int], np.ndarray],
        y_prob: Optional[np.ndarray] = None,
        class_names: Optional[List[str]] = None,
    ):
        self.texts = list(texts)
        self.y_true = np.asarray(y_true, dtype=np.int64)
        self.y_pred = np.asarray(y_pred, dtype=np.int64)
        self.y_prob = y_prob
        self.class_names = class_names or ["Control", "Low Stress", "Moderate Distress", "Severe Crisis"]

        self.df = self._build_error_dataframe()

    def _build_error_dataframe(self) -> pd.DataFrame:
        records = []
        for i in range(len(self.texts)):
            true_cls = self.y_true[i]
            pred_cls = self.y_pred[i]
            is_error = (true_cls != pred_cls)
            conf = float(self.y_prob[i, pred_cls]) if self.y_prob is not None else 1.0

            tax = classify_error_taxonomy(self.texts[i], true_cls, pred_cls) if is_error else "CORRECT"

            records.append({
                "sample_index": i,
                "text": self.texts[i],
                "y_true": true_cls,
                "y_pred": pred_cls,
                "true_label_name": self.class_names[true_cls],
                "pred_label_name": self.class_names[pred_cls],
                "is_error": is_error,
                "severity_step_diff": int(abs(true_cls - pred_cls)),
                "predicted_confidence": round(conf, 4),
                "error_category": tax,
            })
        return pd.DataFrame(records)

    def get_errors(self) -> pd.DataFrame:
        """Returns subset containing only prediction failures."""
        return self.df[self.df["is_error"]].copy()

    def get_severe_false_negatives(self) -> pd.DataFrame:
        """Isolates high-risk Severe Crisis samples that the model failed to detect."""
        severe_idx = len(self.class_names) - 1
        return self.df[(self.df["y_true"] == severe_idx) & (self.df["y_pred"] != severe_idx)].copy()

    def get_severe_false_positives(self) -> pd.DataFrame:
        """Isolates non-crisis posts that the model erroneously escalated to Severe Crisis."""
        severe_idx = len(self.class_names) - 1
        return self.df[(self.df["y_true"] != severe_idx) & (self.df["y_pred"] == severe_idx)].copy()

    def summarize_error_taxonomy(self) -> Dict[str, int]:
        """Counts error frequencies by failure mode category."""
        errors_df = self.get_errors()
        if len(errors_df) == 0:
            return {}
        counts = errors_df["error_category"].value_counts().to_dict()
        return {str(k): int(v) for k, v in counts.items()}

    def export_error_report(self, output_dir: Union[str, Path], report_title: str = "Error Analysis Report") -> Path:
        """
        Exports structured error reports:
          - error_analysis.md: Human-readable markdown audit
          - errors.csv: Granular sample-level table for research inspection
        """
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        errors_df = self.get_errors()
        total_samples = len(self.df)
        total_errors = len(errors_df)
        error_rate = total_errors / max(1, total_samples)

        sev_fn = self.get_severe_false_negatives()
        sev_fp = self.get_severe_false_positives()
        tax_counts = self.summarize_error_taxonomy()

        lines = [
            f"# {report_title}",
            "",
            f"- **Total Samples Audited**: {total_samples}",
            f"- **Total Misclassifications**: {total_errors} ({error_rate * 100:.1f}%)",
            f"- **Severe False Negatives (Missed Crises)**: {len(sev_fn)}",
            f"- **Severe False Positives (False Alarms)**: {len(sev_fp)}",
            "",
            "## Error Taxonomy Distribution",
            "",
            "| Error Category | Description | Count | % of Errors |",
            "|---|---|---|---|",
        ]

        descriptions = {
            "FIGURATIVE_HYPERBOLE": "Colloquial idioms ('killing me', 'literally dying') mistaken for real crisis",
            "DARK_HUMOR_MASK": "Surface laughter ('lol', 'classic me', 'haha') masking authentic severe distress",
            "AMBIGUOUS_INTENT_FALSE_ALARM": "Phrases like 'I'm done' or 'can't take this' misinterpreted without wider context",
            "NEGATION_FAILURE": "Distress keyword detected but negation context ('not suicidal') missed",
            "IMPLICIT_DISTRESS": "Genuine crisis without explicit keywords (numbness, emptiness, despair)",
            "BORDERLINE_ADJACENT": "Adjacent category boundary uncertainty (|y - y_hat| = 1)",
            "CATASTROPHIC_ERROR": "Multi-step severity mismatch (|y - y_hat| >= 2)",
            "OTHER_DISCREPANCY": "General classification error",
        }

        for cat, cnt in tax_counts.items():
            desc = descriptions.get(cat, "Miscellaneous")
            pct = (cnt / total_errors) * 100 if total_errors > 0 else 0.0
            lines.append(f"| `{cat}` | {desc} | {cnt} | {pct:.1f}% |")

        if len(sev_fn) > 0:
            lines.extend([
                "",
                "## Priority Audit: Severe Crisis False Negatives (Missed Crises)",
                "> **Safety Critical**: Samples where ground-truth was Severe Crisis, but model under-triaged.",
                "",
                "| Sample Index | Text Preview | True Label | Pred Label | Confidence | Category |",
                "|---|---|---|---|---|---|",
            ])
            for _, r in sev_fn.iterrows():
                preview = (r['text'][:60] + "...") if len(r['text']) > 60 else r['text']
                preview = preview.replace("|", "\\|")
                lines.append(
                    f"| {r['sample_index']} | {preview} | {r['true_label_name']} | {r['pred_label_name']} | {r['predicted_confidence']:.4f} | `{r['error_category']}` |"
                )

        if len(sev_fp) > 0:
            lines.extend([
                "",
                "## Priority Audit: Severe Crisis False Positives (False Alarms)",
                "> **Resource Burden**: Non-crisis samples escalated to highest alert level.",
                "",
                "| Sample Index | Text Preview | True Label | Pred Label | Confidence | Category |",
                "|---|---|---|---|---|---|",
            ])
            for _, r in sev_fp.iterrows():
                preview = (r['text'][:60] + "...") if len(r['text']) > 60 else r['text']
                preview = preview.replace("|", "\\|")
                lines.append(
                    f"| {r['sample_index']} | {preview} | {r['true_label_name']} | {r['pred_label_name']} | {r['predicted_confidence']:.4f} | `{r['error_category']}` |"
                )

        md_content = "\n".join(lines)
        report_md_path = out_path / "error_analysis.md"
        with open(report_md_path, "w", encoding="utf-8") as f:
            f.write(md_content)

        errors_csv_path = out_path / "errors.csv"
        errors_df.to_csv(errors_csv_path, index=False)

        return report_md_path
