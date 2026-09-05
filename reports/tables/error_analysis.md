# Systematic Error Analysis

- **Total Samples Audited**: 30
- **Total Misclassifications**: 15 (50.0%)
- **Severe False Negatives (Missed Crises)**: 8
- **Severe False Positives (False Alarms)**: 0

## Error Taxonomy Distribution

| Error Category | Description | Count | % of Errors |
|---|---|---|---|
| `BORDERLINE_ADJACENT` | Adjacent category boundary uncertainty (|y - y_hat| = 1) | 7 | 46.7% |
| `IMPLICIT_DISTRESS` | Genuine crisis without explicit keywords (numbness, emptiness, despair) | 6 | 40.0% |
| `NEGATION_FAILURE` | Distress keyword detected but negation context ('not suicidal') missed | 2 | 13.3% |

## Priority Audit: Severe Crisis False Negatives (Missed Crises)
> **Safety Critical**: Samples where ground-truth was Severe Crisis, but model under-triaged.

| Sample Index | Text Preview | True Label | Pred Label | Confidence | Category |
|---|---|---|---|---|---|
| 5 | I've written my goodbye letters and given away my dog. I fin... | Severe Crisis | Moderate Distress | 0.2578 | `IMPLICIT_DISTRESS` |
| 8 | I can't do this anymore. Tonight is going to be my last nigh... | Severe Crisis | Control | 0.2576 | `IMPLICIT_DISTRESS` |
| 12 | I have the pills on my table. I don't want to wake up tomorr... | Severe Crisis | Moderate Distress | 0.2583 | `NEGATION_FAILURE` |
| 19 | I have the pills on my table. I don't want to wake up tomorr... | Severe Crisis | Control | 0.2586 | `NEGATION_FAILURE` |
| 20 | Standing on the ledge right now. There's nothing left for me... | Severe Crisis | Control | 0.2588 | `IMPLICIT_DISTRESS` |
| 22 | The pain is too much to bear and there is no other way out f... | Severe Crisis | Moderate Distress | 0.2582 | `IMPLICIT_DISTRESS` |
| 23 | The pain is too much to bear and there is no other way out f... | Severe Crisis | Control | 0.2586 | `IMPLICIT_DISTRESS` |
| 24 | Everything is arranged. I am ready to disappear forever. Nob... | Severe Crisis | Control | 0.2590 | `IMPLICIT_DISTRESS` |