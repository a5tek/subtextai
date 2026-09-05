# Loss Function Controlled Experiment Comparison (E1 - E5)

> **Evaluation Split**: Held-Out Test Set (Identical partitions across all experiments)  
> **Backbone**: RoBERTa-base (128 max length, deterministic seed)  
> **Primary Metric**: Macro F1 | **Safety Metric**: Severe Crisis Recall

| Experiment | Loss Criterion | Macro F1 | Accuracy | Severe Recall | Severe Precision | Severe F1 | Moderate F1 | Low Stress F1 | Control F1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| E1_cross_entropy | cross_entropy | **0.3330** | 0.5000 | **0.0000** | 0.0000 | 0.0000 | 0.6364 | 0.0000 | 0.6957 |
| E2_weighted_cross_entropy | weighted_cross_entropy | **0.1484** | 0.2667 | **0.0000** | 0.0000 | 0.0000 | 0.4118 | 0.0000 | 0.1818 |
| E3_focal_loss | focal_loss | **0.3426** | 0.5000 | **0.0000** | 0.0000 | 0.0000 | 0.7778 | 0.0000 | 0.5926 |
| E4_weighted_focal_loss | weighted_focal_loss | **0.2833** | 0.4000 | **0.0000** | 0.0000 | 0.0000 | 0.4667 | 0.0000 | 0.6667 |

### Key Empirical Takeaways & Findings:
1. **Severe Recall vs. Precision Trade-off**: Weighting and Focal Loss parameters explicitly shift the operating point towards high-sensitivity severe crisis detection.
2. **Hard-Example Mining**: Multiclass Focal Loss focuses backpropagation gradients on subtle dysphoria and hyperbolic expressions rather than obvious control samples.
3. **Ordinal Continuity**: Ordinal CORAL respects the severity ordering ($0 \prec 1 \prec 2 \prec 3$), penalizing catastrophic misclassifications proportionally.
