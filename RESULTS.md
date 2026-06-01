# Results

This page summarizes the latest MIMIC-IV experiment from `ml_pipeline.ipynb`.

## MIMIC-IV Multi-Label Outcome Prediction

The experiment uses KarmaLego temporal interval patterns as sparse patient-level features for a multi-label prediction task. Features are built from the first 48 hours of each admission, and labels are future events after that 48-hour observation window.

### Setup

| Item | Value |
|---|---:|
| Dataset | MIMIC-IV derived temporal abstraction table |
| Patients | 57,078 |
| Train patients | 42,808 |
| Test patients | 14,270 |
| Labels | 6 |
| Observation window | First 48h |
| Prediction label window | >48h |
| Model | One-vs-rest logistic regression |
| Shared patterns after MVS filter | 29,757 |
| Raw feature columns | 35,702 |
| Minimum vertical support filter | 0.20 |

The model is intentionally a lightweight sparse linear baseline rather than XGBoost, because the TIRP feature space is high-dimensional and sparse.

### Aggregate Metrics

| Metric | Value |
|---|---:|
| Micro PR-AUC | 0.643 |
| Macro PR-AUC | 0.525 |
| Micro ROC-AUC | 0.866 |
| Macro ROC-AUC | 0.832 |
| Micro F1 at 0.5 | 0.597 |
| Macro F1 at 0.5 | 0.517 |

### Per-Outcome Metrics

| Outcome | Test positives | PR-AUC | ROC-AUC | F1 at 0.5 | Best F1 | Best threshold |
|---|---:|---:|---:|---:|---:|---:|
| DISGLYCEMIA_EVENT_Hyperglycemia | 3,760 | 0.744 | 0.870 | 0.681 | 0.683 | 0.444 |
| DISGLYCEMIA_EVENT_Hypoglycemia | 800 | 0.191 | 0.750 | 0.254 | 0.266 | 0.771 |
| HYPEROSMOLALITY_EVENT | 4,228 | 0.701 | 0.819 | 0.646 | 0.646 | 0.504 |
| CARDIO-VASCULAR_DISORDER_EVENT | 293 | 0.334 | 0.901 | 0.405 | 0.428 | 0.690 |
| KIDNEY_COMPLICATION_EVENT | 3,425 | 0.693 | 0.829 | 0.621 | 0.625 | 0.638 |
| DEATH_EVENT | 1,813 | 0.489 | 0.826 | 0.496 | 0.498 | 0.511 |

### Output Files

The generated artifacts are written under:

```text
data/output/ml_pipeline/multilabel_model/
```

Key files:

- `metrics_summary.json`
- `metrics_by_outcome.csv`
- `top_features_by_outcome.csv`
- `test_scores.csv`
- `test_predictions_at_0_5.csv`
- `model.joblib`

The filtered TIRP cache is written once and reused in later runs:

```text
data/output/ml_pipeline/filtered_tirp_cache_mvs_0_20/
```

