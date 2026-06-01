# Results

This page summarizes the MIMIC-IV benchmark experiment produced by `ml_pipeline.ipynb`. The model is a benchmark comparator: a sparse one-vs-rest logistic regression trained on KarmaLego temporal-pattern features, with light context features, not the thesis-main model.

## MIMIC-IV Multi-Label Outcome Prediction

### Prediction Task

The task predicts whether each admission will later experience any of six clinical outcomes. Features are built only from the first 48 hours of the admission; labels are outcome events occurring after that 48-hour observation window.

| Item | Value |
|---|---:|
| Cohort size | 57,078 patients/admissions |
| Observation window | First 48h |
| Label window | >48h after admission |
| Outcomes | 6 |
| Pattern source | KarmaLego TIRPs discovered from training data |
| Model | One-vs-rest logistic regression |
| Logistic regression setting | `C=0.1`, L2 penalty, balanced class weights |
| Feature/HP selection patients | 14,269 |
| Seed-stability evaluation pool | 42,809 |
| Seed-stability train patients per seed | 32,106 |
| Seed-stability test patients per seed | 10,703 |
| Split seeds | 2023, 2024, 2025 |

The benchmark is intentionally lightweight and interpretable. It tests whether temporally ordered interval patterns contain useful signal for downstream prediction before comparing against heavier models.

### Pattern Bank and Feature Selection

KarmaLego patterns were first discovered per outcome, then merged into a shared deduplicated pattern bank and filtered by shared minimum vertical support.

| Step | Count |
|---|---:|
| Deduplicated shared patterns before support filter | 207,439 |
| Shared patterns after MVS >= 0.20 | 29,757 |
| TIRP feature columns after MVS filter | 35,982 |
| Cached sparse matrix shape | 57,078 x 35,982 |
| Cached sparse matrix nonzeros | 518,947,491 |
| Nonzero model columns entering variance selection | 35,684 |
| Selected model columns after 95% cumulative variance filter | 15,928 |

Selected feature composition:

| Feature type | Selected | Not selected |
|---|---:|---:|
| TIRP count features | 15,918 | 1,921 |
| TIRP duration features | 5 | 17,833 |
| Context features | 5 | 2 |

Filtered shared patterns by length:

| Pattern length | Count |
|---|---:|
| 1 | 145 |
| 2 | 4,914 |
| 3 | 15,490 |
| 4 | 9,208 |

Filtered pattern support distribution:

| Statistic | Vertical support |
|---|---:|
| Minimum | 0.200 |
| 25th percentile | 0.216 |
| Median | 0.245 |
| Mean | 0.277 |
| 75th percentile | 0.304 |
| Maximum | 1.000 |

Patterns after filtering were not only outcome-specific. The `source_count` field records in how many outcome-specific discovery runs a pattern appeared before deduplication.

| Outcome-source count | Filtered patterns |
|---|---:|
| 1 | 10,622 |
| 2 | 291 |
| 3 | 294 |
| 4 | 1,235 |
| 5 | 15,106 |
| 6 | 2,209 |

This suggests a large shared temporal physiology vocabulary, plus a smaller set of outcome-specific patterns.

### Per-Outcome Pattern Discovery

| Outcome | Patterns | k=1 | k=2 | k=3 | k=4 | Median vertical support |
|---|---:|---:|---:|---:|---:|---:|
| DISGLYCEMIA_EVENT_Hyperglycemia | 80,511 | 116 | 5,931 | 33,766 | 40,698 | 0.130 |
| DISGLYCEMIA_EVENT_Hypoglycemia | 82,045 | 130 | 6,336 | 34,500 | 41,079 | 0.130 |
| HYPEROSMOLALITY_EVENT | 70,393 | 119 | 5,497 | 29,581 | 35,196 | 0.130 |
| CARDIO-VASCULAR_DISORDER_EVENT | 77,945 | 130 | 5,921 | 32,572 | 39,322 | 0.130 |
| KIDNEY_COMPLICATION_EVENT | 70,880 | 121 | 6,087 | 31,659 | 33,013 | 0.130 |
| DEATH_EVENT | 52,730 | 117 | 4,622 | 21,822 | 26,169 | 0.137 |

Most discovered patterns are length 3-4, which is expected for KarmaLego after Apriori pruning: single events are common, pair patterns provide the bridge, and higher-order temporal combinations dominate the retained feature vocabulary.

### Seed-Stability Aggregate Performance

These are the primary benchmark numbers for the paper because the model is evaluated across three held-out train/test split seeds after a separate feature-selection and hyperparameter-selection slice has already been reserved.

| Metric | Mean +/- SD | Range |
|---|---:|---:|
| Micro PR-AUC | 0.609 +/- 0.002 | 0.608-0.611 |
| Macro PR-AUC | 0.495 +/- 0.005 | 0.489-0.498 |
| Micro ROC-AUC | 0.852 +/- 0.001 | 0.851-0.852 |
| Macro ROC-AUC | 0.810 +/- 0.002 | 0.808-0.811 |
| Micro F1 at 0.5 | 0.573 +/- 0.002 | 0.570-0.575 |
| Macro F1 at 0.5 | 0.486 +/- 0.004 | 0.481-0.488 |

The small standard deviations across seeds indicate that the benchmark is stable under different held-out splits. Performance is strongest for the higher-prevalence outcomes and weaker for rare hypoglycemia and cardiovascular events, where PR-AUC is more sensitive to prevalence.

### Per-Outcome Seed-Stability Performance

| Outcome | Mean test positives | PR-AUC mean +/- SD | ROC-AUC mean +/- SD | F1@0.5 mean +/- SD | Best F1 mean +/- SD |
|---|---:|---:|---:|---:|---:|
| DISGLYCEMIA_EVENT_Hyperglycemia | 2,821 | 0.705 +/- 0.003 | 0.857 +/- 0.001 | 0.662 +/- 0.003 | 0.666 +/- 0.004 |
| DISGLYCEMIA_EVENT_Hypoglycemia | 598 | 0.160 +/- 0.005 | 0.707 +/- 0.006 | 0.227 +/- 0.015 | 0.231 +/- 0.011 |
| HYPEROSMOLALITY_EVENT | 3,173 | 0.681 +/- 0.006 | 0.809 +/- 0.002 | 0.632 +/- 0.003 | 0.634 +/- 0.003 |
| CARDIO-VASCULAR_DISORDER_EVENT | 221 | 0.308 +/- 0.024 | 0.870 +/- 0.006 | 0.328 +/- 0.002 | 0.385 +/- 0.021 |
| KIDNEY_COMPLICATION_EVENT | 2,569 | 0.671 +/- 0.004 | 0.813 +/- 0.004 | 0.605 +/- 0.007 | 0.614 +/- 0.004 |
| DEATH_EVENT | 1,355 | 0.442 +/- 0.002 | 0.803 +/- 0.004 | 0.459 +/- 0.006 | 0.468 +/- 0.007 |

### Earlier Single-Split Performance

The notebook also contains an earlier single split with 42,808 train and 14,270 test patients before the seed-stability protocol was added. It used the same 29,757 shared patterns and a one-vs-rest logistic regression over 35,702 raw model features.

| Metric | Value |
|---|---:|
| Micro PR-AUC | 0.643 |
| Macro PR-AUC | 0.525 |
| Micro ROC-AUC | 0.866 |
| Macro ROC-AUC | 0.832 |
| Micro F1 at 0.5 | 0.597 |
| Macro F1 at 0.5 | 0.517 |

These numbers are useful as a run artifact, but the seed-stability results above should be cited as the main benchmark estimate.

### Linear SHAP Takeaways

The SHAP analysis is a linear-model attribution summary computed from the seed-stability models. For a linear model, the attribution is driven by the selected feature value and the learned coefficient; the reported `mean_abs_shap` is the most reliable importance score. The signed mean SHAP values are often close to zero because they are averaged over centered samples, so the takeaways below emphasize importance and feature identity.

Top selected features show that the benchmark is using clinically plausible temporal structure rather than only demographics:

| Outcome | High-importance examples |
|---|---|
| Hyperglycemia | Low glucose with normal potassium; diabetes diagnosis containing creatinine-on-admission; diabetes diagnosis containing admission event; glucose low singleton. |
| Hypoglycemia | Decreasing glucose trend; low glucose with very low M-SHR; repeated low-glucose patterns; normal urea containing normal glucose. |
| Hyperosmolality | Low glucose containing hyperosmolality event; age at admission; mild hyponatremia; sodium increasing/decreasing trends; potassium or creatinine patterns containing hyperosmolality. |
| Cardiovascular disorder | Urea mildly elevated containing decreasing potassium; normal creatine kinase; creatinine-on-admission and normal creatinine patterns; admission type and hypertension. |
| Kidney complication | Decreasing urea trend; high M-SHR; normal creatinine; moderately elevated urea; hypertension; decreasing sodium trend. |
| Death | Age at admission is dominant; normal platelet/sodium patterns; lunch/meal-context and admission type; increasing infection-WBC trend; hypertension. |

Important interpretation points:

- The selected model is overwhelmingly pattern-count driven: 15,918 of 15,928 selected columns are TIRP count features.
- Duration features were generated but almost entirely removed by variance selection, suggesting that occurrence frequency carried more benchmark signal than normalized active duration in this configuration.
- Several high-importance patterns include early evidence of the same event family, such as early hyperosmolality or dysglycemia-related events. This is clinically meaningful for recurrence/progression prediction, but should be discussed explicitly when comparing to models that exclude prior same-family events.
- Demographic/context variables still matter for death, cardiovascular, and kidney outcomes, especially age, admission type, and hypertension. The benchmark therefore combines temporal-abstraction signal with a small amount of baseline risk context.
- Rare outcomes have respectable ROC-AUC but lower PR-AUC. This is expected under class imbalance and makes PR-AUC the more important comparison metric for hypoglycemia and cardiovascular disorder.

### Generated Artifacts

Primary outputs:

```text
data/output/ml_pipeline/lr_seed_stability/
```

Key files:

- `summary_by_seed.csv`
- `metrics_by_outcome_seed.csv`
- `metrics_by_outcome_seed_aggregate.csv`
- `selected_features.csv`
- `variance_feature_selection_report.csv`
- `lr_validation_tuning_trials.csv`
- `linear_shap_summary/top_linear_shap_features_by_outcome.csv`
- `linear_shap_summary/linear_shap_feature_importance_aggregate.csv`

Pattern and matrix caches:

```text
data/output/ml_pipeline/combined_deduped_patterns.csv
data/output/ml_pipeline/combined_deduped_patterns_mvs_0.20.csv
data/output/ml_pipeline/pattern_cache_summary.json
data/output/ml_pipeline/filtered_tirp_cache_mvs_0_20/
```
