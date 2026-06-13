# Results

This page summarizes the MIMIC-IV benchmark experiment produced by `ml_pipeline.ipynb`. The model is a benchmark comparator: a sparse one-vs-rest logistic regression trained on KarmaLego temporal-pattern features, with light context features, not the thesis-main model.

## MIMIC-IV Multi-Label Outcome Prediction

### Prediction Task

The task predicts whether each admission will later experience any of six clinical outcomes during the 48-336 h prediction horizon. Features are built only from the first 48 hours of the admission; labels are outcome events occurring after that 48-hour observation window. A separate masked-MSE length-of-stay regression head is fit on RELEASE-discharged patients.

| Item | Value |
|---|---:|
| Cohort size | 57,078 patients/admissions |
| Observation window | First 48h |
| Label window | 48-336 h after admission |
| Outcomes | 6 |
| Pattern source | KarmaLego TIRPs discovered from training data |
| Model | One-vs-rest logistic regression + masked-MSE LoS head |
| Logistic regression setting | `C=0.1`, L2 penalty, balanced class weights |
| Feature/HP selection patients | 14,269 |
| Evaluation pool patients | 42,809 |
| Train patients | 32,106 |
| Test patients | 10,703 |
| Split seed | 42 |
| Confidence intervals | 2,000-resample patient-level bootstrap |

The benchmark is intentionally lightweight and interpretable. It tests whether temporally ordered interval patterns contain useful signal for downstream prediction before comparing against heavier models.

### Outcomes

The six trained outcomes follow the Mediator TAK rules verbatim — two severity tiers per disglycemia plus the pass-through complications:

- `HYPERGLYCEMIA_EVENT` — glucose recurrent-or-severe rule
- `HYPOGLYCEMIA_EVENT`  — glucose recurrent-or-severe rule
- `SEVERE_HYPERGLYCEMIA_EVENT` — single glucose `>= 250`
- `SEVERE_HYPOGLYCEMIA_EVENT`  — single glucose `<= 54`
- `KIDNEY_COMPLICATION_EVENT` — creatinine absolute or ratio rule
- `DEATH_EVENT` — pass-through

`HYPEROSMOLALITY_EVENT` and `CARDIO-VASCULAR_DISORDER_EVENT` were configured but dropped before training (low support after the threshold filter). `KETOACIDOSIS_EVENT` and `ACIDOSIS_EVENT` likewise.

### Pattern Bank and Feature Selection

KarmaLego patterns were first discovered per outcome, then merged into a shared deduplicated pattern bank.

| Step | Count |
|---|---:|
| Shared deduplicated patterns | 16,361 |
| TIRP feature columns kept after support filter | 29,262 |
| Features after chi-squared FDR feature selection (α = 0.05) | 25,597 |
| Cached sparse matrix patients | 57,078 |
| Shared TIRP chunk files | 58 |

### Aggregate Performance (B=2000 patient bootstrap, 95 % CI)

The headline numbers below come from the bootstrap evaluation on the held-out test split (`n = 10,703` patients, seed 42). The model is fit once on `n_train = 32,106` patients after a separate `n_selection = 14,269` feature-and-hyperparameter-selection slice; the bootstrap is applied to the test predictions afterward.

| Metric | Mean | 95 % CI |
|---|---:|---:|
| Micro PR-AUC (support-weighted) | 0.5865 | [0.5740, 0.5989] |
| Macro PR-AUC | 0.4616 | [0.4501, 0.4739] |
| Micro ROC-AUC (support-weighted) | 0.8696 | [0.8644, 0.8750] |
| Macro ROC-AUC | 0.8261 | [0.8177, 0.8348] |
| Micro F1@0.5 (support-weighted) | 0.5472 | [0.5376, 0.5567] |
| Macro F1@0.5 | 0.4588 | [0.4475, 0.4703] |
| LoS MAE (hours, RELEASE-event patients only) | 100.12 | [96.70, 103.49] |

Performance is strongest on the higher-prevalence outcomes (HYPERGLYCEMIA, KIDNEY) and weaker on the rare ones (SEVERE_HYPOGLYCEMIA, HYPOGLYCEMIA), where PR-AUC is most sensitive to prevalence.

### Per-Outcome Performance (B=2000 patient bootstrap, 95 % CI)

| Outcome | n_pos | PR-AUC | ROC-AUC | Best F1 |
|---|---:|---:|---:|---:|
| `HYPERGLYCEMIA_EVENT`        | 2,823 | 0.7763 [0.7618, 0.7902] | 0.8837 [0.8763, 0.8911] | 0.6952 [0.6816, 0.7087] |
| `SEVERE_HYPERGLYCEMIA_EVENT` | 1,358 | 0.5209 [0.4915, 0.5487] | 0.8473 [0.8352, 0.8586] | 0.5320 [0.5111, 0.5531] |
| `KIDNEY_COMPLICATION_EVENT`  |   948 | 0.7035 [0.6734, 0.7317] | 0.9105 [0.8986, 0.9217] | 0.6745 [0.6504, 0.6998] |
| `DEATH_EVENT`                | 1,358 | 0.4775 [0.4484, 0.5063] | 0.8270 [0.8142, 0.8386] | 0.4922 [0.4707, 0.5138] |
| `HYPOGLYCEMIA_EVENT`         |   598 | 0.2106 [0.1802, 0.2419] | 0.7543 [0.7335, 0.7750] | 0.2796 [0.2495, 0.3119] |
| `SEVERE_HYPOGLYCEMIA_EVENT`  |   279 | 0.0805 [0.0622, 0.1033] | 0.7339 [0.7048, 0.7649] | 0.1551 [0.1267, 0.1848]  |

### Linear SHAP Takeaways

The SHAP analysis is a linear-model attribution summary computed from the bootstrap model. For a linear model, the attribution is driven by the selected feature value and the learned coefficient; the reported `mean_abs_shap` is the most reliable importance score. The signed mean SHAP values are often close to zero because they are averaged over centered samples, so the takeaways below emphasize importance and feature identity.

Top selected features show that the benchmark is using clinically plausible temporal structure rather than only demographics:

| Outcome | High-importance examples |
|---|---|
| Hyperglycemia | Low glucose with normal potassium; diabetes diagnosis containing creatinine-on-admission; diabetes diagnosis containing admission event; glucose low singleton. |
| Severe Hyperglycemia | Severe-high glucose singletons; severe-high glucose containing prior moderate-high; admission-anchored severe hyper patterns. |
| Hypoglycemia | Decreasing glucose trend; low glucose with very low M-SHR; repeated low-glucose patterns; normal urea containing normal glucose. |
| Severe Hypoglycemia | Severe-low glucose singletons; severe-low containing recurrent moderate-low; antidiabetic-overlap patterns. |
| Kidney complication | Decreasing urea trend; high M-SHR; normal creatinine; moderately elevated urea; hypertension; decreasing sodium trend. |
| Death | Age at admission is dominant; normal platelet/sodium patterns; lunch/meal-context and admission type; increasing infection-WBC trend; hypertension. |

Important interpretation points:

- The selected model is overwhelmingly pattern-count driven.
- Duration features were generated but almost entirely removed by the chi-squared FDR feature selection, suggesting that occurrence frequency carried more benchmark signal than normalized active duration in this configuration.
- Several high-importance patterns include early evidence of the same event family, such as early hyperglycemia or dysglycemia-related events. This is clinically meaningful for recurrence/progression prediction, but should be discussed explicitly when comparing to models that exclude prior same-family events.
- Demographic/context variables still matter for death, cardiovascular, and kidney outcomes, especially age, admission type, and hypertension. The benchmark therefore combines temporal-abstraction signal with a small amount of baseline risk context.
- Rare outcomes have respectable ROC-AUC but lower PR-AUC. This is expected under class imbalance and makes PR-AUC the more important comparison metric for hypoglycemia and severe hypoglycemia.

### Generated Artifacts

Primary outputs:

```text
data/output/ml_pipeline/lr_bootstrap_evaluation/
```

Key files:

- `point_summary.csv` — single-point headline metrics.
- `metrics_summary.json` — JSON of the same headline + model config (LR params, feature counts, split seed).
- `metrics_by_outcome.csv` — per-outcome point estimates (PR-AUC, ROC-AUC, F1@0.5, best F1).
- `bootstrap_summary_ci.csv` — headline metrics with 95 % bootstrap CI.
- `bootstrap_by_outcome_ci.csv` — per-outcome metrics with 95 % bootstrap CI.
- `bootstrap_summary_iterations.csv` / `bootstrap_by_outcome_iterations.csv` — per-iteration raw bootstrap samples.
- `selected_features.csv` — feature names retained by chi-squared FDR selection.
- `chi2_fdr_feature_selection_report.csv` — feature-selection report.
- `lr_validation_tuning_trials.csv` — light grid-search trace on the selection-validation slice.
- `linear_shap_summary/` — SHAP feature-importance summaries.
- `los_model.joblib` / `los_predictions.csv` — LoS regression model + predictions.

Pattern and matrix caches:

```text
data/output/ml_pipeline/combined_deduped_patterns.csv
data/output/ml_pipeline/pattern_cache_summary.json
data/output/ml_pipeline/filtered_tirp_cache/
```
