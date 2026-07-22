# Week 7 — Draft Benchmark Table

**CariSurg MedTech Pathways · Interim Submission**
Dataset: Yale EMMLC ED triage export · Same train/test split as Week 6 (`random_state=42`, stratified, 80/20)
Train set: 44,096 patients · Test set: 11,025 patients · True ESI-1 patients in test set: 16

## Six-axis benchmark

| Model | Accuracy | Recall (ESI-1) | Macro-F1 | Train time | Inference time (per patient) | Interpretability |
|---|---|---|---|---|---|---|
| LogReg (baseline) | 0.667 | 0.250 | 0.492 | 1.9 s | 0.002 ms | High |
| Random Forest (tuned) | 0.608 | 0.313 | 0.475 | 199.2 s | 0.052 ms | Medium |
| Gradient Boosting | 0.550 | 0.313 | 0.416 | 10.5 s | 0.007 ms | Low |
| Small MLP | 0.638 | 0.313 | **0.499** | 102.3 s | 0.016 ms | Low |

*Illustrative caveat inherited from Tutorial 3 no longer applies — this is our own run, not a placeholder.*

## What it shows

- **No model dominates on every axis.** The baseline logistic regression is cheapest, fastest, and most explainable, but it catches only 1 in 4 of the truly critical (ESI-1) patients in the test set.
- **All three advanced models lift ESI-1 recall from 0.250 to 0.313** — each correctly flags roughly 1 additional critical patient per 16 in the test set that the baseline would have missed. This is the clinically important number Week 6 warned about.
- **That recall gain is not free.** Overall accuracy and macro-F1 actually *drop* for Gradient Boosting and the tuned Random Forest relative to the baseline — the models are trading some overall correctness for better performance on the rare, high-stakes class. The Small MLP is the exception: it slightly beats the baseline on both macro-F1 (0.499 vs 0.492) and ESI-1 recall, but at ~50x the training cost and materially lower interpretability.
- **Random Forest hyperparameter tuning genuinely helped**: 8-iteration, 3-fold `RandomizedSearchCV` raised the untuned Random Forest's macro-F1 from 0.390 to 0.475 (best params: `n_estimators=200, min_samples_leaf=8, max_features=None`). Tuning cost real compute — the tuned Random Forest is now the slowest model to train (199 s vs 10–102 s for the others).
- **Adding demographic features (age, gender, ethnicity, race) did not improve the Random Forest** (0.389 macro-F1 with demographics vs 0.390 without) — evidence that the model does not need protected-attribute data to perform, relevant to the fairness/governance discussion in Tutorial 4.
- **Top drivers of Random Forest predictions** are vitals and the engineered clinical features: systolic BP, shock index (HR/SBP), diastolic BP, glucose, pulse pressure, heart rate, and temperature — consistent with clinical intuition about what signals acuity.

## Still to determine (for the final Recommendation Memo)

- Whether the Small MLP's small edge on macro-F1 justifies its cost and low interpretability given IT governance constraints (Martina Griffith will ask about inference cost and auditability).
- A full confusion-matrix / error-analysis pass per model (Tutorial 4 territory) — this draft only reports the top-line ESI-1 recall number, not which specific subgroups are still being missed.
- Final model choice and the honest trade-off being accepted, per the Tutorial 1 "pick two" framing (accuracy vs. interpretability vs. compute cost).
