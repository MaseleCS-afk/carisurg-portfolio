# Model Selection — Audit Trail

**DRAFT — interim submission.** Covers every model trained across Week 6
(baselines) and Week 7 (feature-engineered + tuned candidates). Final
model-decision sentence and full reasoning are in the
[Week 7 Cost-Benefit Memo](../Week7_Cost_Benefit_Memo.docx); this table is
the row-by-row history behind that decision, for Martina Griffith's audit.

All models are evaluated on the same held-out 11,025-patient test set
(fixed `random_state=42`, stratified 80/20 split — same test patients for
every row). ESI 1 is the rarest, most urgent class (16 patients in this
test split); its recall is the primary metric (see Week 6 report §4).

## Week 6 — Baselines

| Model | Key hyperparameters | Accuracy | Macro F1 | Weighted F1 | Recall ESI 1 | Train | Infer |
|---|---|---|---|---|---|---|---|
| Dummy (stratified) | — (random-guess floor) | 0.375 | 0.204 | 0.375 | 0.00 | not measured | not measured |
| Logistic Regression | `max_iter=1000` | 0.667 | 0.492 | 0.661 | 0.25 | not measured | not measured |
| Decision Tree | `max_depth=5` | 0.556 | 0.216 | 0.463 | 0.00 | not measured | not measured |

*Week 6 did not record training/inference time — only Week 7's final
benchmark (below) measured the full six axes.*

## Week 7 — Exploration (feature-engineered, pre-tuning)

These rows are the intermediate models trained while exploring feature
engineering and hyperparameters in `Week7_Final_Baseline_and_Complex_Models.ipynb`.
They are superseded by the tuned/final versions below but kept here for
the audit trail, as requested.

| Model | Key hyperparameters | Macro F1 (test) |
|---|---|---|
| Random Forest (untuned) | `n_estimators=300, class_weight=balanced` | 0.390 |
| Random Forest + demographics | same + one-hot ethnicity/race, age, gender | 0.389 |
| Random Forest (RandomizedSearchCV, best CV score) | 8-iter, 3-fold CV search | 0.464 (CV) |

*Random Forest + demographics was **not** carried forward — the extra
features did not improve macro-F1 (0.389 vs 0.390) and including
race/ethnicity as model inputs is fairness-sensitive (Week 7 ethics note).
Demographics stay excluded by default (`include_demographics: false` in
`config.yaml`).*

## Week 7 — Final Benchmark (six-axis, from the Cost-Benefit Memo)

This is the authoritative comparison behind the model decision — same
four models, scored on accuracy, cost, and interpretability together,
not F1 alone.

| Model | Key hyperparameters | Accuracy | Recall ESI 1 | Macro F1 | Train | Infer/patient | Explain |
|---|---|---|---|---|---|---|---|
| Logistic Regression (baseline) | `max_iter=1000` | 0.667 | 0.250 | 0.492 | 1.9 s | 0.002 ms | High |
| **Random Forest (tuned)** ★ | `n_estimators=200, min_samples_leaf=8, max_features=None, max_depth=None, class_weight=balanced` | 0.608 | **0.313** | 0.475 | 199.2 s | 0.052 ms | Medium |
| Gradient Boosting ★ | `max_depth=6, learning_rate=0.1, max_iter=300, class_weight=balanced` | 0.550 | 0.313 | 0.416 | 10.5 s | 0.007 ms | Low |
| Small MLP | `hidden_layer_sizes=(64,32), alpha=1e-3, max_iter=500` | 0.638 | 0.313 | 0.499 | 102.3 s | 0.016 ms | Low |

★ = pinned finalists in `config.yaml` (`final_models: ["random_forest", "gradient_boosting"]`).

**Winner: tuned Random Forest.** It raises ESI-1 recall from 0.250
(baseline) to 0.313 — the largest clinically meaningful gain of any
candidate — while remaining explainable via feature importances (no SHAP
tooling required) and cheap at inference time (0.052 ms/patient, which is
what matters at ED scale, not the one-time 199 s training cost).

Gradient Boosting matches the same ESI-1 recall gain but with lower
accuracy and materially worse interpretability, so it is not recommended
for deployment at this time — it stays pinned as a documented
alternative, not a shipped model. The Small MLP has the best macro-F1
(0.499) but needs SHAP tooling to explain a single prediction and rests
on the same small (16-patient) ESI-1 sample; it is a candidate to
revisit if a larger validation set confirms the edge.

*Caveat carried from Week 6/7: only 16 of 11,025 test patients are truly
ESI-1, so recall figures here are directional, not statistically
precise — a single additional patient shifts recall by ~6 points.*

Full reasoning: [Week 7 Cost-Benefit Memo](../Week7_Cost_Benefit_Memo.docx).
Full code: `Week6_Baseline_Models.ipynb`, `Week7_Final_Baseline_and_Complex_Models.ipynb` (in `notebooks/`).
