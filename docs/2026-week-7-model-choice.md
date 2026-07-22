# Decision Journal — Week 7 Model Choice

## Context
- The Week 6 logistic regression baseline missed most truly critical (ESI-1) patients (recall 0.250), and Dr. De Freitas / the ED Board flagged this as the key unresolved risk before approving further development.
- The Board asked whether a more sophisticated model is worth the added complexity and compute, rather than simply asking for "the best model."

## Alternatives
- **Gradient Boosting** — matched the ESI-1 recall gain (0.313) but scored lower on overall macro-F1 (0.416) and is the hardest to explain (needs SHAP for any single prediction).
- **Small MLP** — best overall macro-F1 (0.499) and the same ESI-1 recall gain, but the least interpretable option and the second-most expensive to train.
- **Keep the Week 6 baseline unchanged** — cheapest and fully transparent, but leaves the ESI-1 miss rate unaddressed.

## Decision
Adopt the tuned Random Forest (`n_estimators=200, min_samples_leaf=8, max_features=None`) as the ED triage-support model for continued piloting, replacing the Week 6 baseline.

## Reasoning
- It raises ESI-1 recall from 0.250 to 0.313, directly addressing the Board's flagged risk, matching both black-box alternatives on this axis.
- Its feature importances stay directly readable without extra tooling, satisfying Clinical IT's auditability bar in a way Gradient Boosting and the MLP do not.
- Its inference cost (0.052 ms/patient) is negligible for real-time ED use; the higher training cost (199 s) is a one-time or occasional offline cost, not a recurring one.

## Unknowns
- Only 16 true ESI-1 patients existed in the test set, so the recall estimate (5 of 16) carries real statistical uncertainty and needs validation on a larger sample before being trusted operationally.
- Generalizability of a Yale EMMLC-trained model to a Caribbean clinical population remains untested, a concern first raised in the Week 5 feasibility memo.
