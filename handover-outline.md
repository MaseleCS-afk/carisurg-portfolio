# Handover Document — Outline (Interim Draft)

*Full one-pager due at final submission. This outline sketches each of the
six required sections with the content that will fill them.*

## 1. Project Summary (one paragraph)
- What: ML-based ED triage-support tool predicting ESI acuity (1–5) from
  vitals + chief-complaint flags at time of arrival.
- For whom: Mercer General Hospital ED, as a clinician-facing decision
  support aid alongside standard nurse triage — not an autonomous
  triage-maker.
- Built on: Yale EMMLC ED triage dataset (55,121 encounters), Weeks 5–7.

## 2. Final-Model Decision (one sentence + link)
- Verdict sentence (from the Cost-Benefit Memo): *"We ship Random Forest
  by default for its balance of ESI-1 recall, cost, and auditability, and
  keep Gradient Boosting pinned for higher-ceiling deployments."*
- Link out to: `docs/model-selection.md` (full audit trail) and
  `Week7_Cost_Benefit_Memo.docx` (full reasoning).

## 3. How to Run
- `git clone` → `pip install -r requirements.txt` →
  `python scripts/train.py --config config.yaml`
- No manual notebook cell-running required.

## 4. Where the Data Lives
- Path: `data/yaleemmlc_admissionprediction_triage.csv` (git-ignored, not
  committed to the repo).
- Governance status: de-identified but **not** ungoverned — state who may
  access it and that it must not be redistributed (to confirm exact
  access/governance wording with Martina Griffith before final submission).

## 5. Known Limitations (3 honest, specific bullets)
- Single-site (U.S. hospital system) data — Caribbean generalizability
  untested (flagged since the Week 5 feasibility memo).
- ESI-1 recall (0.313) rests on only 16 test patients — read as
  directional, not a precise, stable operating point.
- Demographics (race, ethnicity) excluded from the deployed model by
  design, for fairness — not because they didn't help numerically.

## 6. Who to Ask
- Model questions: [tutor/team contact — confirm before final submission]
- Data questions: [confirm before final submission]
- Clinical questions: Dr. De Freitas (per Week 7 memo addressee) / the ED Board

---
*Still to confirm before final submission: exact named contacts for
Section 6, and the precise data-governance wording for Section 4 — both
flagged above rather than guessed at.*
