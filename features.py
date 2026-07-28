"""
src/features.py — engineer & encode

Turns cleaned columns into model inputs: chooses which columns are fair
game (select_features), builds new clinical features from vitals
(add_clinical_features), and optionally one-hot encodes demographics
(encode_demographics — off by default, see fairness note below).
"""

import pandas as pd

TARGET = "esi"

# Who the patient is (some of these are fairness-sensitive — handle with care).
DEMOGRAPHICS = [
    "age", "gender", "ethnicity", "race", "lang", "religion",
    "maritalstatus", "employstatus", "insurance_status",
]

# Administrative / arrival details.
ADMIN = ["dep_name", "arrivalmode", "arrivalmonth", "arrivalday", "arrivalhour_bin"]

# Outcomes of the visit — known only AFTER triage, so they must never be model inputs.
LEAKAGE = ["disposition", "previousdispo"]


def select_features(df: pd.DataFrame, include_demographics: bool = False):
    """
    Choose which columns the model may use (X) and what it must predict (y).

    Excludes LEAKAGE (post-triage outcomes) and ADMIN always. Excludes
    DEMOGRAPHICS unless include_demographics=True — see the Week 7 ethics
    note: race/ethnicity are encoded to *understand* their effect, but kept
    out of the deployed model by default.
    """
    exclude = LEAKAGE + ADMIN
    if not include_demographics:
        exclude = exclude + DEMOGRAPHICS

    feature_cols = [c for c in df.columns if c != TARGET and c not in exclude]
    X = df[feature_cols].copy()
    y = df[TARGET]
    return X, y


def add_clinical_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    Build new clinical features from existing vitals. Safe to apply to
    train and test separately (row-wise only — no leakage across rows).
    """
    out = data.copy()

    # ratios & combinations
    out["shock_index"] = out["triage_vital_hr"] / out["triage_vital_sbp"]        # HR / SBP (uses BP)
    out["pulse_pressure"] = out["triage_vital_sbp"] - out["triage_vital_dbp"]    # SBP - DBP (uses BP)
    out["spo2_rr_ratio"] = out["triage_vital_o2"] / out["triage_vital_rr"]       # oxygen vs effort (no BP)

    # red-flag flags that do NOT use blood pressure
    out["is_tachypneic"] = (out["triage_vital_rr"] > 20).astype(int)
    out["is_hypoxic"] = (out["triage_vital_o2"] < 92).astype(int)
    out["is_febrile"] = (out["triage_vital_temp"] >= 100.4).astype(int)

    # severity score = how many red flags fire
    out["red_flag_count"] = out[["is_tachypneic", "is_hypoxic", "is_febrile"]].sum(axis=1)

    return out


def encode_demographics(X: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    """
    One-hot encode ethnicity/race and bolt age + gender onto an existing
    feature frame (aligned by row). Off by default in select_features —
    use only with governance sign-off (see Week 7 ethics note).
    """
    demo_1hot = pd.get_dummies(df[["ethnicity", "race"]], prefix=["eth", "race"], dtype=int)
    rows = X.index
    extra = demo_1hot.loc[rows].copy()
    extra["age"] = df.loc[rows, "age"]
    extra["gender"] = df.loc[rows, "gender"]
    return pd.concat([X, extra], axis=1)
