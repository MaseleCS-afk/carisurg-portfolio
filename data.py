"""
src/data.py — load & clean

Everything that turns the raw Yale EMMLC triage CSV into a modelling-ready
table. This is the Week 5/6/7 cleaning logic, extracted from the notebooks
into two testable functions: load_raw() and clean().

No feature engineering here (see src/features.py) — this module's only job
is: raw export in, clean modelling table out.
"""

import numpy as np
import pandas as pd

# Vital-sign columns measured at the front door.
VITALS = [
    "triage_vital_hr",
    "triage_vital_sbp",
    "triage_vital_dbp",
    "triage_vital_rr",
    "triage_vital_o2",
    "triage_vital_temp",
    "triage_glucose",
]


def load_raw(path: str) -> pd.DataFrame:
    """Read the raw triage CSV export into a DataFrame."""
    df_raw = pd.read_csv(path)
    return df_raw


def clean(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Turn the raw, messy export into a modelling-ready table.

    Steps (same cleaning as Week 5 / Week 6 / Week 7, now as one function):
      1. Drop stray index columns (e.g. "Unnamed: 0").
      2. Coerce vitals to numeric; unparseable text becomes NaN.
      3. Drop rows where esi is missing or outside 1-5 — a row with no
         valid triage label cannot teach a triage model.
      4. Blank out physically impossible vitals (e.g. temp < 90 or > 110,
         o2 > 100) so they don't poison the model.
      5. Encode gender to 0/1 (handles odd casings like "m" / "MALE").
      6. Fill remaining missing numeric values with the column median.
    """
    df = df_raw.copy()

    # 0) drop any stray index column pandas adds — it is not real data
    df = df.drop(columns=[c for c in df.columns if c.startswith("Unnamed")], errors="ignore")

    # 1) force the vitals to be NUMBERS; unparseable text becomes NaN
    for col in VITALS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # 2) the ESI label must be 1-5. Drop rows where it is missing or out of range.
    df["esi"] = pd.to_numeric(df["esi"], errors="coerce")
    df = df[df["esi"].isin([1, 2, 3, 4, 5])].copy()

    # 3) blank out physically impossible vitals so they don't poison the model
    df.loc[(df["triage_vital_temp"] < 90) | (df["triage_vital_temp"] > 110), "triage_vital_temp"] = np.nan
    df.loc[df["triage_vital_o2"] > 100, "triage_vital_o2"] = np.nan

    # 4) encode gender to 0/1
    df["gender"] = (
        df["gender"].astype(str).str.strip().str.lower().map({"male": 0, "m": 0, "female": 1, "f": 1})
    )

    # 5) fill remaining missing numeric values with the column median
    for col in VITALS + ["age", "gender"]:
        df[col] = df[col].fillna(df[col].median())

    df["esi"] = df["esi"].astype(int)
    return df
