"""
src/model.py — train & evaluate

Builds a model from config (name + hyperparameters), and scores it on the
metrics that matter for triage support: accuracy, recall for ESI 1
specifically (the primary metric — see Week 6 report Section 4), macro-F1,
and inference time.
"""

import time

from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

BUILDERS = {
    "logistic_regression": lambda params, seed: make_pipeline(
        StandardScaler(), LogisticRegression(random_state=seed, **params)
    ),
    "random_forest": lambda params, seed: RandomForestClassifier(random_state=seed, n_jobs=-1, **params),
    "gradient_boosting": lambda params, seed: HistGradientBoostingClassifier(random_state=seed, **params),
    "mlp": lambda params, seed: make_pipeline(
        StandardScaler(), MLPClassifier(random_state=seed, **params)
    ),
}


def build_model(name: str, params: dict, seed: int = 42):
    """
    Construct a model by name from config, with a fixed seed. `name` must be
    one of the keys in BUILDERS (see config.yaml `models:` section).
    """
    if name not in BUILDERS:
        raise ValueError(f"Unknown model name '{name}'. Choose from: {list(BUILDERS)}")
    return BUILDERS[name](params, seed)


def evaluate(model, X, y) -> dict:
    """
    Score a fitted model on the six axes that drive the model-selection
    decision: accuracy, recall for ESI 1, macro-F1, and inference time
    per patient. (Training time is measured separately in scripts/train.py,
    around the .fit() call, so it isn't duplicated here.)
    """
    start = time.perf_counter()
    preds = model.predict(X)
    elapsed = time.perf_counter() - start
    infer_ms_per_patient = (elapsed / len(X)) * 1000

    return {
        "accuracy": round(accuracy_score(y, preds), 3),
        "macro_precision": round(precision_score(y, preds, average="macro", zero_division=0), 3),
        "macro_recall": round(recall_score(y, preds, average="macro", zero_division=0), 3),
        "macro_f1": round(f1_score(y, preds, average="macro"), 3),
        "recall_esi1": round(recall_score(y, preds, labels=[1], average=None, zero_division=0)[0], 3),
        "infer_ms_per_patient": round(infer_ms_per_patient, 3),
    }
