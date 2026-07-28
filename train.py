"""
scripts/train.py — entry point

One command, top to bottom, no manual cell-running:

    python scripts/train.py --config config.yaml [--model random_forest]

Reads config.yaml, cleans the raw data, engineers features, builds and
trains the pinned model, and prints its evaluation metrics.
"""

import argparse
import time

from sklearn.model_selection import train_test_split

from src.data import clean, load_raw
from src.features import add_clinical_features, select_features
from src.model import build_model, evaluate
from src.utils import load_config, set_seed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--model", default=None, help="Model name from config.yaml `models:`; "
                         "defaults to the first entry in `final_models:`")
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed = cfg["seed"]
    set_seed(seed)

    model_name = args.model or cfg["final_models"][0]

    # 1. Load + clean
    df = clean(load_raw(cfg["data"]["raw_path"]))

    # 2. Select + engineer features
    X, y = select_features(df, include_demographics=cfg.get("include_demographics", False))
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=seed
    )
    X_train = add_clinical_features(X_train)
    X_test = add_clinical_features(X_test)

    # 3. Build + train
    model = build_model(model_name, cfg["models"][model_name], seed)
    start = time.perf_counter()
    model.fit(X_train, y_train)
    train_seconds = round(time.perf_counter() - start, 1)

    # 4. Evaluate
    metrics = evaluate(model, X_test, y_test)
    print(f"Model: {model_name}")
    print(f"Train time: {train_seconds}s")
    for k, v in metrics.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
