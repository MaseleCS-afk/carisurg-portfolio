"""
src/utils.py — shared helpers

Small utilities used across modules: loading config.yaml and setting seeds
consistently so "same data + same seed = same metrics" (see Week 8
Tutorial 2, "Verify the refactor").
"""

import random

import numpy as np
import yaml


def load_config(path: str) -> dict:
    """Load config.yaml into a plain dict."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


def set_seed(seed: int) -> None:
    """Fix random_state everywhere it isn't already passed explicitly."""
    random.seed(seed)
    np.random.seed(seed)
