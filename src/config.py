"""Run settings: defaults, overridden by config.yaml and then by command-line options."""

from pathlib import Path

import yaml

DEFAULTS = {
    "data_dir": "data/raw",
    "results_dir": "reports/results",
    "figures_dir": "reports/figures",
    "symbols": ["TCS"],            # list of symbols, or "all"
    "readme_symbol": "TCS",        # stock used for the README model figures
    "window": 10,
    "test_part": 0.2,
    "cv_splits": 5,
    "random_state": 42,
    "models": [],                  # empty = all models
    "experiments": {
        "run": True,
        "models": ["Ridge", "KNN", "XGBoost"],
        "windows": [5, 10, 20],
    },
}


def load_config(path="config.yaml", **overrides):
    cfg = {k: (dict(v) if isinstance(v, dict) else v) for k, v in DEFAULTS.items()}
    if path and Path(path).exists():
        with open(path, encoding="utf-8") as f:
            user = yaml.safe_load(f) or {}
        unknown = set(user) - set(DEFAULTS)
        if unknown:
            raise ValueError(f"Unknown setting(s) in {path}: {sorted(unknown)}")
        for key, value in user.items():
            if key == "experiments" and isinstance(value, dict):
                cfg["experiments"].update(value)
            else:
                cfg[key] = value
    cfg.update({k: v for k, v in overrides.items() if v is not None})

    if cfg["window"] < 3:
        raise ValueError("window must be at least 3")
    if not 0 < cfg["test_part"] < 1:
        raise ValueError("test_part must be between 0 and 1")
    return cfg
