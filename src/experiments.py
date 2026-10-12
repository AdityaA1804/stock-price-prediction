"""Experiments that test whether the EDA-driven inputs improve the models.

feature_ablation     raw prices vs returns vs engineered features, and each group removed in turn
window_sensitivity   effect of the input window length
best_per_stock       best ML model for every symbol
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.features import ENGINEERED, FEATURE_SETS
from src.pipeline import run_symbol
from src.visualize import plot_ablation, plot_all_stocks, plot_window_sensitivity


def ablation_sets():
    """Feature sets compared in the ablation: the base sets plus 'all groups minus one'."""
    sets = dict(FEATURE_SETS)
    for group in ENGINEERED:
        sets[f"Engineered - {group}"] = [g for g in ENGINEERED if g != group]
    return sets


def _collect(run, symbol, label_name, label):
    t = run["table"]
    t = t[t["Type"] == "ML"][["Model", "RMSE / Naive", "Direction Acc %"]].copy()
    t.insert(0, label_name, label)
    t.insert(0, "Symbol", symbol)
    return t


def feature_ablation(raw, symbols, cfg, verbose=True):
    """Long table (symbol, feature set, model) and a pivot of mean RMSE / Naive across symbols."""
    models = cfg["experiments"]["models"]
    parts = []
    for name, groups in ablation_sets().items():
        if verbose:
            print(f"  feature set: {name}")
        for symbol in symbols:
            run = run_symbol(raw, symbol, cfg, groups=groups, models=models, save=False, verbose=False)
            parts.append(_collect(run, symbol, "Feature set", name))
    long = pd.concat(parts, ignore_index=True)
    pivot = (long.pivot_table(index="Feature set", columns="Model", values="RMSE / Naive", aggfunc="mean")
                 .reindex(list(ablation_sets()))[models])
    return long, pivot


def window_sensitivity(raw, symbols, cfg, verbose=True):
    """Long table (symbol, window, model) and a pivot of mean RMSE / Naive across symbols."""
    models = cfg["experiments"]["models"]
    parts = []
    for window in cfg["experiments"]["windows"]:
        if verbose:
            print(f"  window: {window}")
        wcfg = {**cfg, "window": window}
        for symbol in symbols:
            run = run_symbol(raw, symbol, wcfg, models=models, save=False, verbose=False)
            parts.append(_collect(run, symbol, "Window", window))
    long = pd.concat(parts, ignore_index=True)
    pivot = long.pivot_table(index="Window", columns="Model", values="RMSE / Naive", aggfunc="mean")[models]
    return long, pivot


def best_per_stock(summary):
    """Best ML model (lowest test RMSE) for each symbol."""
    ml = summary[summary["Type"] == "ML"]
    return (ml.loc[ml.groupby("Symbol")["RMSE"].idxmin(),
                   ["Symbol", "Model", "RMSE", "RMSE / Naive", "Direction Acc %", "Parameters"]]
              .sort_values("RMSE / Naive").reset_index(drop=True))


def run_experiments(raw, symbols, summary, cfg, verbose=True):
    """Run all experiments and save their tables and figures."""
    results_dir, figures_dir = Path(cfg["results_dir"]), Path(cfg["figures_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)

    best = best_per_stock(summary)
    best.to_csv(results_dir / "best_model_per_stock.csv", index=False)
    plt.close(plot_all_stocks(best, figures_dir / "best_model_per_stock.png"))

    if verbose:
        print("\nFeature ablation")
    long, pivot = feature_ablation(raw, symbols, cfg, verbose)
    long.to_csv(results_dir / "feature_ablation.csv", index=False)
    pivot.to_csv(results_dir / "feature_ablation_summary.csv")
    plt.close(plot_ablation(pivot, figures_dir / "feature_ablation.png"))

    if verbose:
        print("\nWindow sensitivity")
    wlong, wpivot = window_sensitivity(raw, symbols, cfg, verbose)
    wlong.to_csv(results_dir / "window_sensitivity.csv", index=False)
    wpivot.to_csv(results_dir / "window_sensitivity_summary.csv")
    plt.close(plot_window_sensitivity(wpivot, figures_dir / "window_sensitivity.png"))

    return {"best": best, "ablation": pivot, "windows": wpivot}
