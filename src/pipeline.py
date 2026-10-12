"""End-to-end run for one symbol: features -> split -> baselines and tuned models -> metrics."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import TimeSeriesSplit

from src.baselines import baseline_preds
from src.data_loader import load_stock
from src.evaluate import evaluate
from src.features import ENGINEERED, build_features, split_future, train_test
from src.models import format_params, model_specs, tune
from src.visualize import plot_model_comparison, plot_prediction


def safe_name(name):
    return name.replace(" ", "_").replace("+", "plus").replace("/", "_")


def prepare(df, window, test_part, groups=ENGINEERED):
    """Features, target and chronological split for one symbol."""
    X_all, Y_all = build_features(df, window, groups)
    X, Y, X_future = split_future(X_all, Y_all)
    x_train, y_train, x_test, y_test = train_test(X, Y, test_part)
    return {"X": X, "Y": Y, "X_future": X_future,
            "x_train": x_train, "y_train": y_train, "x_test": x_test, "y_test": y_test,
            "close_today": df["Close"].loc[x_test.index],
            "close_next": df["Close"].shift(-1).loc[x_test.index],
            "last_close": df["Close"].iloc[-1]}


def run_symbol(raw, symbol, cfg, groups=ENGINEERED, models=None, save=True, verbose=True):
    """Evaluate all baselines and models for one symbol.

    Returns a dict with the ranked results table, test-set price predictions, the fitted
    grid searches and the data split. With save=True, writes the table and plots to
    <results_dir>/<symbol>/.
    """
    df = load_stock(raw, symbol)
    d = prepare(df, cfg["window"], cfg["test_part"], groups)
    tscv = TimeSeriesSplit(n_splits=cfg["cv_splits"])
    min_fold_rows = min(len(tr) for tr, _ in tscv.split(d["x_train"]))

    rows, preds, searches = [], {}, {}

    def add_row(name, kind, pred_price, forecast, cv_rmse=np.nan, params="-"):
        preds[name] = pred_price
        rows.append({"Symbol": symbol, "Model": name, "Type": kind,
                     **evaluate(pred_price, d["close_next"], d["close_today"]),
                     "CV RMSE (return)": cv_rmse,
                     "Next-day forecast": forecast,
                     "Parameters": params})

    # Baselines: drift from training data for the test set, from all data for the forecast
    base_test = baseline_preds(df, d["x_test"].index, cfg["window"], d["y_train"].mean())
    base_future = baseline_preds(df, d["X_future"].index, cfg["window"], d["Y"].mean())
    for name in base_test.columns:
        add_row(name, "Baseline", base_test[name], base_future[name].iloc[0])

    # ML models: tune on training folds, score on test, refit on all data for the forecast
    include = models if models is not None else cfg["models"]
    for name, (estimator, grid) in model_specs(min_fold_rows, cfg["random_state"], include).items():
        if verbose:
            print(f"  {name:<18}", end="", flush=True)
        gs = tune(estimator, grid, d["x_train"], d["y_train"], tscv)
        searches[name] = gs
        pred_price = d["close_today"] * (1 + gs.best_estimator_.predict(d["x_test"]))
        final = clone(gs.best_estimator_).fit(d["X"], d["Y"])
        forecast = d["last_close"] * (1 + final.predict(d["X_future"])[0])
        add_row(name, "ML", pred_price, forecast, -gs.best_score_, format_params(gs.best_params_))
        if verbose:
            print(f"test RMSE {rows[-1]['RMSE']:.2f}  [{rows[-1]['Parameters']}]")

    table = pd.DataFrame(rows)
    table["RMSE / Naive"] = table["RMSE"] / table.loc[table["Model"] == "Naive", "RMSE"].iloc[0]
    table["Last close"] = d["last_close"]
    table["Forecast for"] = f"next trading day after {d['X_future'].index[0]:%d-%b-%Y}"
    table = table.sort_values("RMSE").reset_index(drop=True)
    table.insert(0, "Rank", range(1, len(table) + 1))
    predictions = pd.DataFrame(preds)

    if save:
        save_symbol_outputs(table, predictions, d, symbol, cfg["results_dir"])
    return {"table": table, "predictions": predictions, "searches": searches, "data": d, "df": df}


def save_symbol_outputs(table, predictions, d, symbol, results_dir):
    out_dir = Path(results_dir) / symbol
    period = f"{d['x_test'].index.min():%d-%b-%Y} to {d['x_test'].index.max():%d-%b-%Y}"
    for name in predictions.columns:
        fig = plot_prediction(predictions[name], d["close_today"], d["close_next"],
                              f"{symbol} - {name} on test set ({period})",
                              out_dir / "plots" / f"{safe_name(name)}.png")
        plt.close(fig)
    plt.close(plot_model_comparison(table, symbol, out_dir / "comparison.png"))
    table.to_csv(out_dir / "results.csv", index=False)
