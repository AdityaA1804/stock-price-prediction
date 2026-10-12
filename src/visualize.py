"""All plots. Each function returns the figure and saves it when `path` is given."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _finish(fig, path):
    fig.tight_layout()
    if path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=120, bbox_inches="tight")
    return fig


# ---- EDA ----
def plot_price(df, symbol, window, path=None):
    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.plot(df.index, df["Close"], label="Close")
    ax.plot(df.index, df["Close"].rolling(window).mean(), label=f"{window}-day moving average")
    ax.set_title(f"{symbol} - close price")
    ax.set_ylabel("Close (₹)")
    ax.legend()
    return _finish(fig, path)


def plot_return_distribution(df, symbol, path=None):
    ret = df["Close"].pct_change().dropna() * 100
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].hist(ret, bins=80, color="tab:blue", alpha=0.8)
    ax[0].set_title(f"{symbol} - daily returns (%)")
    ax[0].set_xlabel("Return (%)")
    ax[1].plot(ret.index, ret, lw=0.8)
    ax[1].set_title("Daily returns over time")
    ax[1].set_ylabel("Return (%)")
    return _finish(fig, path)


def plot_volume_delivery(df, symbol, window, path=None):
    fig, ax = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    ax[0].plot(df.index, df["Volume"] / 1e6, lw=0.9)
    ax[0].set_title(f"{symbol} - volume (millions of shares)")
    ax[1].plot(df.index, np.log(df["Volume"] / df["Volume"].rolling(window).mean()), lw=0.9, color="tab:green")
    ax[1].axhline(0, color="grey", lw=0.8)
    ax[1].set_title(f"log(volume / {window}-day mean) - the transform used as a feature")
    ax[2].plot(df.index, df["Del_pct"].rolling(window).mean(), lw=0.9, color="tab:orange")
    ax[2].set_title(f"Delivery % ({window}-day mean)")
    return _finish(fig, path)


def plot_correlation_heatmap(corr, title, path=None):
    n = len(corr)
    fig, ax = plt.subplots(figsize=(1.1 * n + 2, 0.9 * n + 1.5))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(n))
    ax.set_xticklabels(corr.columns, rotation=40, ha="right")
    ax.set_yticks(range(n))
    ax.set_yticklabels(corr.index)
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax)
    ax.set_title(title)
    return _finish(fig, path)


def plot_autocorrelation(acf, path=None):
    fig, ax = plt.subplots(figsize=(10, 4))
    width = 0.4
    lags = acf.index.values
    ax.bar(lags - width / 2, acf["Close level"], width, label="Close level")
    ax.bar(lags + width / 2, acf["Daily return"], width, label="Daily return")
    ax.axhline(0, color="grey", lw=0.8)
    ax.set_xlabel("Lag (days)")
    ax.set_ylabel("Autocorrelation")
    ax.set_title("Autocorrelation: price level vs daily return")
    ax.legend()
    return _finish(fig, path)


def plot_split(df, train_index, test_index, symbol, path=None):
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(train_index, df["Close"].loc[train_index], label="Train")
    ax.plot(test_index, df["Close"].loc[test_index], label="Test")
    ax.set_title(f"{symbol} - chronological train / test split")
    ax.set_ylabel("Close (₹)")
    ax.legend()
    return _finish(fig, path)


# ---- Models ----
def plot_prediction(pred, naive, true, title, path=None):
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(true.index, true, color="black", lw=1.6, label="Actual")
    ax.plot(naive.index, naive, color="grey", ls="--", lw=1, alpha=0.8, label="Naive")
    ax.plot(pred.index, pred, color="tab:blue", lw=1.3, label="Predicted")
    ax.set_title(title)
    ax.set_xlabel("Forecast date (prediction is for the next trading day)")
    ax.set_ylabel("Close (₹)")
    ax.legend(loc="upper left")
    fig.autofmt_xdate()
    return _finish(fig, path)


def plot_predictions_grid(pred_df, true, title, ncols=2, path=None):
    names = list(pred_df.columns)
    nrows = int(np.ceil(len(names) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 3.5 * nrows),
                             sharex=True, sharey=True, squeeze=False)
    for ax, name in zip(axes.flat, names):
        rmse = np.sqrt(((true - pred_df[name]) ** 2).mean())
        ax.plot(true.index, true, color="black", lw=1.4, label="Actual")
        ax.plot(pred_df.index, pred_df[name], lw=1.1, alpha=0.85, label="Predicted")
        ax.set_title(f"{name}  (RMSE {rmse:.2f})")
        ax.legend(loc="upper left", fontsize=8)
    for ax in axes.flat[len(names):]:
        ax.set_visible(False)
    fig.suptitle(title, fontsize=14)
    fig.autofmt_xdate()
    return _finish(fig, path)


def plot_model_comparison(table, symbol, path=None):
    t = table.sort_values("RMSE / Naive", ascending=False)
    colors = ["tab:grey" if kind == "Baseline" else "tab:blue" for kind in t["Type"]]
    fig, ax = plt.subplots(figsize=(9, 0.4 * len(t) + 1.5))
    ax.barh(t["Model"], t["RMSE / Naive"], color=colors)
    ax.axvline(1, color="red", ls="--", lw=1)
    ax.set_xlim(max(t["RMSE / Naive"].min() * 0.95, 0), min(t["RMSE / Naive"].max() * 1.02, 3))
    ax.set_title(f"{symbol} - test RMSE relative to Naive (grey = baseline)")
    return _finish(fig, path)


def plot_feature_importance(importance, title, top=15, path=None):
    imp = importance.nlargest(top)
    fig, ax = plt.subplots(figsize=(8, 0.35 * len(imp) + 1.5))
    ax.barh(imp.index[::-1], imp.values[::-1])
    ax.set_xlabel("Importance")
    ax.set_title(title)
    return _finish(fig, path)


def plot_cv_curve(x, scores, xlabel, title, log_x=False, path=None):
    fig, ax = plt.subplots(figsize=(8, 4))
    for label, y in scores.items():
        ax.plot(x, y, marker="o", label=label)
    if log_x:
        ax.set_xscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("CV RMSE (return)")
    ax.set_title(title)
    ax.legend()
    return _finish(fig, path)


# ---- Experiments ----
def plot_ablation(pivot, path=None):
    """pivot: rows = feature set, columns = model, values = mean RMSE / Naive."""
    fig, ax = plt.subplots(figsize=(11, 0.25 * len(pivot) * len(pivot.columns) + 2))
    n_models = len(pivot.columns)
    height = 0.8 / n_models
    y = np.arange(len(pivot))
    for i, model in enumerate(pivot.columns):
        ax.barh(y + i * height, pivot[model], height, label=model)
    ax.set_yticks(y + height * (n_models - 1) / 2)
    ax.set_yticklabels(pivot.index)
    ax.invert_yaxis()
    ax.axvline(1, color="red", ls="--", lw=1, label="Naive")
    lo, hi = min(pivot.min().min(), 1), max(pivot.max().max(), 1)
    pad = (hi - lo) * 0.15 or 0.01
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_xlabel("Test RMSE / Naive (lower is better)")
    ax.set_title("Feature ablation - effect of each input set")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1))
    return _finish(fig, path)


def plot_window_sensitivity(pivot, path=None):
    """pivot: rows = window, columns = model, values = mean RMSE / Naive."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for model in pivot.columns:
        ax.plot(pivot.index, pivot[model], marker="o", label=model)
    ax.axhline(1, color="red", ls="--", lw=1, label="Naive")
    ax.set_xlabel("Window (days)")
    ax.set_ylabel("Test RMSE / Naive")
    ax.set_title("Sensitivity to the input window length")
    ax.legend()
    return _finish(fig, path)


def plot_all_stocks(best, path=None):
    """best: one row per symbol with columns 'Symbol', 'Model', 'RMSE / Naive'."""
    b = best.sort_values("RMSE / Naive")
    colors = ["tab:green" if v < 1 else "tab:red" for v in b["RMSE / Naive"]]
    fig, ax = plt.subplots(figsize=(10, 0.45 * len(b) + 1.5))
    ax.barh(b["Symbol"] + "  (" + b["Model"] + ")", b["RMSE / Naive"], color=colors)
    ax.axvline(1, color="black", ls="--", lw=1)
    ax.invert_yaxis()
    ax.set_xlabel("Best ML model: test RMSE / Naive (green = beats naive)")
    ax.set_title("Best model per stock")
    return _finish(fig, path)
