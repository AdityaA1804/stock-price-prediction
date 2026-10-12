"""Rule-based next-day forecasts used as benchmarks."""

import numpy as np
import pandas as pd


def baseline_preds(df, dates, window, drift):
    """Next-day close predicted by simple rules for each date in `dates`.

    drift is the average daily return, estimated on training data only.
    """
    close = df["Close"]
    rows = {}
    for d, p in zip(dates, close.index.get_indexer(dates)):
        last_3d = close.iloc[p - 2:p + 1].values
        rows[d] = {"Naive": close.iloc[p],
                   "Naive + drift": close.iloc[p] * (1 + drift),
                   "Linear_3d": np.polyval(np.polyfit([0, 1, 2], last_3d, 1), 3),
                   "Quadratic_3d": np.polyval(np.polyfit([0, 1, 2], last_3d, 2), 3),
                   "Window_mean": close.iloc[p - window + 1:p + 1].mean()}
    return pd.DataFrame.from_dict(rows, orient="index")
