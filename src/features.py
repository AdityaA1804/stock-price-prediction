"""Window features grouped by type, the next-day-return target and the chronological split.

Feature groups (per lag k = 0 .. window-1 unless noted):
    returns       daily return
    price_shape   open / high / low / average (and past closes) relative to today's close
    volume        log volume and log trades relative to their rolling window mean
    delivery      delivery-to-traded ratio
    window_stats  window return, volatility, gaps to short and full moving averages (once)
    raw_levels    raw prices and volume, untransformed (used only to show why EDA-driven
                  transforms are needed)
"""

import numpy as np
import pandas as pd

ENGINEERED = ["returns", "price_shape", "volume", "delivery", "window_stats"]
FEATURE_SETS = {
    "Raw price levels": ["raw_levels"],
    "Returns only": ["returns"],
    "Engineered (all groups)": ENGINEERED,
}


def _group(df, window, name):
    close = df["Close"]
    daily_ret = close / df["Prev Close"] - 1
    f = {}
    if name == "returns":
        for k in range(window):
            f[f"ratio_{k}"] = daily_ret.shift(k)
    elif name == "price_shape":
        for k in range(window):
            for col in ["Open", "High", "Low", "Average"]:
                f[f"{col}_{k}"] = df[col].shift(k) / close - 1
            if k > 0:
                f[f"Close_{k}"] = close.shift(k) / close - 1
    elif name == "volume":
        vol_mean = df["Volume"].rolling(window).mean()
        trd_mean = df["Trades"].rolling(window).mean()
        for k in range(window):
            f[f"vol_{k}"] = np.log(df["Volume"].shift(k) / vol_mean)
            f[f"Trade_{k}"] = np.log(df["Trades"].shift(k) / trd_mean)
    elif name == "delivery":
        for k in range(window):
            f[f"Del_{k}"] = df["Del_pct"].shift(k) / 100
    elif name == "window_stats":
        f["long_close"] = close / df["Prev Close"].shift(window - 1) - 1
        f["long_std"] = daily_ret.rolling(window).std()
        f["window_mean"] = close / close.rolling(window).mean() - 1
        f["half_window_mean"] = close / close.rolling(max(window // 2, 1)).mean() - 1
    elif name == "raw_levels":
        for k in range(window):
            for col in ["Open", "High", "Low", "Close", "Average", "Volume"]:
                f[f"raw_{col}_{k}"] = df[col].shift(k)
    else:
        raise ValueError(f"Unknown feature group {name!r}")
    return f


_PREFIX_GROUP = {"ratio": "returns", "Open": "price_shape", "High": "price_shape", "Low": "price_shape",
                 "Average": "price_shape", "Close": "price_shape", "vol": "volume", "Trade": "volume",
                 "Del": "delivery", "raw": "raw_levels"}


def feature_group(column):
    """Group name of a feature column (e.g. 'vol_3' -> 'volume')."""
    return _PREFIX_GROUP.get(column.split("_")[0], "window_stats")


def build_features(df, window, groups=ENGINEERED):
    """Features for every day (rows with an incomplete window are dropped) and the target.

    Returns X and Y, where Y is the next-day return (NaN on the last day).
    Every feature uses only information available at the close of that day.
    """
    feats = {}
    for name in groups:
        feats.update(_group(df, window, name))
    X = pd.DataFrame(feats, index=df.index)
    Y = df["Close"].shift(-1) / df["Close"] - 1
    valid = X.notna().all(axis=1)
    return X[valid], Y[valid]


def split_future(X, Y):
    """Separate rows with a known target from the last row (used for the next-day forecast)."""
    known = Y.notna()
    return X[known], Y[known], X[~known]


def train_test(X, Y, fraction):
    """Chronological split: the last `fraction` of rows is the test set."""
    n_test = int(len(X) * fraction)
    if n_test < 1 or n_test >= len(X):
        raise ValueError(f"test_part={fraction} leaves no train or test rows for {len(X)} samples")
    return X.iloc[:-n_test], Y.iloc[:-n_test], X.iloc[-n_test:], Y.iloc[-n_test:]
