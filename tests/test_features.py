import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.baselines import baseline_preds  # noqa: E402
from src.data_loader import load_data, load_stock  # noqa: E402
from src.evaluate import evaluate  # noqa: E402
from src.features import ENGINEERED, build_features, split_future, train_test  # noqa: E402


@pytest.fixture(scope="module")
def tcs():
    return load_stock(load_data(ROOT / "data" / "raw"), "TCS")


def test_clean_series(tcs):
    assert tcs.index.is_unique and tcs.index.is_monotonic_increasing
    assert tcs.isna().sum().sum() == 0


@pytest.mark.parametrize("window", [3, 5, 10, 20])
def test_feature_count(tcs, window):
    X, Y = build_features(tcs, window)
    assert X.shape[1] == 9 * window + 3
    assert Y.isna().sum() == 1          # only the last day has no target


def test_no_look_ahead(tcs):
    """Features for a date must not change when all later data is removed."""
    X_full, _ = build_features(tcs, 10)
    cut = X_full.index[len(X_full) // 2]
    X_cut, _ = build_features(tcs.loc[:cut], 10)
    assert np.allclose(X_cut.loc[cut], X_full.loc[cut])


def test_engineered_features_are_scale_free(tcs):
    X, _ = build_features(tcs, 10, ENGINEERED)
    assert X.abs().max().max() < 10


def test_chronological_split(tcs):
    X, Y, _ = split_future(*build_features(tcs, 10))
    x_tr, _, x_te, _ = train_test(X, Y, 0.2)
    assert x_tr.index.max() < x_te.index.min()


def test_naive_baseline(tcs):
    dates = tcs.index[20:25]
    preds = baseline_preds(tcs, dates, 10, drift=0.0)
    assert np.allclose(preds["Naive"], tcs["Close"].loc[dates])


def test_direction_metric():
    today = pd.Series([100.0, 100.0, 100.0])
    true = pd.Series([101.0, 99.0, 102.0])
    assert np.isnan(evaluate(today, true, today)["Direction Acc %"])
    assert evaluate(pd.Series([100.5, 100.5, 99.0]), true, today)["Direction Acc %"] == pytest.approx(100 / 3)
