"""Forecast metrics, measured in price units against the actual next-day close."""

import numpy as np


def evaluate(pred, true, today):
    """RMSE, MAE, MAPE and direction accuracy of price forecasts.

    Direction accuracy is undefined (NaN) for a forecast that never predicts a change.
    """
    err = true - pred
    pred_dir, true_dir = np.sign(pred - today), np.sign(true - today)
    dir_acc = np.nan if (pred == today).all() else (pred_dir == true_dir).mean() * 100
    return {"RMSE": np.sqrt((err ** 2).mean()),
            "MAE": err.abs().mean(),
            "MAPE %": (err.abs() / true).mean() * 100,
            "Direction Acc %": dir_acc}
