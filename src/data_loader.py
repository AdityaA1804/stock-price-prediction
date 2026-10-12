"""Loading the raw NSE CSV files and extracting one clean daily series per symbol."""

from pathlib import Path

import pandas as pd

COLUMN_MAP = {"No. of Trades": "Trades",
              "Total Traded Quantity": "Volume",
              "% Dly Qt to Traded Qty": "Del_pct",
              "Average Price": "Average",
              "Open Price": "Open",
              "Close Price": "Close",
              "High Price": "High",
              "Low Price": "Low"}
KEEP_COLS = ["Prev Close", "Open", "High", "Low", "Close", "Average", "Volume", "Trades", "Del_pct"]


def load_data(data_dir):
    """Read and concatenate every .csv file in data_dir."""
    files = sorted(Path(data_dir).glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No .csv files found in {Path(data_dir).resolve()}")
    raw = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    raw["Date"] = pd.to_datetime(raw["Date"], format="%d-%b-%y")
    return raw


def list_symbols(raw):
    return sorted(raw.loc[raw["Series"] == "EQ", "Symbol"].unique())


def resolve_symbols(raw, symbols):
    """Accept a list of symbols or the string "all"."""
    if symbols == "all" or symbols == ["all"]:
        return list_symbols(raw)
    return list(symbols)


def load_stock(raw, symbol):
    """Regular-session (EQ) daily data for one symbol, indexed by date.

    Block-deal (BL) rows are dropped: they duplicate the trading day of the EQ row.
    """
    df = raw[(raw["Series"] == "EQ") & (raw["Symbol"] == symbol)].copy()
    if df.empty:
        raise ValueError(f"Symbol {symbol!r} not found. Available: {list_symbols(raw)}")
    df = df.rename(columns=COLUMN_MAP).drop_duplicates("Date").sort_values("Date").set_index("Date")
    return df[KEEP_COLS].astype(float)
