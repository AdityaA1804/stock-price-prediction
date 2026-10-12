# Data

`raw/Stock_Price_Data.csv` — daily NSE trading data for ten listed companies, 05-Mar-2019 to 03-Mar-2021, in long format (one row per symbol, series and trading day).

**Symbols:** TCS, INFY, WIPRO, TECHM, LTI, LTTS, MINDTREE, MPHASIS, OFSS (IT services) and BEL (defence electronics).

## Columns

| Column | Description | Name used in code |
|---|---|---|
| Symbol | NSE ticker | – |
| Series | `EQ` = regular equity session, `BL` = block deal | – |
| Date | Trading date (`DD-Mon-YY`) | index |
| Prev Close | Previous day's closing price | `Prev Close` |
| Open Price | Opening price | `Open` |
| High Price | Day's high | `High` |
| Low Price | Day's low | `Low` |
| Last Price | Last traded price | not used |
| Close Price | Official closing price (prediction target) | `Close` |
| Average Price | Volume-weighted average price (VWAP) | `Average` |
| Total Traded Quantity | Shares traded | `Volume` |
| Turnover | Traded value (₹) | not used |
| No. of Trades | Number of trades | `Trades` |
| Deliverable Qty | Shares taken into delivery | not used |
| % Dly Qt to Traded Qty | Delivery as % of traded quantity | `Del_pct` |

## Cleaning

- Only `EQ` rows are kept. `BL` (block-deal) rows appear on a few dates and duplicate the trading day of the `EQ` row.
- Each symbol is sorted by date and indexed by it; no missing values need imputation.

## Using other data

Any `.csv` files with the same columns placed in `raw/` are read and combined automatically.
