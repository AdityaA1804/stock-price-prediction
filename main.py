"""Next-day stock price prediction - full pipeline.

Run:  python main.py                       (settings from config.yaml)
      python main.py --symbols TCS INFY --window 10 --test-part 0.2
      python main.py --symbols all --skip-experiments
"""

import argparse
import os
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.exceptions import ConvergenceWarning

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)                                      # relative paths work from any launch folder
sys.path.insert(0, str(ROOT))

from src.config import load_config                  # noqa: E402
from src.data_loader import load_data, resolve_symbols  # noqa: E402
from src.experiments import run_experiments         # noqa: E402
from src.pipeline import run_symbol                 # noqa: E402
from src.visualize import plot_model_comparison, plot_prediction  # noqa: E402

warnings.filterwarnings("ignore", category=ConvergenceWarning)

SHOW_COLS = ["Symbol", "Rank", "Model", "RMSE", "RMSE / Naive", "Direction Acc %",
             "Next-day forecast", "Parameters"]


def save_readme_figures(run, symbol, figures_dir):
    """Model comparison and best-model prediction plot for the README."""
    table, d = run["table"], run["data"]
    best = table.loc[table["Type"] == "ML", "Model"].iloc[0]
    plt.close(plot_model_comparison(table, symbol, Path(figures_dir) / "model_comparison.png"))
    plt.close(plot_prediction(run["predictions"][best], d["close_today"], d["close_next"],
                              f"{symbol} - best model ({best}) on the test set",
                              Path(figures_dir) / "best_model_prediction.png"))


def main():
    parser = argparse.ArgumentParser(description="Next-day stock price prediction")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--symbols", nargs="+", help='symbols to run, or "all"')
    parser.add_argument("--window", type=int, help="number of past days used as input")
    parser.add_argument("--test-part", type=float, help="fraction of days held out for testing")
    parser.add_argument("--models", nargs="+", help="subset of models (default: all)")
    parser.add_argument("--skip-experiments", action="store_true",
                        help="skip the feature-ablation and window experiments")
    args = parser.parse_args()

    try:
        cfg = load_config(args.config, symbols=args.symbols, window=args.window,
                          test_part=args.test_part, models=args.models)
        raw = load_data(cfg["data_dir"])
        symbols = resolve_symbols(raw, cfg["symbols"])

        readme_symbol = cfg["readme_symbol"] if cfg["readme_symbol"] in symbols else symbols[0]
        tables = []
        for symbol in symbols:
            print(f"\n{symbol}  (window = {cfg['window']} days, test = last {cfg['test_part']:.0%})")
            run = run_symbol(raw, symbol, cfg)
            tables.append(run["table"])
            if symbol == readme_symbol:
                save_readme_figures(run, symbol, cfg["figures_dir"])
    except (FileNotFoundError, ValueError) as e:
        sys.exit(f"\nError: {e}")

    summary = pd.concat(tables, ignore_index=True)
    results_dir = Path(cfg["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(results_dir / "summary.csv", index=False)

    print("\n" + "=" * 100)
    with pd.option_context("display.width", 250, "display.max_colwidth", 70,
                           "display.float_format", "{:.3f}".format):
        print(summary[SHOW_COLS].to_string(index=False))

    if cfg["experiments"]["run"] and not args.skip_experiments:
        out = run_experiments(raw, symbols, summary, cfg)
        with pd.option_context("display.float_format", "{:.3f}".format):
            print("\nBest model per stock\n", out["best"].to_string(index=False))
            print("\nFeature ablation (mean test RMSE / Naive)\n", out["ablation"].to_string())
            print("\nWindow sensitivity (mean test RMSE / Naive)\n", out["windows"].to_string())

    print(f"\nTables: {results_dir.resolve()}")
    print(f"Figures: {Path(cfg['figures_dir']).resolve()}")


if __name__ == "__main__":
    main()
