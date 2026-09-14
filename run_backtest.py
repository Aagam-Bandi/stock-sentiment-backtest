"""Score headlines, join to prices, generate signals, backtest.

    python run_backtest.py
    python run_backtest.py --ticker AAPL
"""

import argparse
import logging

import pandas as pd

from src import (
    MIN_OVERLAP_DAYS,
    aggregate_daily,
    backtest,
    generate_signals,
    score_headlines,
    walk_forward,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the sentiment backtest.")
    parser.add_argument("--news", default="data/news_data.csv")
    parser.add_argument("--prices", default="data/stock_data.csv")
    parser.add_argument("--ticker", help="restrict to one ticker")
    parser.add_argument("--cash", type=float, default=1000.0)
    parser.add_argument("--walk-forward", action="store_true",
                        help="also run out-of-sample window analysis")
    parser.add_argument("--window", type=int, default=60,
                        help="walk-forward window length in days")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s  %(levelname)-7s %(message)s",
    )

    news = pd.read_csv(args.news)
    prices = pd.read_csv(args.prices)

    scored = aggregate_daily(score_headlines(news))

    merged = prices.merge(scored, on=["Date", "Stock Name"], how="inner")
    merged["Date"] = pd.to_datetime(merged["Date"])

    tickers = [args.ticker] if args.ticker else sorted(merged["Stock Name"].unique())

    for ticker in tickers:
        subset = merged[merged["Stock Name"] == ticker]
        if len(subset) < MIN_OVERLAP_DAYS:
            print(f"\n{ticker}: only {len(subset)} overlapping days "
                  f"(need {MIN_OVERLAP_DAYS}) — skipping")
            continue

        _, perf = backtest(generate_signals(subset), initial_cash=args.cash)
        print(f"\n===== {ticker} =====")
        print(perf)

        if args.walk_forward:
            wf = walk_forward(subset, window_days=args.window, initial_cash=args.cash)
            print(f"\n--- walk-forward ({args.window}-day windows) ---")
            print(wf)


if __name__ == "__main__":
    main()
