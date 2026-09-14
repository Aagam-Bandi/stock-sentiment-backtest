# Data

- `news_data.csv` — 1,457 NYT headlines, 2020-01-03 to 2023-12-19, ticker-tagged
- `stock_data.csv` — daily OHLCV from yfinance, 2023-07-03 to 2024-06-13

These two ranges overlap on only 77 dates. That mismatch is the subject of
the post-mortem in the top-level README; the files are included unmodified
so the finding can be reproduced.

Note: `news_data.csv` contains a stray duplicate header row, an artefact of
the original append-mode write. Filter rows where `Date == "Date"`.
