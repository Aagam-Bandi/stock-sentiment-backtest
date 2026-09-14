# Stock Sentiment Backtest — A Post-Mortem

A news-sentiment trading pipeline: pull headlines, score them with VADER, join to daily prices, generate signals, backtest.

It originally reported a Sharpe ratio of 1.25 across 100+ signals. **That number was wrong.** This repository contains the corrected, tested pipeline and a written account of the three bugs that produced it.

```bash
pip install -r requirements.txt
python run_backtest.py --walk-forward
pytest                                  # 26 tests
```

The failure analysis is the point. The rebuilt pipeline now has the instrumentation that would have caught all three bugs on day one: a buy-and-hold benchmark, transaction costs, walk-forward windows, sample-adequacy gating, and 26 tests — several written specifically to assert the accounting invariants the original violated.

---

## What went wrong

### 1. The backtest accounting was inverted

The original `calculate_performance` loop:

```python
if row['Order'] == -1 and cash > 0:      # labelled "Buy signal"
    position = (cash * position_size_pct) / row['Close']
    cash += position * row['Close']       # ← adds cash when buying
    purchase_price = row['Close']
elif row['Order'] == 1 and position > 0:  # labelled "Sell signal"
    cash -= position * row['Close']       # ← removes cash when selling
```

Two errors compounding. Entries fired on `Order == -1`, which `generate_signals` produces for a *bearish* crossover. And the cash accounting ran backwards — buying credited the account instead of debiting it.

The result was an equity curve that rose smoothly regardless of whether the strategy had any predictive power, because every trade added money. A Sharpe ratio computed over that curve measures nothing.

The corrected version is in `src/backtest.py`.

### 2. The classifier was learning its own labels

The original pipeline scored headlines with VADER, wrote the result to `sentiment_label`, then trained a TF-IDF + RandomForest classifier to predict `sentiment_label` from the headline text.

VADER is a deterministic function of the headline. So the classifier was trained to reproduce a rule-based scorer from the same input that scorer used. The reported 75% accuracy measures how well RandomForest approximates VADER — not how well either measures sentiment, and certainly not how well sentiment predicts returns. There was no independent ground truth anywhere in the loop.

The supervised stage has been removed. VADER's compound score is now used directly, which is what the pipeline was effectively doing anyway, minus the misleading accuracy metric.

### 3. There was almost no data

The one that invalidates everything else:

| Source | Range | Rows |
|---|---|---|
| NYT headlines | 2020-01-03 → 2023-12-19 | 1,457 |
| Yahoo Finance prices | 2023-07-03 → 2024-06-13 | 1,440 |

**Overlapping dates: 77.** Across six tickers. Roughly thirteen usable days each.

The headline collection loop ran over 2020–2023. The price download requested 2023-07-01 onward. The inner join between them silently dropped ~95% of both datasets, and nothing in the pipeline reported it — the merge succeeded, the backtest ran, numbers came out.

There were never 100+ signals. There was never enough data for a Sharpe ratio to mean anything.

A fourth, smaller issue: `news_data.csv` was written in append mode with a header check against the *wrong* filename, so a duplicate header row is embedded mid-file.

---

## Honest results

Running the corrected pipeline, with costs and a benchmark, on the data that actually exists:

```
AAPL: only 22 overlapping days (need 30) — skipping
AMZN: only 28 overlapping days (need 30) — skipping
DIS:  only 16 overlapping days (need 30) — skipping
TM:   only  4 overlapping days (need 30) — skipping

===== GOOGL =====
Total return           -0.02%
Buy & hold              6.67%
Excess return          -6.68%
Trades                      1
  ** 32 days / 1 trades — sample too small **

===== TSLA =====
Total return            0.35%
Buy & hold            -12.52%
Excess return          12.87%
Trades                      0
  ** 35 days / 0 trades — sample too small **
```

The benchmark column is what makes this legible, and it was absent from the original.

**GOOGL underperformed buy-and-hold by 6.7%.** In a rising market the strategy traded once and finished flat.

**TSLA appears to beat the benchmark by 12.9% — on zero trades.** The strategy never entered. It sat in cash while TSLA fell 12.5%. Without a benchmark column that reads like alpha; with one, it's obviously just non-participation.

Both are flagged as statistically meaningless anyway. The runner refuses to backtest fewer than 30 overlapping days, and `Performance.is_statistically_meaningful` additionally requires at least 5 trades — a strategy that never trades has not been tested regardless of how many days it covers.

Walk-forward reports "insufficient data for any window," which is the correct answer at this sample size.

---

## What the pipeline does

```
NYT Archive API ──► headlines (ticker-tagged, deduplicated by day)
                         │
                         ▼
                    VADER compound polarity
                         │
                         ▼
                    daily mean per ticker
                         │
yfinance ──► OHLCV ──────┤
                         ▼
                    inner join on (Date, Ticker)
                         │
                         ▼
              signal = sentiment sign AND price vs 20-day SMA
                         │
                         ▼
              long-only backtest, stop-loss 10%, take-profit 20%
```

**Sentiment plus trend, not sentiment alone.** A long requires positive polarity *and* price above the 20-day moving average. Sentiment on its own is noisy; requiring price confirmation filters headlines that land against the prevailing trend.

**Continuous polarity, not buckets.** VADER's compound score is kept as a float rather than bucketed into positive/neutral/negative. Bucketing discards magnitude, and magnitude is what separates a mildly positive headline from a strongly positive one.

**Daily mean, not sum.** A ticker with twenty headlines in a day shouldn't get twenty times the signal strength of one with a single headline.

**Costs are charged on both sides.** 10 bps commission and 5 bps adverse slippage per side. Buys fill above the close, sells below it. Any edge as small as this strategy produces lives or dies on that assumption, so free trading is not an acceptable simplification.

**Walk-forward, not one contiguous period.** A strategy can look excellent because a single stretch happened to suit it. `walk_forward` splits the series into consecutive non-overlapping windows and reports the spread of excess return, plus how many windows actually beat the benchmark. `is_consistent` requires at least three windows and a 60% win rate — beating the benchmark on average across a handful of windows where one dominates is not an edge.

---

## What's fixed, and what still isn't

**Fixed in this repo:**

- Trade accounting corrected, with tests asserting that buying reduces cash and a flat market loses exactly the costs
- Circular supervised labelling removed
- Transaction costs and adverse slippage charged on both sides
- Buy-and-hold benchmark and excess return reported
- Walk-forward window analysis with a consistency criterion
- Sample-adequacy gating — the pipeline refuses to report on too few days or too few trades
- 26 tests, including regression tests for each accounting bug

**Still open, and it needs you to re-run collection:**

The date ranges don't overlap. Headlines cover 2020–2023, prices cover 2023–2024. No amount of code fixes a 77-day intersection — the data has to be collected again over a common window. Once it is, everything above is already in place to evaluate it honestly.

The remaining methodological gap is ground truth: either hand-label a sample of headlines, or drop classification entirely and test whether VADER scores predict forward returns, which is the question that actually matters.

---

## Install and run

```bash
git clone https://github.com/Aagam-Bandi/stock-sentiment-backtest
cd stock-sentiment-backtest
pip install -r requirements.txt

cp .env.example .env
# add your NYT Archive API key if re-collecting headlines

python run_backtest.py               # all tickers
python run_backtest.py --ticker GOOGL
```

The original collection and modelling steps are preserved in `notebooks/sentiment_pipeline.ipynb` for reference.

---

## Layout

```
src/
  sentiment.py   VADER scoring and daily aggregation
  backtest.py    signals, costed backtest, benchmark, walk-forward
tests/           26 tests, including regression tests for each bug above
run_backtest.py  end-to-end runner
notebooks/       original exploratory work
data/            collected headlines and prices
```

```bash
python run_backtest.py --ticker GOOGL
python run_backtest.py --walk-forward --window 60
pytest
```

---

## Stack

Python · pandas · NLTK (VADER) · yfinance · NYT Archive API · pytest

Originally built for the FinClub project, IIT Roorkee. Audited and corrected afterwards; the audit is the contribution.

## License

MIT
