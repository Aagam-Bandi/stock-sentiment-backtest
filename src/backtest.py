"""Signal generation and backtesting.

The original version of this module had inverted trade accounting — the buy
branch fired on a sell signal and *added* cash on entry instead of deducting
it. The corrected logic is below; see README "What went wrong" for why the
resulting performance figures changed so much.
"""

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

SMA_WINDOW = 20
TRADING_DAYS = 252

# Round-trip cost assumption. Retail brokerage plus exchange fees plus the
# half-spread paid on entry and exit. Any edge as small as the one this
# strategy produces lives or dies on this number, so it is not optional.
COMMISSION_PCT = 0.001      # 10 bps per side
SLIPPAGE_PCT = 0.0005       # 5 bps per side, adverse fill

MIN_OVERLAP_DAYS = 30


@dataclass
class Performance:
    final_value: float
    total_return_pct: float
    sharpe_ratio: float
    max_drawdown_pct: float
    num_trades: int
    win_rate: float
    total_costs: float = 0.0
    benchmark_return_pct: float = 0.0
    n_days: int = 0

    @property
    def excess_return_pct(self) -> float:
        """Return above buy-and-hold. This is the only number that matters:
        a strategy returning 8% in a market that returned 12% destroyed value."""
        return self.total_return_pct - self.benchmark_return_pct

    @property
    def is_statistically_meaningful(self) -> bool:
        return self.n_days >= MIN_OVERLAP_DAYS and self.num_trades >= 5

    def __str__(self) -> str:
        verdict = "" if self.is_statistically_meaningful else \
            f"\n  ** {self.n_days} days / {self.num_trades} trades — sample too small **"
        return (
            f"Final value      {self.final_value:>12,.2f}\n"
            f"Total return     {self.total_return_pct:>11.2f}%\n"
            f"Buy & hold       {self.benchmark_return_pct:>11.2f}%\n"
            f"Excess return    {self.excess_return_pct:>11.2f}%\n"
            f"Sharpe ratio     {self.sharpe_ratio:>12.2f}\n"
            f"Max drawdown     {self.max_drawdown_pct:>11.2f}%\n"
            f"Trades           {self.num_trades:>12d}\n"
            f"Win rate         {self.win_rate:>11.2f}%\n"
            f"Costs paid       {self.total_costs:>12,.2f}"
            f"{verdict}"
        )


def buy_and_hold_return(df: pd.DataFrame) -> float:
    """Benchmark: buy at the first close, hold to the last."""
    prices = df["Close"].dropna()
    if len(prices) < 2:
        return 0.0
    return float((prices.iloc[-1] / prices.iloc[0] - 1) * 100)


def generate_signals(df: pd.DataFrame) -> pd.DataFrame:
    """Combine news sentiment with a trend filter.

    Sentiment alone is a weak and noisy signal. Requiring price to agree with
    it — above the moving average for longs, below for shorts — filters the
    cases where a positive headline lands in a downtrend.
    """
    df = df.sort_values("Date").copy()
    df["SMA"] = df["Close"].rolling(window=SMA_WINDOW).mean()

    df["Signal"] = 0
    df.loc[(df["polarity_score"] > 0) & (df["Close"] > df["SMA"]), "Signal"] = 1
    df.loc[(df["polarity_score"] < 0) & (df["Close"] < df["SMA"]), "Signal"] = -1

    # Trade on signal *changes*, not on the signal being held.
    df["Order"] = df["Signal"].diff().fillna(0)
    return df


def backtest(
    df: pd.DataFrame,
    initial_cash: float = 1000.0,
    stop_loss_pct: float = 0.10,
    take_profit_pct: float = 0.20,
    position_size_pct: float = 0.20,
    commission_pct: float = COMMISSION_PCT,
    slippage_pct: float = SLIPPAGE_PCT,
) -> tuple[pd.DataFrame, Performance]:
    """Long-only backtest with stop-loss, take-profit, costs and a benchmark.

    Costs are charged on both sides and slippage is applied adversely — buys
    fill above the close, sells below it. Still no shorting and no intraday
    fills, so this remains optimistic, but it no longer assumes trading is free.
    """
    df = df.copy()
    cash = initial_cash
    shares = 0.0
    entry_price = 0.0
    equity: list[float] = []
    trades: list[float] = []
    costs = 0.0

    def _sell(price: float) -> tuple[float, float]:
        """Exit at an adverse fill, net of commission."""
        fill = price * (1 - slippage_pct)
        gross = shares * fill
        fee = gross * commission_pct
        return gross - fee, fee

    for idx, row in df.iterrows():
        price = row["Close"]
        if pd.isna(price):
            equity.append(cash + shares * (entry_price or 0))
            continue

        # ENTER on a fresh long signal, if flat.
        if row["Order"] > 0 and shares == 0:
            spend = cash * position_size_pct
            if spend > 0:
                fill = price * (1 + slippage_pct)      # adverse fill on entry
                fee = spend * commission_pct
                shares = (spend - fee) / fill
                cash -= spend
                costs += fee
                entry_price = fill

        # EXIT on a fresh short/neutral signal, if holding.
        elif row["Order"] < 0 and shares > 0:
            proceeds, fee = _sell(price)
            cash += proceeds
            costs += fee
            trades.append((price * (1 - slippage_pct) - entry_price) / entry_price)
            shares = 0.0
            entry_price = 0.0

        # Risk exits, checked only while holding.
        elif shares > 0 and (
            price <= entry_price * (1 - stop_loss_pct)
            or price >= entry_price * (1 + take_profit_pct)
        ):
            proceeds, fee = _sell(price)
            cash += proceeds
            costs += fee
            trades.append((price * (1 - slippage_pct) - entry_price) / entry_price)
            shares = 0.0
            entry_price = 0.0

        equity.append(cash + shares * price)

    df["equity"] = equity
    series = pd.Series(equity)

    returns = series.pct_change().dropna()
    sharpe = (
        np.sqrt(TRADING_DAYS) * returns.mean() / returns.std()
        if len(returns) > 1 and returns.std() > 0
        else 0.0
    )

    running_max = series.cummax()
    drawdown = (series - running_max) / running_max.replace(0, np.nan)

    wins = [t for t in trades if t > 0]

    return df, Performance(
        final_value=float(series.iloc[-1]) if len(series) else initial_cash,
        total_return_pct=float((series.iloc[-1] / initial_cash - 1) * 100) if len(series) else 0.0,
        sharpe_ratio=float(sharpe),
        max_drawdown_pct=float(drawdown.min() * 100) if len(drawdown.dropna()) else 0.0,
        num_trades=len(trades),
        win_rate=float(len(wins) / len(trades) * 100) if trades else 0.0,
        total_costs=float(costs),
        benchmark_return_pct=buy_and_hold_return(df),
        n_days=len(df),
    )


@dataclass
class WalkForwardResult:
    """Out-of-sample performance across sequential windows.

    A single contiguous backtest says nothing about robustness — a strategy can
    look excellent because one period happened to suit it. Walk-forward splits
    the series into consecutive windows and reports the spread, which is what
    reveals whether an edge is stable or an artefact of one lucky stretch.
    """
    window_returns: list[float]
    window_excess: list[float]

    @property
    def mean_excess(self) -> float:
        return float(np.mean(self.window_excess)) if self.window_excess else 0.0

    @property
    def std_excess(self) -> float:
        return float(np.std(self.window_excess)) if self.window_excess else 0.0

    @property
    def win_windows(self) -> int:
        return sum(1 for x in self.window_excess if x > 0)

    @property
    def is_consistent(self) -> bool:
        """An edge worth anything beats the benchmark in most windows, not on
        average across a handful where one dominates."""
        return (
            len(self.window_excess) >= 3
            and self.win_windows / len(self.window_excess) >= 0.6
        )

    def __str__(self) -> str:
        if not self.window_excess:
            return "Walk-forward: insufficient data for any window"
        verdict = "consistent" if self.is_consistent else "NOT consistent"
        return (
            f"Windows              {len(self.window_excess):>8d}\n"
            f"Mean excess return   {self.mean_excess:>7.2f}%\n"
            f"Std of excess        {self.std_excess:>7.2f}%\n"
            f"Windows beating B&H  {self.win_windows}/{len(self.window_excess)}  ({verdict})"
        )


def walk_forward(df: pd.DataFrame, window_days: int = 60, **kwargs) -> WalkForwardResult:
    """Backtest over consecutive non-overlapping windows."""
    df = df.sort_values("Date").reset_index(drop=True)
    returns: list[float] = []
    excess: list[float] = []

    for start in range(0, len(df) - window_days + 1, window_days):
        window = df.iloc[start:start + window_days]
        if len(window) < window_days:
            break
        _, perf = backtest(generate_signals(window), **kwargs)
        returns.append(perf.total_return_pct)
        excess.append(perf.excess_return_pct)

    return WalkForwardResult(window_returns=returns, window_excess=excess)
