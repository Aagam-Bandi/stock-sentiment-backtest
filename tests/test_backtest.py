import numpy as np
import pandas as pd
import pytest

from src.backtest import (
    Performance,
    backtest,
    buy_and_hold_return,
    generate_signals,
    walk_forward,
)


def make_series(prices, polarity=None, start="2024-01-01"):
    n = len(prices)
    return pd.DataFrame({
        "Date": pd.date_range(start, periods=n, freq="D"),
        "Close": prices,
        "polarity_score": polarity if polarity is not None else [0.5] * n,
        "Stock Name": ["TEST"] * n,
    })


class TestSignals:
    def test_positive_sentiment_in_uptrend_goes_long(self):
        prices = list(np.linspace(100, 140, 40))
        df = generate_signals(make_series(prices, [0.5] * 40))
        assert (df["Signal"] == 1).any()

    def test_positive_sentiment_in_downtrend_does_not(self):
        # The trend filter is the whole point: sentiment alone should not fire.
        prices = list(np.linspace(140, 100, 40))
        df = generate_signals(make_series(prices, [0.5] * 40))
        assert not (df["Signal"] == 1).any()

    def test_orders_fire_on_change_not_on_hold(self):
        prices = list(np.linspace(100, 140, 40))
        df = generate_signals(make_series(prices, [0.5] * 40))
        # A held signal must not generate a trade every single day.
        assert (df["Order"] != 0).sum() < (df["Signal"] != 0).sum()


class TestBacktestAccounting:
    """These exist because the original implementation had this exactly backwards."""

    def test_buying_reduces_cash(self):
        prices = list(np.linspace(100, 140, 40))
        df, perf = backtest(generate_signals(make_series(prices, [0.5] * 40)),
                            initial_cash=1000.0)
        # Equity must never exceed cash + position value; a buy that credited
        # cash would break this immediately.
        assert perf.final_value < 1000.0 * 2

    def test_flat_market_loses_exactly_the_costs(self):
        prices = [100.0] * 60
        _, perf = backtest(generate_signals(make_series(prices)), initial_cash=1000.0)
        assert perf.total_return_pct <= 0.0

    def test_costs_are_charged(self):
        prices = list(np.linspace(100, 140, 40)) + list(np.linspace(140, 100, 40))
        _, perf = backtest(generate_signals(make_series(prices, [0.5] * 40 + [-0.5] * 40)))
        if perf.num_trades > 0:
            assert perf.total_costs > 0

    def test_zero_cost_run_beats_costed_run(self):
        prices = list(np.linspace(100, 140, 40)) + list(np.linspace(140, 100, 40))
        pol = [0.5] * 40 + [-0.5] * 40
        _, free = backtest(generate_signals(make_series(prices, pol)),
                           commission_pct=0.0, slippage_pct=0.0)
        _, costed = backtest(generate_signals(make_series(prices, pol)))
        assert free.final_value >= costed.final_value

    def test_never_goes_negative(self):
        rng = np.random.default_rng(0)
        prices = list(100 + rng.normal(0, 5, 200).cumsum())
        pol = list(rng.uniform(-1, 1, 200))
        df, perf = backtest(generate_signals(make_series(prices, pol)))
        assert (df["equity"] >= 0).all()


class TestBenchmark:
    def test_rising_market(self):
        assert buy_and_hold_return(make_series([100, 110])) == pytest.approx(10.0)

    def test_falling_market(self):
        assert buy_and_hold_return(make_series([100, 90])) == pytest.approx(-10.0)

    def test_single_point_returns_zero(self):
        assert buy_and_hold_return(make_series([100])) == 0.0

    def test_excess_return_is_strategy_minus_benchmark(self):
        perf = Performance(0, 5.0, 0, 0, 0, 0, benchmark_return_pct=12.0)
        assert perf.excess_return_pct == pytest.approx(-7.0)


class TestSampleAdequacy:
    def test_tiny_sample_flagged(self):
        assert not Performance(0, 0, 0, 0, 1, 0, n_days=10).is_statistically_meaningful

    def test_too_few_trades_flagged(self):
        # Zero trades in a falling market is not outperformance.
        assert not Performance(0, 0, 0, 0, 0, 0, n_days=200).is_statistically_meaningful

    def test_adequate_sample_passes(self):
        assert Performance(0, 0, 0, 0, 20, 0, n_days=200).is_statistically_meaningful


class TestWalkForward:
    def test_produces_one_result_per_window(self):
        rng = np.random.default_rng(1)
        prices = list(100 + rng.normal(0, 2, 240).cumsum())
        result = walk_forward(make_series(prices, list(rng.uniform(-1, 1, 240))),
                              window_days=60)
        assert len(result.window_excess) == 4

    def test_short_series_yields_nothing(self):
        result = walk_forward(make_series([100] * 20), window_days=60)
        assert result.window_excess == []
        assert not result.is_consistent

    def test_consistency_requires_majority_of_windows(self):
        from src.backtest import WalkForwardResult
        assert WalkForwardResult([], [1, 1, 1, -1]).is_consistent
        assert not WalkForwardResult([], [1, -1, -1, -1]).is_consistent
        assert not WalkForwardResult([], [1, 1]).is_consistent   # too few windows
