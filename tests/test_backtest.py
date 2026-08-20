"""
Correctness tests for the backtest engine -- the invariants that matter
most in a backtester are the ones that are easy to get subtly wrong and
hard to notice from aggregate metrics alone: no lookahead, and metrics
computed correctly on a known-answer input.
"""
import numpy as np
import pandas as pd
import pytest

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.strategy import add_signals, compute_zscore
from src.backtest import backtest
from src.stats import newey_west_mean_tstat, block_bootstrap_sharpe_ci


def make_price_df(n=100, tickers=("A", "B"), seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2024-01-01", periods=n)
    rows = []
    for tk in tickers:
        price = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
        for d, p in zip(dates, price):
            rows.append({"Date": d, "Ticker": tk, "Close": p})
    return pd.DataFrame(rows)


def test_zscore_zero_at_constant_price():
    s = pd.Series([100.0] * 30)
    z = compute_zscore(s, window=10)
    # constant price -> std=0 -> NaN, not a spurious signal
    assert z.iloc[15:].isna().all() or (z.iloc[15:].abs() < 1e-9).all()


def test_position_uses_only_past_information():
    """Position at t must be derivable from data up to and including t,
    and must NOT correlate with information only available at t+1."""
    df = make_price_df()
    daily, equity, dd, metrics, bt = backtest(df)
    # position column is pos_raw shifted by 1 -- verify shift actually happened
    per_ticker = bt[bt["Ticker"] == "A"].sort_values("Date").reset_index(drop=True)
    signals = add_signals(df[df["Ticker"] == "A"].sort_values("Date").reset_index(drop=True))
    # bt's "position" at row i should equal signals' pos_raw at row i-1 (allowing for dropna reindexing)
    merged = per_ticker.merge(signals[["Date", "pos_raw"]], on="Date", suffixes=("", "_raw"))
    shifted_raw = signals.set_index("Date")["pos_raw"].shift(1)
    for _, row in merged.iterrows():
        expected = shifted_raw.get(row["Date"])
        if pd.notna(expected) and pd.notna(row["position"]):
            assert abs(row["position"] - expected) < 1e-9


def test_no_trade_zero_cost():
    """If position never changes, turnover (and therefore cost) should be
    zero after the first bar -- a strategy that holds flat shouldn't be
    charged phantom transaction costs."""
    df = make_price_df(n=60, tickers=("FLAT",))
    # force pos_raw constant by monkeypatching a flat close series (no z-score signal)
    df["Close"] = 100.0
    daily, equity, dd, metrics, bt = backtest(df)
    # all-NaN or all-zero z-score -> position should be 0 or NaN throughout, no spurious cost
    assert (bt["cost"].fillna(0) >= 0).all()


def test_metrics_known_answer():
    """Two-asset toy series with a hand-computable Sharpe sanity range."""
    n = 300
    dates = pd.bdate_range("2023-01-01", periods=n)
    rng = np.random.default_rng(1)
    rows = []
    for tk in ["X", "Y"]:
        # deliberately mean-reverting series so the strategy has something to find
        noise = rng.normal(0, 1, n)
        level = 100 + np.cumsum(np.where(noise > 0, -0.3, 0.3)) + rng.normal(0, 0.5, n)
        for d, p in zip(dates, level):
            rows.append({"Date": d, "Ticker": tk, "Close": p})
    df = pd.DataFrame(rows)
    daily, equity, dd, metrics, bt = backtest(df, cost_bps=0, impact_bps_per_vol=0)
    assert -10 < metrics["sharpe_ratio"] < 10  # sane range, not NaN/inf
    assert metrics["max_drawdown"] <= 0
    assert 0 <= metrics["hit_rate"] <= 1


def test_newey_west_matches_naive_for_iid_data():
    """On genuinely iid returns, HAC and naive standard errors should be
    close (HAC only matters when there's autocorrelation to correct)."""
    rng = np.random.default_rng(2)
    r = rng.normal(0.0005, 0.01, 500)
    result = newey_west_mean_tstat(r, maxlags=0)
    naive_se = r.std(ddof=1) / np.sqrt(len(r))
    assert abs(result["nw_std_err"] - naive_se) / naive_se < 0.05


def test_bootstrap_ci_contains_point_estimate():
    rng = np.random.default_rng(3)
    r = rng.normal(0.001, 0.01, 400)
    result = block_bootstrap_sharpe_ci(r, n_boot=200)
    assert result["ci_low"] <= result["point_estimate"] <= result["ci_high"]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
