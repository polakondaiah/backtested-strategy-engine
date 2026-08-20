"""
Proper backtest loop with lookahead guard, transaction costs, and metrics.
Shift signal by 1: signal at close t -> position for t+1 return.
"""
import json
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA = Path(__file__).parent.parent / "data" / "prices.csv"
RESULTS = Path(__file__).parent.parent / "results"
COST_BPS = 5  # 5 bps per unit turnover

def load_prices():
    df = pd.read_csv(DATA, parse_dates=["Date"])
    df = df.sort_values(["Ticker","Date"])
    return df

def backtest(df: pd.DataFrame, cost_bps: int = COST_BPS, z_window: int = 20,
             impact_bps_per_vol: float = 0.5):
    """
    impact_bps_per_vol: extra slippage (bps) per unit of trailing 20-day
    realized volatility, on top of the flat cost_bps -- a flat cost alone
    understates real execution cost in high-volatility regimes, so impact
    scales with recent realized vol as a simple, standard proxy (Almgren-
    Chriss-style linear impact, without the full optimal-execution model).
    """
    from .strategy import add_signals
    df = add_signals(df, window=z_window)

    # next-bar-close forward return per ticker (position decided at t, earned at t+1)
    df["fwd_ret"] = df.groupby("Ticker")["Close"].transform(lambda s: s.pct_change().shift(-1))
    # position at t is the signal computed using data up to close[t], shifted
    # forward one bar so it only ever trades on already-known information
    df["position"] = df.groupby("Ticker")["pos_raw"].shift(1)

    df["turnover"] = df.groupby("Ticker")["position"].diff().abs().fillna(df["position"].abs())
    realized_vol = df.groupby("Ticker")["Close"].transform(
        lambda s: s.pct_change().rolling(20).std().fillna(0))
    df["cost_bps_effective"] = cost_bps + impact_bps_per_vol * (realized_vol * 100)
    df["cost"] = df["turnover"] * (df["cost_bps_effective"] / 10000.0)
    df["strategy_ret"] = df["position"] * df["fwd_ret"] - df["cost"]
    # drop NaNs (warmup + last bar)
    bt = df.dropna(subset=["position","fwd_ret","strategy_ret"]).copy()
    # equal-weight portfolio daily return
    daily = bt.groupby("Date")["strategy_ret"].mean().sort_index()
    # equity curve
    equity = (1 + daily).cumprod()
    # metrics
    ann_ret = daily.mean() * 252
    ann_vol = daily.std() * np.sqrt(252)
    sharpe = ann_ret / ann_vol if ann_vol != 0 else 0
    # max drawdown
    running_max = equity.cummax()
    drawdown = (equity - running_max) / running_max
    max_dd = drawdown.min()
    hit_rate = (daily > 0).mean()
    # lookahead check: ensure no position uses contemporaneous fwd_ret without shift
    # (pos is shifted by 1, fwd_ret is next-day return, so no lookahead)
    metrics = {
        "days": int(len(daily)),
        "annualized_return": float(ann_ret),
        "annualized_vol": float(ann_vol),
        "sharpe_ratio": float(sharpe),
        "max_drawdown": float(max_dd),
        "hit_rate": float(hit_rate),
        "cost_bps": cost_bps,
        "universe": sorted(df["Ticker"].unique().tolist()),
        "start": str(daily.index.min()),
        "end": str(daily.index.max()),
    }
    return daily, equity, drawdown, metrics, bt

def plot_equity(equity: pd.Series, drawdown: pd.Series, outpath: Path):
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True, gridspec_kw={"height_ratios":[3,1]})
    axes[0].plot(equity.index, equity.values, label="Strategy equity")
    axes[0].set_ylabel("Cumulative return (1 = start)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[1].fill_between(drawdown.index, drawdown.values, 0, color="red", alpha=0.3)
    axes[1].set_ylabel("Drawdown")
    axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()
    print(f"[plot] saved to {outpath}")

def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    df = load_prices()
    print(f"[backtest] Loaded {len(df):,} rows, {df['Ticker'].nunique()} tickers")
    daily, equity, dd, metrics, bt = backtest(df)
    print("[metrics]", json.dumps(metrics, indent=2))
    with open(RESULTS / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    daily.to_csv(RESULTS / "daily_returns.csv", header=True)
    equity.to_csv(RESULTS / "equity_curve.csv", header=True)
    bt.to_csv(RESULTS / "trades.csv", index=False)
    plot_equity(equity, dd, RESULTS / "equity_curve.png")
    print(f"[done] Results in {RESULTS.resolve()}")

if __name__ == "__main__":
    main()
