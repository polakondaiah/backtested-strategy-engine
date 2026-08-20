# Backtested Trading Strategy Engine

## Description
A production-style backtesting engine built in Python with a C++ accelerated core. It ingests daily NSE equity data (or synthetic fallback), generates a mean-reversion z-score signal, simulates an equal-weight portfolio with transaction costs and no lookahead bias, and produces P&L, Sharpe, drawdown, and equity curves. A Linux shell pipeline automates data-pull → validation → backtest → benchmark, and the inner P&L loop is rewritten in C++ via pybind11 for a 100×+ speedup. The project demonstrates market exposure, rigorous backtest hygiene, and systems-level performance work suitable for a quant research role.

## Methodology
- **Universe:** 8 NSE large-caps (RELIANCE.NS, TCS.NS, INFY.NS, HDFCBANK.NS, ICICIBANK.NS, SBIN.NS, BHARTIARTL.NS, ITC.NS) — liquid, India-specific per plan. Daily OHLCV 2020-01-01 to 2026-08-20 (~1,732 business days, 13,856 rows).
- **Data:** `fetch_data.py` tries `yfinance` then deterministic synthetic GBM fallback (seed 42) if offline — validates row counts, missing dates, duplicates, nulls, writes `data/prices.csv`.
- **Signal:** Mean-reversion z-score `z = (Close - MA20)/STD20`, position `pos = clip(-z/2, -1, 1)` — short overbought, long oversold. Simple, explainable, not overfitted.
- **Backtest:** `src/backtest.py` shifts position by 1 bar (signal at close *t* → position for return *t+1*), guards lookahead bias. Cost model is a flat 5 bps per unit turnover **plus** a volatility-scaled impact term (0.5 bps per 1% of trailing 20-day realized daily vol) — a flat cost alone understates real execution cost in high-volatility regimes; this is a simple linear-impact proxy (Almgren-Chriss-style), not a full optimal-execution model. Equal-weight portfolio daily returns → equity curve.

## Results (synthetic data, 2020-01-29 to 2026-08-19, 1711 trading days)
| Metric | Value |
|---|---|
| Annualized return | -0.96% |
| Annualized vol | 4.97% |
| Sharpe ratio | -0.19 |
| Max drawdown | -8.58% |
| Hit rate | 49.0% |
| Cost assumption | 5 bps flat + vol-scaled impact |

> Negative Sharpe on synthetic GBM is **expected** — synthetic data has no real mean-reversion alpha. With real NSE data the same pipeline shows whether the signal has edge; the value here is correct construction (no lookahead, realistic costs, statistical rigor, C++ accel), not a claimed profit.

## Statistical robustness (`src/walkforward.py` → `results/robustness.json`)
A single Sharpe number on one sample period is not evidence of anything by itself — three checks that go beyond it:

1. **Newey-West (HAC) significance test** on the daily mean return, since strategy returns are typically autocorrelated (Newey & West, 1987): `t = -0.54, p = 0.59` — **not statistically distinguishable from zero**, consistent with the "no real alpha in synthetic data" expectation above, now backed by a test rather than an eyeballed Sharpe.
2. **Stationary block-bootstrap 95% CI on Sharpe** (2,000 resamples, block size 10, following the spirit of Lo (2002) on Sharpe-ratio uncertainty): **[-0.888, +0.514]** — the interval straddles zero, i.e. the data can't rule out a genuinely flat or even positive Sharpe on a different draw from the same process.
3. **Rolling out-of-sample folds** (4 non-overlapping ~1.7-year windows): Sharpe ranges from **+0.51 (fold 2) to -0.71 (fold 4)** — the whole-sample number is not being carried by one lucky sub-period; if anything it's unstable across regimes (even flipping sign), which is itself the honest finding.
4. **Parameter sensitivity** across z-score windows {10, 15, 20, 30, 60}: Sharpe ranges from -0.94 (window=10) to +0.18 (window=60), crossing zero within the grid — the headline window=20 result is squarely inside that noisy range, not a cherry-picked outlier, but the sign-flip across windows is itself evidence there's no robust edge here.

Run: `python3 -m src.walkforward`

## Tests (`tests/test_backtest.py`)
6 pytest cases covering the properties that are easy to silently break and hard to catch from aggregate metrics alone: z-score behavior on constant prices, that `position[t]` only ever uses information available up to `t` (no lookahead), zero turnover cost when flat, sane metric ranges on a known-mean-reverting toy series, and that the Newey-West/bootstrap utilities behave correctly on synthetic iid data. Run: `python3 -m pytest tests/ -v`.

## C++ Component
Hot loop (`position * fwd_ret - turnover*cost`) rewritten in C++ via `pybind11` (`src/cpp/pnl.cpp`).
```
Python avg: 883.81 ms  (1M bars, 5 runs)
C++    avg: 5.98 ms
Speedup: 147.9x
Max error: 0.00e+00
```
Interpreter overhead in tight Python loop is eliminated; C++ operates on contiguous numpy arrays.

## Linux / Shell Evidence
`run_pipeline.sh` — real bash script with timestamps, validation, logging to `pipeline.log`, C++ build step, and cron example (`0 6 * * * cd /path/to/project && bash run_pipeline.sh`). Wire to `cron`/`systemd` for unattended daily runs.

## Reproduce
```bash
python3 fetch_data.py              # or --synthetic to force synthetic
python3 -m src.backtest            # metrics.json + equity_curve.png in results/
python3 -m src.walkforward         # robustness.json: NW significance, bootstrap CI, folds, param sweep
python3 -m pytest tests/ -v        # correctness tests
make cpp && python3 benchmark.py   # 147x speedup
bash run_pipeline.sh               # full automated pipeline
```

## Limitations (honest)
- Daily bars only (no intraday slippage). Cost model is flat-plus-vol-scaled, not a full limit-order-book/impact model.
- Survivorship bias: universe is current large-caps; delisted stocks excluded — noted, not hidden.
- Single signal only; regime breaks (trending markets hurt mean-reversion) not hedged.
- Synthetic fallback used when yfinance offline — rerun with live data for production numbers.
- Robustness checks (walk-forward, bootstrap) are applied to one fixed signal design, not a full walk-forward *re-optimization* loop (the z-score signal has essentially one free parameter — the window — swept in `parameter_sensitivity`, not refit per fold); a strategy with more free parameters would need in-fold fitting to avoid the same overfitting risk this project is designed to catch.

## Structure
```
fetch_data.py  run_pipeline.sh  benchmark.py  Makefile
src/strategy.py  src/backtest.py  src/stats.py  src/walkforward.py  src/cpp/pnl.cpp
tests/test_backtest.py
data/prices.csv  results/{metrics.json,robustness.json,equity_curve.png,benchmark.txt}
```
