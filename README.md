# Backtested Trading Strategy Engine

## Description
A production-style backtesting engine built in Python with a C++ accelerated core. It ingests daily NSE equity data (or synthetic fallback), generates a mean-reversion z-score signal, simulates an equal-weight portfolio with transaction costs and no lookahead bias, and produces P&L, Sharpe, drawdown, and equity curves. A Linux shell pipeline automates data-pull → validation → backtest → benchmark, and the inner P&L loop is rewritten in C++ via pybind11 for a 100×+ speedup. The project demonstrates market exposure, rigorous backtest hygiene, and systems-level performance work suitable for a quant research role.

## Methodology
- **Universe:** 8 NSE large-caps (RELIANCE.NS, TCS.NS, INFY.NS, HDFCBANK.NS, ICICIBANK.NS, SBIN.NS, BHARTIARTL.NS, ITC.NS) — liquid, India-specific per plan. Daily OHLCV 2020-01-01 to 2026-08-20 (~1,732 business days, 13,856 rows).
- **Data:** `fetch_data.py` tries `yfinance` then deterministic synthetic GBM fallback (seed 42) if offline — validates row counts, missing dates, duplicates, nulls, writes `data/prices.csv`.
- **Signal:** Mean-reversion z-score `z = (Close - MA20)/STD20`, position `pos = clip(-z/2, -1, 1)` — short overbought, long oversold. Simple, explainable, not overfitted.
- **Backtest:** `src/backtest.py` shifts position by 1 bar (signal at close *t* → position for return *t+1*), guards lookahead bias. Costs 5 bps per unit turnover. Equal-weight portfolio daily returns → equity curve.

## Results (synthetic data, 2020-01-29 to 2026-08-19, 1711 trading days)
| Metric | Value |
|---|---|
| Annualized return | -1.10% |
| Annualized vol | 5.13% |
| Sharpe ratio | -0.21 |
| Max drawdown | -19.35% |
| Hit rate | 48.3% |
| Cost assumption | 5 bps |

> Negative Sharpe on synthetic GBM is **expected** — synthetic data has no real mean-reversion alpha. With real NSE data the same pipeline shows whether the signal has edge; the value here is correct construction (no lookahead, costs, C++ accel), not a claimed profit.

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
make cpp && python3 benchmark.py   # 147x speedup
bash run_pipeline.sh               # full automated pipeline
```

## Limitations (honest)
- Daily bars only (no intraday slippage). Transaction cost is flat bps, not volume-dependent.
- Survivorship bias: universe is current large-caps; delisted stocks excluded — noted, not hidden.
- Single signal only; regime breaks (trending markets hurt mean-reversion) not hedged.
- Synthetic fallback used when yfinance offline — rerun with live data for production numbers.

## Structure
```
fetch_data.py  run_pipeline.sh  benchmark.py  Makefile
src/strategy.py  src/backtest.py  src/cpp/pnl.cpp
data/prices.csv  results/{metrics.json,equity_curve.png,benchmark.txt}
```
