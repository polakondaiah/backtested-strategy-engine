"""
Walk-forward robustness check, replacing "backtest once on the whole
history and report the number" with the standard research safeguard
against regime-specific luck and parameter overfitting:

1. Rolling out-of-sample folds: refit nothing (the z-score signal has no
   fitted parameters beyond the window), but evaluate Sharpe/hit-rate on
   successive non-overlapping folds to check the result isn't carried by
   one lucky sub-period.
2. Parameter sensitivity: re-run the full pipeline across a grid of
   z-score windows. A signal that only "works" (or only fails) at exactly
   window=20 is a red flag for a spurious fit, even in a small grid like
   this one.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .backtest import load_prices, backtest
from .stats import newey_west_mean_tstat, block_bootstrap_sharpe_ci

RESULTS = Path(__file__).parent.parent / "results"
N_FOLDS = 4
WINDOW_GRID = [10, 15, 20, 30, 60]


def rolling_folds(daily: pd.Series, n_folds: int = N_FOLDS) -> list[dict]:
    idx = daily.index
    fold_edges = np.linspace(0, len(idx), n_folds + 1, dtype=int)
    folds = []
    for i in range(n_folds):
        seg = daily.iloc[fold_edges[i]:fold_edges[i + 1]]
        if len(seg) < 10:
            continue
        ann_ret = seg.mean() * 252
        ann_vol = seg.std() * np.sqrt(252)
        sharpe = ann_ret / ann_vol if ann_vol > 0 else 0.0
        folds.append({
            "fold": i + 1,
            "start": str(seg.index.min()),
            "end": str(seg.index.max()),
            "n_days": int(len(seg)),
            "sharpe": float(sharpe),
            "hit_rate": float((seg > 0).mean()),
        })
    return folds


def parameter_sensitivity(df: pd.DataFrame, windows: list[int] = WINDOW_GRID) -> list[dict]:
    rows = []
    for w in windows:
        daily, equity, dd, metrics, bt = backtest(df.copy(), z_window=w)
        rows.append({
            "window": w,
            "sharpe": metrics["sharpe_ratio"],
            "annualized_return": metrics["annualized_return"],
            "max_drawdown": metrics["max_drawdown"],
            "hit_rate": metrics["hit_rate"],
        })
    return rows


def main():
    df = load_prices()
    daily, equity, dd, metrics, bt = backtest(df.copy())

    folds = rolling_folds(daily)
    nw = newey_west_mean_tstat(daily.values)
    boot = block_bootstrap_sharpe_ci(daily.values)
    sens = parameter_sensitivity(df)

    out = {
        "rolling_folds": folds,
        "newey_west_significance": nw,
        "bootstrap_sharpe_ci": boot,
        "parameter_sensitivity": sens,
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    with open(RESULTS / "robustness.json", "w") as f:
        json.dump(out, f, indent=2)

    print("=== Rolling out-of-sample folds ===")
    print(pd.DataFrame(folds).to_string(index=False))
    print("\n=== Newey-West significance (whole-sample daily mean return) ===")
    print(json.dumps(nw, indent=2))
    print("\n=== Block-bootstrap Sharpe 95% CI ===")
    print(json.dumps(boot, indent=2))
    print("\n=== Parameter sensitivity (z-score window) ===")
    print(pd.DataFrame(sens).to_string(index=False))
    print(f"\n[done] wrote {RESULTS / 'robustness.json'}")


if __name__ == "__main__":
    main()
