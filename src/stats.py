"""
Statistical significance utilities for strategy return series.

Daily strategy returns are typically autocorrelated (overlapping signal
windows, slow-moving positions), so a naive iid t-test on the mean
overstates significance. Newey-West (HAC) standard errors correct for
this; reported alongside a stationary block-bootstrap confidence interval
on the Sharpe ratio, which doesn't assume normality of returns at all.

References: Newey & West (1987, Econometrica) for HAC SEs; Lo (2002,
"The Statistics of Sharpe Ratios") for the asymptotic Sharpe-ratio
variance this bootstrap is checked against.
"""
import numpy as np
import statsmodels.api as sm


def newey_west_mean_tstat(returns: np.ndarray, maxlags: int | None = None) -> dict:
    """HAC-adjusted t-stat for H0: mean daily return == 0."""
    r = np.asarray(returns, dtype=float)
    r = r[~np.isnan(r)]
    n = len(r)
    if maxlags is None:
        maxlags = int(np.floor(4 * (n / 100) ** (2 / 9)))  # Newey-West (1994) rule of thumb
    X = np.ones((n, 1))
    model = sm.OLS(r, X).fit(cov_type="HAC", cov_kwds={"maxlags": maxlags})
    return {
        "n": n,
        "maxlags": maxlags,
        "mean_daily_return": float(model.params[0]),
        "nw_std_err": float(model.bse[0]),
        "t_stat": float(model.tvalues[0]),
        "p_value": float(model.pvalues[0]),
        "significant_at_5pct": bool(model.pvalues[0] < 0.05),
    }


def block_bootstrap_sharpe_ci(returns: np.ndarray, n_boot: int = 2000, block_size: int = 10,
                               ci: float = 0.95, seed: int = 42) -> dict:
    """Stationary block bootstrap CI for annualized Sharpe -- blocks preserve
    short-horizon autocorrelation instead of assuming iid resampling."""
    r = np.asarray(returns, dtype=float)
    r = r[~np.isnan(r)]
    n = len(r)
    rng = np.random.default_rng(seed)
    boot_sharpes = np.empty(n_boot)
    n_blocks = int(np.ceil(n / block_size))
    for b in range(n_boot):
        starts = rng.integers(0, n - block_size, size=n_blocks)
        sample = np.concatenate([r[s:s + block_size] for s in starts])[:n]
        mu, sd = sample.mean(), sample.std(ddof=1)
        boot_sharpes[b] = (mu / sd) * np.sqrt(252) if sd > 0 else 0.0
    lo_pct, hi_pct = (1 - ci) / 2 * 100, (1 + ci) / 2 * 100
    return {
        "n_boot": n_boot,
        "block_size": block_size,
        "point_estimate": float((r.mean() / r.std(ddof=1)) * np.sqrt(252)) if r.std(ddof=1) > 0 else 0.0,
        "ci_low": float(np.percentile(boot_sharpes, lo_pct)),
        "ci_high": float(np.percentile(boot_sharpes, hi_pct)),
        "ci_level": ci,
        "excludes_zero": bool(np.percentile(boot_sharpes, lo_pct) > 0 or np.percentile(boot_sharpes, hi_pct) < 0),
    }
