"""Strategy signals: mean-reversion z-score and momentum rank."""
import pandas as pd
import numpy as np

def compute_zscore(close: pd.Series, window: int = 20) -> pd.Series:
    """z = (price - rolling_mean) / rolling_std. NaN for first window-1."""
    ma = close.rolling(window).mean()
    std = close.rolling(window).std(ddof=0)
    z = (close - ma) / std
    return z

def compute_momentum(close: pd.Series, window: int = 20) -> pd.Series:
    """Simple momentum: return over window."""
    return close.pct_change(window)

def add_signals(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """
    Input long df with columns Date,Ticker,Close.
    Returns df with zscore, momentum columns (per ticker).
    """
    df = df.sort_values(["Ticker","Date"]).copy()
    df["zscore"] = df.groupby("Ticker")["Close"].transform(lambda s: compute_zscore(s, window))
    df["momentum"] = df.groupby("Ticker")["Close"].transform(lambda s: compute_momentum(s, window))
    # position: mean-reversion -> short high z, long low z
    # clip z to [-2,2] and invert: pos = -z/2 clipped to [-1,1]
    df["pos_raw"] = (-df["zscore"] / 2.0).clip(-1, 1)
    return df
