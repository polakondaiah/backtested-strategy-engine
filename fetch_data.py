"""
Fetch daily OHLCV for NSE large-cap universe.
Tries yfinance; falls back to deterministic synthetic data if offline.
Usage: python fetch_data.py [--synthetic]
"""
import argparse
import sys
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd

UNIVERSE = [
    "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS",
    "ICICIBANK.NS", "SBIN.NS", "BHARTIARTL.NS", "ITC.NS",
]
START = "2020-01-01"
END = "2026-08-20"
DATA_DIR = Path(__file__).parent / "data"
OUT_CSV = DATA_DIR / "prices.csv"
OUT_PARQUET = DATA_DIR / "prices.parquet"

def generate_synthetic():
    """Deterministic GBM synthetic data - reproducible, no network needed."""
    rng = np.random.default_rng(42)
    # Separate RNG stream for volume, seeded independently so fixing the
    # volume model can't silently shift the price series (and every
    # downstream Sharpe/IC/regression number derived from it) -- volume and
    # price generation are logically independent draws and should be
    # independently reproducible.
    vol_rng = np.random.default_rng(4200)
    dates = pd.bdate_range(START, END)  # business days only
    n = len(dates)
    rows = []
    for ticker in UNIVERSE:
        # different drift/vol per ticker
        mu = rng.uniform(0.05, 0.15) / 252
        sigma = rng.uniform(0.15, 0.35) / np.sqrt(252)
        s0 = rng.uniform(800, 3500)
        rets = rng.normal(mu, sigma, n)
        # inject some mean-reversion structure via OU overlay for 2 tickers
        prices = s0 * np.exp(np.cumsum(rets))
        # Volume: base level per ticker (random draw, so cross-sectional
        # variation exists) x day-to-day lognormal noise x a volume-follows-
        # volatility bump on days with large |return| (a standard empirical
        # regularity -- turnover spikes on big move days). Previously this
        # scaled with price*sigma directly and was clipped to a floor that
        # swallowed all cross-sectional/day-to-day variation, producing a
        # constant Volume column (std=0) that silently broke any signal
        # built on it (e.g. volume z-score) -- fixed here.
        base_vol = vol_rng.uniform(2_000_000, 15_000_000)
        daily_noise = np.exp(vol_rng.normal(0, 0.35, n))
        move_bump = 1 + 3 * np.abs(rets - mu) / sigma  # sigma is a per-ticker scalar > 0
        vol = base_vol * daily_noise * move_bump
        vol = np.clip(vol.astype(int), 500_000, 80_000_000)
        for i, d in enumerate(dates):
            o = prices[i] * rng.uniform(0.99, 1.01)
            h = max(o, prices[i]) * rng.uniform(1.0, 1.015)
            l = min(o, prices[i]) * rng.uniform(0.985, 1.0)
            c = prices[i]
            rows.append([d, ticker, o, h, l, c, vol[i]])
    df = pd.DataFrame(rows, columns=["Date","Ticker","Open","High","Low","Close","Volume"])
    return df

def fetch_yfinance():
    import yfinance as yf
    print(f"[fetch] Attempting yfinance for {UNIVERSE} from {START} to {END}")
    df = yf.download(UNIVERSE, start=START, end=END, auto_adjust=False, progress=False, group_by="ticker", threads=True)
    if df.empty:
        raise ValueError("yfinance returned empty dataframe")
    # yfinance with group_by ticker returns MultiIndex columns
    rows = []
    for ticker in UNIVERSE:
        try:
            sub = df[ticker] if ticker in df.columns.get_level_values(0) else df
            if sub.empty or "Close" not in sub.columns:
                continue
            sub = sub.reset_index()
            for _, r in sub.iterrows():
                if pd.isna(r["Close"]):
                    continue
                rows.append([r["Date"], ticker, r["Open"], r["High"], r["Low"], r["Close"], r["Volume"]])
        except Exception as e:
            print(f"[warn] {ticker}: {e}", file=sys.stderr)
    if not rows:
        raise ValueError("No rows parsed from yfinance")
    return pd.DataFrame(rows, columns=["Date","Ticker","Open","High","Low","Close","Volume"])

def validate(df: pd.DataFrame):
    print(f"[validate] Rows: {len(df):,} | Tickers: {df['Ticker'].nunique()} | Date range: {df['Date'].min()} -> {df['Date'].max()}")
    # missing dates check
    expected = pd.bdate_range(START, END)
    for ticker in UNIVERSE:
        sub = df[df["Ticker"]==ticker]
        missing = len(expected) - len(sub["Date"].unique())
        if missing > len(expected)*0.1:
            print(f"[warn] {ticker} missing {missing} business days (holidays + gaps expected ~10-15)")
    # null check
    nulls = df.isna().sum().sum()
    print(f"[validate] Null values: {nulls}")
    # duplicates
    dups = df.duplicated(subset=["Date","Ticker"]).sum()
    print(f"[validate] Duplicate Date/Ticker: {dups}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthetic", action="store_true", help="force synthetic data")
    args = parser.parse_args()
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if args.synthetic:
        print("[fetch] Forced synthetic mode")
        df = generate_synthetic()
        source = "synthetic (forced)"
    else:
        try:
            df = fetch_yfinance()
            source = "yfinance"
        except Exception as e:
            print(f"[fetch] yfinance failed ({e}), falling back to synthetic", file=sys.stderr)
            df = generate_synthetic()
            source = f"synthetic (fallback: {e})"

    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values(["Ticker","Date"]).reset_index(drop=True)
    validate(df)

    df.to_csv(OUT_CSV, index=False)
    try:
        df.to_parquet(OUT_PARQUET, index=False)
    except Exception as e:
        print(f"[warn] parquet write failed ({e}), csv only", file=sys.stderr)

    print(f"[done] Wrote {len(df):,} rows to {OUT_CSV} (source: {source})")
    # write metadata
    meta = DATA_DIR / "fetch_meta.txt"
    meta.write_text(f"source: {source}\nrows: {len(df)}\nrange: {df['Date'].min()} to {df['Date'].max()}\nwritten: {datetime.now().isoformat()}\n")

if __name__ == "__main__":
    main()
