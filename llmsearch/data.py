"""Data loading with an integrity check at the loader boundary.

Nothing downstream is allowed to see a frame that has not been through
`integrity_report`. Two prior projects here died of data artifacts that a check
at this exact boundary would have caught, so the report is printed, not just
computed.
"""

from __future__ import annotations

import os
import time

import numpy as np
import pandas as pd

CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data_cache")


def load(ticker: str, start: str = "1990-01-01", end: str | None = None,
         use_cache: bool = True) -> pd.DataFrame:
    """Daily OHLCV, split/dividend adjusted, indexed by date (tz-naive)."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    # The cache key MUST include the requested start. Keyed on ticker alone, a
    # run asking for SPY from 2000 would silently be handed a file another run
    # had fetched from 2013, and the only symptom is a short panel -- which is
    # exactly what happened once. Date alone cannot detect it after the fact,
    # because a series legitimately starting late (an ETF's inception) looks
    # identical to a truncated cache.
    safe = "".join(c if c.isalnum() or c in "-._" else "_" for c in ticker)
    path = os.path.join(CACHE_DIR, f"{safe}__from{start}.csv")

    if use_cache and os.path.exists(path) and time.time() - os.path.getmtime(path) < 12 * 3600:
        df = pd.read_csv(path, index_col=0, parse_dates=True)
    else:
        import yfinance as yf
        df = yf.download(ticker, start=start, end=end, auto_adjust=True,
                         progress=False, actions=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        if df.empty:
            raise RuntimeError(f"no data returned for {ticker}")
        df.to_csv(path)

    df.index = pd.to_datetime(df.index).tz_localize(None)
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df = df[[c for c in ("Open", "High", "Low", "Close", "Volume") if c in df.columns]]
    df = df[df["Close"] > 0].dropna(subset=["Close"])
    if end is not None:
        df = df[df.index <= pd.Timestamp(end)]
    df = df[df.index >= pd.Timestamp(start)]
    df.attrs["ticker"] = ticker
    return df


def integrity_report(df: pd.DataFrame, name: str = "", spike: float = 1.0,
                     max_gap_days: int = 10) -> dict:
    """Facts about the panel, plus a `flags` list of things worth stopping for."""
    c = df["Close"]
    r = c.pct_change().dropna()
    gaps = df.index.to_series().diff().dt.days.dropna()
    rep = {
        "name": name or df.attrs.get("ticker", "?"),
        "start": df.index[0].date(),
        "end": df.index[-1].date(),
        "n_bars": len(df),
        "years": (df.index[-1] - df.index[0]).days / 365.25,
        "max_ret": float(r.max()),
        "min_ret": float(r.min()),
        "ann_vol": float(r.std(ddof=0) * np.sqrt(252)),
        "n_zero_ret": int((r == 0).sum()),
        "max_gap_days": int(gaps.max()),
        "n_nan_close": int(c.isna().sum()),
        "flags": [],
    }
    if rep["max_ret"] > spike:
        rep["flags"].append(f"max daily return {rep['max_ret']:.1%} > {spike:.0%} -- verify, not a result")
    if rep["min_ret"] < -spike:
        rep["flags"].append(f"min daily return {rep['min_ret']:.1%} < -{spike:.0%} -- verify")
    if rep["max_gap_days"] > max_gap_days:
        rep["flags"].append(f"calendar gap of {rep['max_gap_days']}d in the series")
    if rep["ann_vol"] > 3.0 or rep["ann_vol"] < 0.01:
        rep["flags"].append(f"annualised vol {rep['ann_vol']:.1%} implausible")
    if rep["n_zero_ret"] > 0.05 * len(r):
        rep["flags"].append(f"{rep['n_zero_ret']} zero-return bars ({rep['n_zero_ret']/len(r):.1%}) -- stale prices?")
    if rep["n_nan_close"]:
        rep["flags"].append(f"{rep['n_nan_close']} NaN closes")
    return rep


def print_report(rep: dict) -> None:
    print(f"  {rep['name']:10s} {rep['start']} -> {rep['end']}  "
          f"{rep['n_bars']:>5d} bars / {rep['years']:.1f}y   "
          f"vol {rep['ann_vol']:.1%}  ret range [{rep['min_ret']:+.1%}, {rep['max_ret']:+.1%}]  "
          f"maxgap {rep['max_gap_days']}d")
    for f in rep["flags"]:
        print(f"     !! {f}")


def assert_clean(rep: dict) -> None:
    if rep["flags"]:
        raise AssertionError(f"{rep['name']}: integrity flags -> {rep['flags']}")


def to_monthly(df: pd.DataFrame) -> pd.DataFrame:
    """Month-end bars. Close = last close of the month, Volume = month total.

    The index is the actual last *trading* day of the month, not a synthetic
    calendar month-end, so 'decide at this close, trade at the next' stays true.
    """
    per = df.index.to_period("M")
    pos = pd.Series(np.arange(len(df)), index=per)
    first_pos = pos.groupby(level=0).first().to_numpy()
    last_pos = pos.groupby(level=0).last().to_numpy()

    close = df["Close"].to_numpy()
    out = pd.DataFrame(index=df.index[last_pos])
    out.index.name = "Date"
    out["Open"] = (df["Open"].to_numpy() if "Open" in df else close)[first_pos]
    out["Close"] = close[last_pos]
    grp = df.groupby(per, sort=True)
    out["High"] = (grp["High"].max() if "High" in df else grp["Close"].max()).to_numpy()
    out["Low"] = (grp["Low"].min() if "Low" in df else grp["Close"].min()).to_numpy()
    out["Volume"] = grp["Volume"].sum().to_numpy() if "Volume" in df else np.nan
    out.attrs["ticker"] = df.attrs.get("ticker", "?")
    return out.dropna(subset=["Close"])
