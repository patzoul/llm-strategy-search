"""Third null family: the shift-the-signal skill test from Vincent Maciejewski's
llm-assisted-trading-strategy-search, run against this repo's eight strategies.

    python shiftnull.py

WHY: this repo's two nulls (sign-flip, stationary block bootstrap) randomise the
*data* and re-run the whole fit. His randomises the *alignment* instead: the
asset returns and each config's exposure profile are kept EXACTLY, and only the
correspondence between signal and return is destroyed, by circularly rolling the
position series. That is a different and in one respect stricter idea --
a volatility gate cannot harvest volatility clustering under it, because the
exposure profile is deliberately misaligned with the volatility it was gating on.

Faithful to his implementation (backtest.py::shift_null_pvalue):
  * selection-aware -- `real` and every null draw take the MAX over the SAME
    sampled configs, so "I tried many settings and kept the best" inflates both
    sides and cancels;
  * near-identity shifts excluded (offset drawn from [min_gap, n-min_gap]),
    because a small shift barely moves a slow position and inflates p. min_gap
    must exceed the longest indicator lookback; his default is 250 bars. Here it
    is taken from each strategy's own declared parameter space, so it adapts to
    monthly panels. His fallback to n//3 with a warning is preserved.
  * p = (b+1)/(m+1), not b/m (Phipson & Smyth 2010) -- the floor is 1/(m+1);
  * 1-bar execution lag and cost on turnover applied after the roll, matching
    how the fitness prices trades.

MAPPING onto this repo's conventions. A pair strategy here is scored on active
return against leg B, which is exactly `w * (ret_a - ret_b)`. So the pair case
passes the spread as `asset_ret` and the weight as the position, and doubles the
cost because a switch trades both legs. Single-asset strategies pass their own
returns. In both cases the quantity scored is the same one the CV optimised.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from llmsearch import backtest, data
from llmsearch.panel import Pair, Panel
from run import CONFIGS


def sharpe(r: np.ndarray, ppy: float) -> float:
    r = r[np.isfinite(r)]
    if len(r) < 20:
        return -np.inf
    sd = r.std(ddof=0)
    return float(r.mean() / sd * np.sqrt(ppy)) if sd > 0 else -np.inf


def score_position(pos, asset_ret, cost, ppy):
    """Numerically identical to how this repo's engine prices a position."""
    held = np.empty(len(pos), dtype=float)
    held[0] = 0.0
    held[1:] = pos[:-1]                      # decide t-1, hold t
    turn = np.abs(np.diff(held, prepend=0.0))
    return sharpe(held * asset_ret - cost * turn, ppy)


def make_shift_offsets(n, n_shifts, seed, min_gap):
    gap = int(min_gap)
    if 2 * gap >= n:
        gap = max(1, n // 3)
        warnings.warn(f"n={n} too short for min_gap={min_gap}; using {gap}. Near-identity "
                      f"exclusion is weaker than advertised, so p may be inflated.", stacklevel=2)
    hi = max(gap + 1, n - gap)
    return np.random.default_rng(seed).integers(gap, hi, size=int(n_shifts)), gap


def shift_null(strategy, d, cost_bps, ppy, is_pair, n_configs=256, n_shifts=300, seed=0):
    if is_pair:
        asset_ret = np.asarray(d.ret_a - d.ret_b, dtype=float)   # active return per unit tilt
        cost = 2.0 * cost_bps / 1e4                              # a switch trades both legs
    else:
        asset_ret = np.asarray(d.ret, dtype=float)
        cost = cost_bps / 1e4
    n = len(asset_ret)

    rng = np.random.default_rng(seed)
    lo = np.array([b[0] for b in strategy.bounds])
    hi = np.array([b[1] for b in strategy.bounds])
    pos_list = []
    for _ in range(n_configs):
        x = lo + rng.random(len(lo)) * (hi - lo)
        try:
            pos_list.append(np.asarray(strategy.signal(d, x), dtype=float))
        except Exception:
            pos_list.append(np.zeros(n))

    def maxscore(shift):
        best = -np.inf
        for pos in pos_list:
            pp = pos if shift == 0 else np.roll(pos, int(shift))
            s = score_position(pp, asset_ret, cost, ppy)
            if s > best:
                best = s
        return best

    # min_gap must exceed the longest window the structure can request
    min_gap = int(max([b[1] for p, b in zip(strategy.names, strategy.bounds)
                       if strategy.params[p].kind == "int"] or [10]))
    offsets, used_gap = make_shift_offsets(n, n_shifts, seed + 1, min_gap)
    real = maxscore(0)
    null = np.array([maxscore(k) for k in offsets], dtype=float)
    b, m = int((null >= real).sum()), len(null)
    return dict(real=real, null_mean=float(null.mean()), null_q95=float(np.quantile(null, 0.95)),
                null_max=float(null.max()), pvalue=float((b + 1) / (m + 1)), beat=b, n_shifts=m,
                n_configs=n_configs, min_gap_requested=min_gap, min_gap_used=used_gap, n_bars=n)


def build_insample(key):
    cfg = CONFIGS[key]
    ppy, is_pair = cfg["ppy"], cfg.get("kind") == "pair"
    raw = data.load(cfg["signal_ticker"], start=cfg["start"])
    frame = data.to_monthly(raw) if cfg["freq"] == "M" else raw
    fb = None
    if is_pair:
        fb = data.load(cfg["ticker_b"], start=cfg["start"])
        fb = data.to_monthly(fb) if cfg["freq"] == "M" else fb
        ix = frame.index.intersection(fb.index)
        frame, fb = frame.loc[ix], fb.loc[ix]
    fx = {}
    for tk in (cfg.get("exog_tickers") or ([cfg["exog_ticker"]] if cfg.get("exog_ticker") else [])):
        fxr = data.load(tk, start=cfg["start"])
        fxr = data.to_monthly(fxr) if cfg["freq"] == "M" else fxr
        ix = frame.index.intersection(fxr.index)
        frame = frame.loc[ix]
        if fb is not None:
            fb = fb.loc[ix]
        fx = {k: v.loc[ix] for k, v in fx.items()}
        fx[tk] = fxr.loc[ix]
    m = frame.index <= pd.Timestamp(cfg["split"])
    fxi = {k: v[m] for k, v in fx.items()} or None
    d = (Pair.build(frame[m], fb[m], ppy, key, fx=fxi) if is_pair
         else Panel.from_frame(frame[m], ppy, key, fx=fxi))
    return cfg, d, is_pair


if __name__ == "__main__":
    order = ["btc", "iuse", "ewcw", "valgro", "momlv", "goldbtc", "smlg", "vixts"]
    print(f"{'strategy':<9} {'bars':>5} {'real':>7} {'nullmean':>9} {'nullq95':>8} "
          f"{'nullmax':>8} {'p':>6}  gap(req/used)")
    print("-" * 74)
    rows = {}
    for k in order:
        cfg, d, is_pair = build_insample(k)
        r = shift_null(cfg["strategy"](), d, cfg["cost_bps"], cfg["ppy"], is_pair)
        rows[k] = r
        print(f"{k:<9} {r['n_bars']:>5} {r['real']:>7.3f} {r['null_mean']:>9.3f} "
              f"{r['null_q95']:>8.3f} {r['null_max']:>8.3f} {r['pvalue']:>6.3f}  "
              f"{r['min_gap_requested']}/{r['min_gap_used']}", flush=True)
    import json
    json.dump(rows, open("results/shiftnull.json", "w"), indent=1, default=str)
    print("\nwrote results/shiftnull.json")
