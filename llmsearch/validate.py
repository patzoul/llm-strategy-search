"""Validation: blocked-quarter CV, null bars, holdout, deflated Sharpe.

The article's three layers, with one addition.

  1. Blocked-quarter CV -- fit on a random 75% of quarters, score on the other
     25%, many times. Quarter blocks (not random days) so that a test bar is not
     sandwiched between its own training neighbours.
  2. Null bar -- run the *identical* fit-and-validate procedure on surrogate
     series that contain no exploitable structure, and see how good a CV score
     the procedure achieves on pure noise. The real CV score has to beat that.
     This is the honest answer to "how many things did you try": it measures the
     whole search, not one backtest.
  3. Holdout -- a date-forward slice touched exactly once, at the end.

  (addition) Two surrogate families, not one. The article sign-flips returns,
  which removes the asset's drift as well as its structure -- a bar that is too
  easy for a long-only strategy on a rising asset. So a stationary block
  bootstrap is run alongside it: it keeps the drift and the volatility
  clustering and destroys only the ordering. The strategy has to clear both.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats

from . import backtest, fit, tools
from .panel import Panel


# --------------------------------------------------------------------------
# blocked-quarter cross-validation
# --------------------------------------------------------------------------

@dataclass
class CV:
    train: np.ndarray
    test: np.ndarray
    xs: list = field(default_factory=list)

    @property
    def mean_test(self) -> float:
        return float(np.nanmean(self.test))

    @property
    def median_test(self) -> float:
        return float(np.nanmedian(self.test))

    @property
    def frac_positive(self) -> float:
        return float(np.nanmean(self.test > 0))


def blocked_cv(strategy, d: Panel, cost_bps: float, n_splits: int = 40,
               train_frac: float = 0.75, seed: int = 0, budget: str = "normal",
               keep_x: bool = True) -> CV:
    q = d.index.to_period("Q").to_numpy()
    uq = np.unique(q)
    rng = np.random.default_rng(seed)
    n_tr = max(2, int(round(train_frac * len(uq))))

    tr_s, te_s, xs = [], [], []
    for i in range(n_splits):
        perm = rng.permutation(len(uq))
        tr_mask = np.isin(q, uq[perm[:n_tr]])
        f = fit.fit(strategy, d, cost_bps, mask=tr_mask, seed=seed * 1000 + i,
                    budget=budget)
        s_te = backtest.score(d, strategy.signal(d, f.x), cost_bps, ~tr_mask)
        tr_s.append(f.train_score if np.isfinite(f.train_score) else np.nan)
        te_s.append(s_te if np.isfinite(s_te) else np.nan)
        if keep_x:
            xs.append(f.x)
    return CV(train=np.array(tr_s, dtype=float), test=np.array(te_s, dtype=float), xs=xs)


# --------------------------------------------------------------------------
# surrogates
# --------------------------------------------------------------------------

def _rebuild(d: Panel, r: np.ndarray) -> Panel:
    """Rebuild a full panel from a surrogate return path.

    Highs/lows are scaled by the same per-bar factor the close moved by, so
    range-based indicators still see a plausible bar shape.
    """
    r = np.asarray(r, dtype=float).copy()
    r[0] = 0.0                      # anchor the path: bar 0 has no return
    close = float(d.close[0]) * np.cumprod(1.0 + r)
    scale = close / d.close
    df = pd.DataFrame({"Close": close, "High": d.high * scale, "Low": d.low * scale,
                       "Open": d.open * scale, "Volume": d.volume}, index=d.index)
    return Panel.from_frame(df, d.ppy, d.name + "~surrogate")


def _block_idx(n: int, block: int, rng: np.random.Generator) -> np.ndarray:
    """Index path for a stationary block bootstrap."""
    out = np.empty(n, dtype=int)
    i = 0
    while i < n:
        L = min(1 + rng.geometric(1.0 / block), n - i)
        out[i:i + L] = (rng.integers(0, n) + np.arange(L)) % n
        i += L
    return out


def _rebuild_leg(p: Panel, r: np.ndarray) -> pd.DataFrame:
    r = np.asarray(r, dtype=float).copy()
    r[0] = 0.0
    close = float(p.close[0]) * np.cumprod(1.0 + r)
    scale = close / p.close
    f = pd.DataFrame({"Close": close, "High": p.high * scale, "Low": p.low * scale,
                      "Open": p.open * scale, "Volume": p.volume}, index=p.index)
    f.attrs["ticker"] = p.name
    return f


def surrogate_pair(d, mode: str, rng: np.random.Generator, block: int = 3):
    """Surrogates for a rotation strategy.

    A rotation rule only ever earns `w * (ret_a - ret_b)`. So the null must
    destroy the predictability of that **spread** while leaving the common
    equity market — which the strategy neither controls nor is being credited
    for — completely intact. Applying the single-asset nulls to each leg
    independently would break the correlation between them and test the wrong
    hypothesis entirely.

    `signflip`: decompose each bar into a common part m=(ra+rb)/2 and a spread
      part s=(ra-rb)/2, then flip the sign of s in blocks. The market factor is
      untouched bar for bar; only *which leg won* is randomised. Under this null
      the expected active return of any causal rule is exactly zero, which makes
      it the sharpest possible test of the rotation decision.
    `blockboot`: resample (ra, rb) *jointly* in blocks — same time indices for
      both legs. Preserves each leg's drift, their contemporaneous correlation
      and the volatility clustering; destroys only the time ordering, so no
      timing rule can work but a constant tilt keeps whatever it historically
      earned.
    """
    ra, rb = np.asarray(d.ret_a, float), np.asarray(d.ret_b, float)
    n = len(ra)
    if mode == "signflip":
        m, s = (ra + rb) / 2.0, (ra - rb) / 2.0
        nb = int(np.ceil(n / block))
        eps = np.repeat(rng.choice([-1.0, 1.0], size=nb), block)[:n]
        ra2, rb2 = m + eps * s, m - eps * s
    elif mode == "blockboot":
        idx = _block_idx(n, block, rng)
        ra2, rb2 = ra[idx], rb[idx]
    else:
        raise ValueError(mode)
    # `with_legs` carries any exogenous series through untouched and in its
    # original time order -- that is what makes this a null for the *conditional*
    # hypothesis "the exogenous series predicts the spread".
    return d.with_legs(_rebuild_leg(d.a, ra2), _rebuild_leg(d.b, rb2),
                       d.name + "~surrogate")


def surrogate(d, mode: str, rng: np.random.Generator, block: int = 21):
    """`signflip`: random sign flips in blocks -- kills drift and direction,
    keeps the volatility clustering.
    `blockboot`: stationary block bootstrap -- keeps drift *and* clustering,
    kills only the time ordering. The harder, more honest bar for a long-only
    strategy on a rising asset.
    """
    if backtest.is_pair(d):
        return surrogate_pair(d, mode, rng, block)
    r = np.asarray(d.ret, dtype=float)
    n = len(r)
    if mode == "signflip":
        nb = int(np.ceil(n / block))
        s = np.repeat(rng.choice([-1.0, 1.0], size=nb), block)[:n]
        return _rebuild(d, r * s)
    if mode == "blockboot":
        return _rebuild(d, r[_block_idx(n, block, rng)])
    raise ValueError(mode)


@dataclass
class NullBar:
    mode: str
    scores: np.ndarray

    @property
    def bar_max(self) -> float:
        return float(np.nanmax(self.scores))

    @property
    def bar_p95(self) -> float:
        return float(np.nanpercentile(self.scores, 95))

    @property
    def mean(self) -> float:
        return float(np.nanmean(self.scores))


def null_bar(strategy, d: Panel, cost_bps: float, mode: str, n_null: int = 12,
             n_splits: int = 40, seed: int = 0, budget: str = "normal",
             block: int = 21, progress=None, resume: list | None = None) -> NullBar:
    """Run the whole fit-and-CV pipeline on `n_null` structureless surrogates.

    `resume` is a list of scores from surrogates already completed in an earlier
    process. Their surrogates are still *generated* so the RNG stream advances
    exactly as it would have, which makes a resumed run bit-identical to an
    uninterrupted one -- otherwise the resumed surrogates would be drawn from a
    different part of the stream and the bar would not be the one the completed
    scores belong to.
    """
    rng = np.random.default_rng(seed + 77)
    out = list(resume or [])
    for i in range(n_null):
        sd = surrogate(d, mode, rng, block=block)
        if i < len(out):
            continue
        cv = blocked_cv(strategy, sd, cost_bps, n_splits=n_splits,
                        seed=seed * 31 + i, budget=budget, keep_x=False)
        out.append(cv.mean_test)
        if progress:
            progress(mode, i + 1, n_null, cv.mean_test)
    tools.clear_cache()
    return NullBar(mode=mode, scores=np.array(out, dtype=float))


def null_pvalue(observed: float, nb: NullBar) -> float:
    """Fraction of surrogate runs that matched or beat the real CV score."""
    s = nb.scores[np.isfinite(nb.scores)]
    if len(s) == 0:
        return np.nan
    return float((np.sum(s >= observed) + 1) / (len(s) + 1))


# --------------------------------------------------------------------------
# reporting helpers
# --------------------------------------------------------------------------

def deflated_sharpe(sr: float, n_obs: int, n_trials: int, skew: float = 0.0,
                    kurt: float = 3.0, ppy: float = 252.0,
                    var_trials: float | None = None) -> float:
    """Bailey & Lopez de Prado DSR: P(true Sharpe > 0) given `n_trials` tried.

    `sr` is annualised on the way in and de-annualised here. `var_trials` is the
    variance of the Sharpes across the trials that were run; the usual default
    1/(n_obs-1) is the sampling variance of a zero-edge Sharpe estimate.
    """
    if not np.isfinite(sr) or n_obs < 10 or n_trials < 1:
        return np.nan
    sr_b = sr / np.sqrt(ppy)
    v = (1.0 / (n_obs - 1)) if var_trials is None else max(var_trials, 1e-12)
    e = 0.5772156649
    m = max(n_trials, 2)
    z = stats.norm.ppf(1 - 1.0 / m)
    z2 = stats.norm.ppf(1 - 1.0 / (m * np.e))
    sr0 = np.sqrt(v) * ((1 - e) * z + e * z2)   # expected max from m zero-edge trials
    denom = np.sqrt(max(1e-12, 1 - skew * sr_b + (kurt - 1) / 4.0 * sr_b ** 2))
    return float(stats.norm.cdf((sr_b - sr0) * np.sqrt(n_obs - 1) / denom))


def buy_and_hold(d, weight: float = 1.0) -> backtest.Result:
    """Constant-weight benchmark. For a pair, `weight` is the fixed tilt to leg A
    (1.0 = always equal-weight, 0.0 = always cap-weight, 0.5 = a fixed blend)."""
    w = np.full(len(d), float(weight))
    if backtest.is_pair(d):
        return backtest.run_pair(d, w, cost_bps=0.0)
    return backtest.run_panel(d, w, cost_bps=0.0)


def consensus(cv: CV) -> np.ndarray:
    """The parameter vector actually deployed: the per-coordinate median over
    the CV fits, restricted to the splits that generalised (positive test score).

    Taking the single best split's parameters would be selecting on the very
    thing the CV is supposed to be measuring. The median across splits is the
    part of the fit that was stable rather than split-specific.
    """
    ok = [x for x, s in zip(cv.xs, cv.test) if np.isfinite(s) and s > 0]
    pool = ok if len(ok) >= max(3, 0.2 * len(cv.xs)) else cv.xs
    return np.median(np.vstack(pool), axis=0)
