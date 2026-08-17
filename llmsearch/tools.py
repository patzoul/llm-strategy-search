"""Causal indicator library.

Every function here is strictly backward-looking: the value at index t uses only
data at or before t. Nothing in a strategy may touch raw prices directly -- it
composes these. That is the article's constraint and it is what makes the
look-ahead audit in selfcheck.py a two-line test instead of a code review.

Everything is float64 numpy, not pandas. The whole library sits inside a
differential-evolution inner loop that calls it tens of thousands of times per
fit, and pandas' per-operation overhead (index alignment, `__finalize__`, block
manager dispatch) was ~60% of the runtime. Series are reconstructed once, at
reporting time.

Windows are integers in *bars* -- days on the BTC daily panel, months on the
IUSE monthly panel. Results are cached per (function, window, array
fingerprint), which pays off because `spec.Param` quantises integer parameters
onto a coarse grid, so the optimiser revisits the same windows constantly.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import lfilter

try:
    import bottleneck as bn
except ImportError:                      # pure-numpy fallback, far slower on pctile
    bn = None

_CACHE: dict = {}
_MAXCACHE = 40000
_CTX = [0]

Arr = np.ndarray


def set_context(tok) -> None:
    """Namespace the cache to one dataset.

    Caching on `id(array)` would be wrong: CPython recycles ids, so a surrogate
    series could silently collide with the real one. Instead each dataset gets a
    token, and within a token an array is identified by a cheap fingerprint --
    length plus both endpoints, distinct enough that close, returns and any
    derived array in the same panel never collide.
    """
    if _CTX[0] != tok:
        _CACHE.clear()
        _CTX[0] = tok


def _f(v) -> float:
    v = float(v)
    return 0.0 if v != v else v


def _cached(name, a: Arr, *args, fn=None):
    # The fingerprint must include a whole-array term. Length plus endpoints is
    # not enough: `pctile(dy)` and `pctile(abs(dy))` share all three whenever the
    # last value is positive, so the absolute-value series was silently served
    # the signed series' cached result. nansum separates them, costs a few
    # microseconds, and the look-ahead test in selfcheck.py is what caught it.
    fp = (_CTX[0], len(a), _f(a[0]) if len(a) else 0.0, _f(a[-1]) if len(a) else 0.0,
          _f(np.nansum(a)) if len(a) else 0.0)
    k = (name, fp, args)
    hit = _CACHE.get(k)
    if hit is not None:
        return hit
    out = fn()
    out.flags.writeable = False           # cached arrays are shared; never mutate in place
    if len(_CACHE) > _MAXCACHE:
        _CACHE.clear()
    _CACHE[k] = out
    return out


def clear_cache():
    _CACHE.clear()


def _nan_head(a: Arr, k: int) -> Arr:
    if k > 0:
        a[:min(k, len(a))] = np.nan
    return a


def _ewm(x: Arr, alpha: float) -> Arr:
    """Recursive mean, pandas `ewm(adjust=False)` semantics, seeded with x[0]."""
    x = np.nan_to_num(np.asarray(x, dtype=float), nan=0.0)
    if len(x) == 0:
        return x
    zi = np.array([(1.0 - alpha) * x[0]])
    y, _ = lfilter([alpha], [1.0, -(1.0 - alpha)], x, zi=zi)
    return y


# --------------------------------------------------------------------------
# trend / level
# --------------------------------------------------------------------------

def sma(s: Arr, n: int) -> Arr:
    n = max(1, int(n))

    def _g():
        if bn is not None:
            return bn.move_mean(s, window=n, min_count=n)
        c = np.concatenate(([0.0], np.cumsum(np.nan_to_num(s))))
        out = np.full(len(s), np.nan)
        out[n - 1:] = (c[n:] - c[:-n]) / n
        return _nan_head(out, n - 1)

    return _cached("sma", s, n, fn=_g)


def ema(s: Arr, n: int) -> Arr:
    n = max(1, int(n))
    return _cached("ema", s, n, fn=lambda: _nan_head(_ewm(s, 2.0 / (n + 1.0)), n - 1))


def roc(s: Arr, n: int) -> Arr:
    """Rate of change over n bars, as a fraction."""
    n = max(1, int(n))

    def _g():
        out = np.full(len(s), np.nan)
        if len(s) > n:
            with np.errstate(invalid="ignore", divide="ignore"):
                out[n:] = s[n:] / s[:-n] - 1.0
        return out

    return _cached("roc", s, n, fn=_g)


def slope(s: Arr, n: int) -> Arr:
    """Per-bar log slope of an n-bar OLS fit."""
    n = max(3, int(n))

    def _g():
        y = np.log(np.asarray(s, dtype=float))
        x = np.arange(n, dtype=float)
        xc = x - x.mean()
        out = np.full(len(y), np.nan)
        if len(y) >= n:
            win = np.lib.stride_tricks.sliding_window_view(y, n)
            out[n - 1:] = (win - win.mean(axis=1, keepdims=True)) @ xc / (xc ** 2).sum()
        return out

    return _cached("slope", s, n, fn=_g)


# --------------------------------------------------------------------------
# oscillators
# --------------------------------------------------------------------------

def rsi(s: Arr, n: int) -> Arr:
    """Wilder RSI in [0, 100]."""
    n = max(2, int(n))

    def _g():
        d = np.diff(s, prepend=np.nan)
        up = np.where(np.isnan(d), 0.0, np.maximum(d, 0.0))
        dn = np.where(np.isnan(d), 0.0, np.maximum(-d, 0.0))
        a = 1.0 / n
        au, ad = _ewm(up, a), _ewm(dn, a)
        with np.errstate(invalid="ignore", divide="ignore"):
            out = np.where(ad > 0.0, 100.0 - 100.0 / (1.0 + au / ad),
                           np.where(au > 0.0, 100.0, 50.0))
        return _nan_head(out.astype(float), n)

    return _cached("rsi", s, n, fn=_g)


def zscore(s: Arr, n: int) -> Arr:
    """(x - rolling mean) / rolling sd."""
    n = max(2, int(n))

    def _g():
        m = sma(s, n)
        sd = _rolling_std(s, n)
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(sd > 0, (s - m) / sd, np.nan)

    return _cached("z", s, n, fn=_g)


def bollinger_pos(s: Arr, n: int) -> Arr:
    """Position of price inside its own n-bar band, in sd units."""
    return zscore(s, n)


def breakout(s: Arr, n: int) -> Arr:
    """Where price sits in its trailing n-bar range, in [0, 1].

    The current bar is part of its own range, which is causal: a new high is
    known at the close that makes it.
    """
    n = max(2, int(n))

    def _g():
        if bn is not None:
            hi = bn.move_max(s, window=n, min_count=n)
            lo = bn.move_min(s, window=n, min_count=n)
        else:
            w = np.lib.stride_tricks.sliding_window_view(s, n)
            hi = np.full(len(s), np.nan); lo = np.full(len(s), np.nan)
            hi[n - 1:] = w.max(axis=1); lo[n - 1:] = w.min(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(hi > lo, (s - lo) / (hi - lo), np.nan)

    return _cached("brk", s, n, fn=_g)


# --------------------------------------------------------------------------
# risk / regime
# --------------------------------------------------------------------------

def _rolling_std(s: Arr, n: int) -> Arr:
    if bn is not None:
        return bn.move_std(s, window=n, min_count=n, ddof=0)
    w = np.lib.stride_tricks.sliding_window_view(s, n)
    out = np.full(len(s), np.nan)
    out[n - 1:] = w.std(axis=1)
    return out


def rvol(ret: Arr, n: int, ann: float = 1.0) -> Arr:
    """Realised volatility of a return series over n bars."""
    n = max(2, int(n))
    return _cached("rvol", ret, n, ann, fn=lambda: _rolling_std(ret, n) * np.sqrt(ann))


def vol_ratio(ret: Arr, n_fast: int, n_slow: int) -> Arr:
    """Fast vol over slow vol. >1 = volatility is picking up."""
    f, s = rvol(ret, n_fast), rvol(ret, n_slow)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(s > 0, f / s, np.nan)


def pctile(s: Arr, n: int) -> Arr:
    """Rank of the current value within its own trailing n-bar window, in [0,1].

    This is the regime classifier the article leans on: it turns any raw
    indicator into a self-normalising one, so a threshold fitted on 2015 data
    still means something in 2025.
    """
    n = max(5, int(n))

    def _g():
        if bn is not None:
            # move_rank is normalised to [-1, 1] with ties averaged; rescale.
            return (bn.move_rank(s, window=n, min_count=n) + 1.0) / 2.0
        out = np.full(len(s), np.nan)
        if len(s) >= n:
            w = np.lib.stride_tricks.sliding_window_view(s, n)
            last = w[:, -1][:, None]
            valid = ~np.isnan(w)
            cnt = valid.sum(axis=1)
            less = np.nansum((w < last) & valid, axis=1)
            ties = np.nansum((w == last) & valid, axis=1) - 1
            with np.errstate(invalid="ignore", divide="ignore"):
                out[n - 1:] = np.where(cnt > 1, (less + 0.5 * ties) / np.maximum(cnt - 1, 1),
                                       np.nan)
        return out

    return _cached("pct", s, n, fn=_g)


def drawdown(s: Arr) -> Arr:
    """Current drawdown from the running peak, <= 0."""
    return _cached("dd", s, fn=lambda: s / np.maximum.accumulate(s) - 1.0)


# --------------------------------------------------------------------------
# combination primitives
# --------------------------------------------------------------------------

def gate(x: Arr, thresh: float, width: float = 0.0) -> Arr:
    """Soft step: 0 well below `thresh`, 1 well above. width=0 -> hard step.

    A soft gate matters for fitting: a hard step makes the objective piecewise
    constant, which differential evolution can still handle but explores badly.
    """
    if width <= 1e-9:
        return np.where(np.isnan(x), np.nan, (x > thresh).astype(float))
    z = np.clip((x - thresh) / width, -60.0, 60.0)
    return np.where(np.isnan(x), np.nan, 1.0 / (1.0 + np.exp(-z)))


def absv(x: Arr) -> Arr:
    """Magnitude, ignoring direction -- for 'how big is the move' regime tests."""
    return np.abs(x)


def blend(a: Arr, b: Arr, w: Arr) -> Arr:
    """w*a + (1-w)*b -- the regime switch, with w usually coming from gate()."""
    return w * a + (1.0 - w) * b


def clip_signal(s: Arr, lo: float = -1.0, hi: float = 1.0) -> Arr:
    """Final call of every strategy. NaN -> flat, and hard bounds on leverage."""
    return np.clip(np.nan_to_num(np.asarray(s, dtype=float), nan=0.0,
                                 posinf=hi, neginf=lo), lo, hi)
