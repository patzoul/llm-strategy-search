"""Invariants for the framework. Run this before trusting any number it prints.

    python selfcheck.py

The one that matters most is LOOK-AHEAD: it rewrites the entire future of the
price series and asserts the signal history up to the cut is byte-identical. If
any indicator peeks, that test fails and every result in this repo is void.
"""

from __future__ import annotations

import sys
import time
import traceback

import numpy as np
import pandas as pd

from llmsearch import backtest, data, fit, tools, validate
from llmsearch.panel import Panel
from llmsearch.spec import Param, Strategy
from strategies.btc_regime import BtcVolRegime
from strategies.ew_cw_rotation import EwCwRotation
from strategies.gold_btc import GoldBtcRotation
from strategies.iuse_monthly import IuseMonthlyTrend
from strategies.mom_lowvol import MomLowVolRotation
from strategies.value_growth import ValueGrowthRotation

PASS, FAIL = [], []


def check(name):
    def deco(fn):
        try:
            fn()
            PASS.append(name)
            print(f"  ok    {name}", flush=True)
        except Exception as e:
            FAIL.append((name, e))
            print(f"  FAIL  {name}: {e}", flush=True)
            traceback.print_exc()
        return fn
    return deco


def synth_frame(n=3000, seed=0):
    rng = np.random.default_rng(seed)
    r = rng.normal(0.0004, 0.02, n)
    idx = pd.bdate_range("2005-01-03", periods=n)
    c = 100 * np.cumprod(1 + r)
    return pd.DataFrame({"Open": c, "High": c * 1.01, "Low": c * 0.99,
                         "Close": c, "Volume": 1e6}, index=idx)


def synth(n=3000, seed=0, ppy=252):
    return Panel.from_frame(synth_frame(n, seed), ppy, f"synth{seed}")


D = synth()
S = BtcVolRegime()
X = np.array([np.mean(b) for b in S.bounds])


# ---------------------------------------------------------------- 1
@check("look-ahead: future prices cannot change past signal")
def _():
    cut = 2000
    base_sig = S.signal(D, X)[:cut]
    f = synth_frame()
    for seed in (1, 2):
        alt = f.copy()
        rng = np.random.default_rng(seed)
        tail = rng.normal(0.01, 0.05, len(alt) - cut)
        newc = float(alt["Close"].iloc[cut - 1]) * np.cumprod(1 + tail)
        for col, mult in (("Close", 1.0), ("Open", 1.0), ("High", 1.01), ("Low", 0.99)):
            alt.iloc[cut:, alt.columns.get_loc(col)] = newc * mult
        got = S.signal(Panel.from_frame(alt, 252, f"alt{seed}"), X)[:cut]
        assert np.array_equal(base_sig, got, equal_nan=True), "signal changed"


# ---------------------------------------------------------------- 2
@check("execution lag: position over bar t is the signal from bar t-1")
def _():
    sig = np.zeros(len(D)); sig[100] = 1.0
    res = backtest.run_panel(D, sig, 0.0)
    assert res.pos[100] == 0.0 and res.pos[101] == 1.0 and res.pos[102] == 0.0


# ---------------------------------------------------------------- 3
@check("buy&hold: sig=1 with zero cost reproduces the asset exactly")
def _():
    res = validate.buy_and_hold(D)
    assert abs(res.equity[-1] / (D.close[-1] / D.close[0]) - 1) < 1e-9


# ---------------------------------------------------------------- 4
@check("costs are monotone: more bps never helps")
def _():
    sig = np.nan_to_num(np.clip(np.sign(tools.roc(D.close, 20)), 0, 1))
    prev = np.inf
    for bps in (0, 5, 15, 50, 200):
        m = backtest.metrics(backtest.run_panel(D, sig, bps).ret, 252)
        assert m["CAGR"] <= prev + 1e-12, bps
        prev = m["CAGR"]


# ---------------------------------------------------------------- 5
@check("turnover accounting equals sum of absolute position changes")
def _():
    sig = np.random.default_rng(3).random(len(D))
    res = backtest.run_panel(D, sig, 0.0)
    manual = np.abs(np.diff(res.pos, prepend=0.0))
    assert abs(res.turnover.sum() - manual.sum()) < 1e-9


# ---------------------------------------------------------------- 6
@check("long/flat unleveraged: drawdown never below -100%, position in [0,1]")
def _():
    sig = S.signal(D, X)
    assert sig.min() >= 0.0 and sig.max() <= 1.0 and np.isfinite(sig).all()
    res = backtest.run_panel(D, sig, 15)
    m = backtest.metrics(res.ret, 252, res.pos, res.turnover)
    assert -1.0 <= m["maxDD"] <= 0.0, m["maxDD"]


# ---------------------------------------------------------------- 7
@check("indicator ranges: rsi in [0,100], pctile in [0,1], gate in [0,1]")
def _():
    c = D.close
    for a, lo, hi in ((tools.rsi(c, 14), 0, 100), (tools.pctile(c, 100), 0, 1),
                      (tools.breakout(c, 50), 0, 1),
                      (tools.gate(tools.pctile(c, 100), 0.5, 0.1), 0, 1)):
        v = a[np.isfinite(a)]
        assert v.min() >= lo - 1e-9 and v.max() <= hi + 1e-9, (v.min(), v.max())
    hard = tools.gate(tools.pctile(c, 100), 0.5, 0.0)
    assert set(np.unique(hard[np.isfinite(hard)])) <= {0.0, 1.0}


# ---------------------------------------------------------------- 8
@check("cache is namespaced: same window, different data, different answer")
def _():
    a, b = synth(seed=7), synth(seed=8)
    tools.set_context(a.tok); ra = tools.sma(a.close, 50)[-1]
    tools.set_context(b.tok); rb = tools.sma(b.close, 50)[-1]
    tools.set_context(a.tok); ra2 = tools.sma(a.close, 50)[-1]
    assert not np.isclose(ra, rb) and np.isclose(ra, ra2)


# ---------------------------------------------------------------- 9
@check("degenerate signals score -inf, not a flattering NaN")
def _():
    assert backtest.score(D, np.zeros(len(D)), 15) == -np.inf
    assert backtest.score(D, np.ones(len(D)), 15) == -np.inf     # no turnover


# ---------------------------------------------------------------- 10
@check("surrogates: same length/start, and signflip destroys the drift")
def _():
    rng = np.random.default_rng(0)
    up = synth(seed=11)
    for mode in ("signflip", "blockboot"):
        s = validate.surrogate(up, mode, rng)
        assert len(s) == len(up)
        assert abs(s.close[0] / up.close[0] - 1) < 1e-9
        assert s.close.min() > 0
    d = [validate.surrogate(up, "signflip", rng).ret.mean() for _ in range(40)]
    assert abs(np.mean(d)) < abs(up.ret.mean()), (np.mean(d), up.ret.mean())


# ---------------------------------------------------------------- 11
@check("blockboot keeps the drift it is supposed to keep")
def _():
    rng = np.random.default_rng(1)
    up = synth(seed=12)
    d = [validate.surrogate(up, "blockboot", rng).ret.mean() for _ in range(60)]
    assert abs(np.mean(d) / up.ret.mean() - 1) < 0.35, (np.mean(d), up.ret.mean())


# ---------------------------------------------------------------- 12
@check("monthly resample: index dates are real trading days, one bar per month")
def _():
    f = synth_frame()
    m = data.to_monthly(f)
    assert set(m.index).issubset(set(f.index))
    assert m.index.to_period("M").is_unique
    for dt in m.index[:60]:
        month = f[f.index.to_period("M") == dt.to_period("M")]
        assert dt == month.index[-1]
        assert abs(m.loc[dt, "Close"] - month["Close"].iloc[-1]) < 1e-9


# ---------------------------------------------------------------- 13
@check("monthly strategy: one bar per month, position changes at most monthly")
def _():
    mp = Panel.from_frame(data.to_monthly(synth_frame()), 12, "m")
    st = IuseMonthlyTrend()
    sig = st.signal(mp, np.array([np.mean(b) for b in st.bounds]))
    assert len(sig) == len(mp) and np.isfinite(sig).all()
    res = backtest.run_panel(mp, sig, 10)
    assert len(res.pos) == len(mp)


# ---------------------------------------------------------------- 14
@check("mask scoring is a subset, not a reslice: masked score uses the same signal")
def _():
    sig = S.signal(D, X)
    mask = D.index.to_period("Q").astype(str).to_numpy() < "2012Q1"
    res = backtest.run_panel(D, sig, 15)
    sub = res.ret[mask]
    exp = sub.mean() / sub.std(ddof=0) * np.sqrt(252)
    got = backtest.score(D, sig, 15, mask, min_turn=0.0)
    assert abs(exp - got) < 1e-9, (exp, got)


# ---------------------------------------------------------------- 15
@check("deflated Sharpe falls as the number of trials rises")
def _():
    a, b, c = (validate.deflated_sharpe(1.0, 2000, k) for k in (1, 100, 10000))
    assert 1.0 >= a > b > c > 0.0, (a, b, c)


# ---------------------------------------------------------------- 16
@check("metrics on a known series: 10% ann vol, zero drift")
def _():
    r = np.random.default_rng(5).normal(0, 0.10 / np.sqrt(252), 252 * 20)
    m = backtest.metrics(r, 252)
    assert abs(m["vol"] - 0.10) < 0.01 and abs(m["Sharpe"]) < 0.5, m


# ---------------------------------------------------------------- 17
@check("integer parameters are quantised, floats are not")
def _():
    p = Param("int", 30, 250)
    vals = {p.decode(v) for v in np.linspace(30, 250, 500)}
    assert len(vals) <= 41 and min(vals) == 30 and max(vals) <= 250, sorted(vals)[:5]
    q = Param("float", 0.0, 1.0)
    assert len({q.decode(v) for v in np.linspace(0, 1, 500)}) == 500


# ---------------------------------------------------------------- 18
@check("bottleneck fast path matches the pure-numpy reference")
def _():
    real_bn = tools.bn
    if real_bn is None:
        raise AssertionError("bottleneck not installed -- fast path untested")
    c, r = D.close, D.ret
    for n in (10, 63, 250):
        tools.clear_cache()
        fastp, fasts, fastv = tools.pctile(c, n), tools.sma(c, n), tools.rvol(r, n)
        fastb = tools.breakout(c, n)
        tools.bn = None
        tools.clear_cache()
        slowp, slows, slowv = tools.pctile(c, n), tools.sma(c, n), tools.rvol(r, n)
        slowb = tools.breakout(c, n)
        tools.bn = real_bn
        for a, b, nm in ((fastp, slowp, "pctile"), (fasts, slows, "sma"),
                         (fastv, slowv, "rvol"), (fastb, slowb, "breakout")):
            assert np.array_equal(np.isnan(a), np.isnan(b)), f"{nm} n={n}: NaN masks differ"
            m = ~np.isnan(a)
            assert np.allclose(a[m], b[m], atol=1e-10), f"{nm} n={n}"
    tools.clear_cache()


# ---------------------------------------------------------------- 19
@check("cached arrays are read-only, so a strategy cannot poison the cache")
def _():
    tools.set_context(D.tok)
    a = tools.sma(D.close, 50)
    try:
        a[0] = 123.0
    except ValueError:
        return
    raise AssertionError("cached array was writeable")


# ---------------------------------------------------------------- 20
@check("optimiser beats the middle of the box on its own training slice")
def _():
    d = synth(n=2500, seed=21)
    base = backtest.score(d, S.signal(d, X), 15)
    f = fit.fit(S, d, 15, seed=0, budget="fast")
    assert f.train_score >= base or not np.isfinite(base), (f.train_score, base)


# ---------------------------------------------------------------- 21
@check("one DE fit stays inside a workable time budget")
def _():
    d = synth(n=2200, seed=31)
    t0 = time.time()
    f = fit.fit(S, d, 15, seed=0, budget="normal")
    el = time.time() - t0
    print(f"        ('normal' DE fit, {len(d)} bars, {f.n_evals} evals: "
          f"{el:.1f}s = {1000*el/f.n_evals:.2f}ms/eval)", flush=True)
    assert el < 40, f"{el:.0f}s per fit makes the null bar unaffordable"


# ---------------------------------------------------------------- 22
@check("a resumed null bar is identical to an uninterrupted one")
def _():
    mp = Panel.from_frame(data.to_monthly(synth_frame(2000, seed=41)), 12, "res")
    st = IuseMonthlyTrend()
    kw = dict(n_null=3, n_splits=2, seed=5, budget="fast", block=3)
    full = validate.null_bar(st, mp, 10.0, "blockboot", **kw)
    part = validate.null_bar(st, mp, 10.0, "blockboot",
                             resume=list(full.scores[:2]), **kw)
    assert np.allclose(full.scores, part.scores), (full.scores, part.scores)


# ------------------------------------------------------------- pair support
def synth_pair(n=1200, seed=50):
    """Two correlated assets sharing a market factor plus an idiosyncratic spread."""
    rng = np.random.default_rng(seed)
    mkt = rng.normal(0.0004, 0.010, n)
    sp = rng.normal(0.0000, 0.004, n)
    idx = pd.bdate_range("2004-01-01", periods=n)

    def mk(r):
        c = 100 * np.cumprod(1 + r)
        f = pd.DataFrame({"Open": c, "High": c * 1.01, "Low": c * 0.99,
                          "Close": c, "Volume": 1e6}, index=idx)
        return f

    from llmsearch.panel import Pair
    return Pair.build(mk(mkt + sp), mk(mkt - sp), 252, "pair")


P = synth_pair()
PS = EwCwRotation()
PX = np.array([np.mean(b) for b in PS.bounds])


# ---------------------------------------------------------------- 23
@check("pair: w=1 reproduces leg A, w=0 reproduces leg B, exactly")
def _():
    for w, leg in ((1.0, P.a), (0.0, P.b)):
        r = validate.buy_and_hold(P, w)
        assert abs(r.equity[-1] / (leg.close[-1] / leg.close[0]) - 1) < 1e-9, w


# ---------------------------------------------------------------- 24
@check("pair: a switch trades both legs, so notional is 2x the weight change")
def _():
    sig = np.zeros(len(P)); sig[100:] = 1.0
    _, _, turn, pos = backtest.pnl(P, sig, 0.0)
    assert abs(turn.sum() - 2.0) < 1e-9, turn.sum()      # one 0->1 switch
    free = backtest.run_pair(P, sig, 0.0).ret
    charged = backtest.run_pair(P, sig, 100.0).ret       # 100bps
    assert abs((free - charged).sum() - 2.0 * 0.01) < 1e-9


# ---------------------------------------------------------------- 25
@check("pair: active return is exactly w*spread, market factor cancels")
def _():
    sig = np.random.default_rng(9).random(len(P))
    _, act, _, pos = backtest.pnl(P, sig, 0.0)
    assert np.allclose(act, pos * (P.ret_a - P.ret_b), atol=1e-15)
    tot, act2, _, _ = backtest.pnl(P, sig, 7.0)
    bh = validate.buy_and_hold(P, 0.0).ret               # 100% leg B
    assert np.allclose(tot - bh, act2, atol=1e-14)


# ---------------------------------------------------------------- 26
@check("pair signflip null: market factor preserved bar-for-bar, spread randomised")
def _():
    rng = np.random.default_rng(2)
    s = validate.surrogate(P, "signflip", rng, block=3)
    assert np.allclose((s.ret_a + s.ret_b)[1:], (P.ret_a + P.ret_b)[1:], atol=1e-12)
    d0, d1 = P.ret_a - P.ret_b, s.ret_a - s.ret_b
    assert np.allclose(np.abs(d1[1:]), np.abs(d0[1:]), atol=1e-12)
    assert not np.allclose(d1[1:], d0[1:])


# ---------------------------------------------------------------- 27
@check("pair signflip null: expected active return of any fixed rule is ~zero")
def _():
    rng = np.random.default_rng(3)
    w = np.random.default_rng(4).random(len(P))
    means = []
    for _ in range(60):
        s = validate.surrogate(P, "signflip", rng, block=3)
        _, act, _, _ = backtest.pnl(s, w, 0.0)
        means.append(act.mean())
    se = np.std(means) / np.sqrt(len(means))
    assert abs(np.mean(means)) < 3 * se + 1e-9, (np.mean(means), se)


# ---------------------------------------------------------------- 28
@check("pair blockboot null: both drifts and the cross-correlation survive")
def _():
    rng = np.random.default_rng(5)
    c0 = np.corrcoef(P.ret_a[1:], P.ret_b[1:])[0, 1]
    ma, mb, cs = [], [], []
    for _ in range(40):
        s = validate.surrogate(P, "blockboot", rng, block=21)
        ma.append(s.ret_a[1:].mean()); mb.append(s.ret_b[1:].mean())
        cs.append(np.corrcoef(s.ret_a[1:], s.ret_b[1:])[0, 1])
    assert abs(np.mean(ma) / P.ret_a[1:].mean() - 1) < 0.4, np.mean(ma)
    assert abs(np.mean(mb) / P.ret_b[1:].mean() - 1) < 0.4, np.mean(mb)
    assert abs(np.mean(cs) - c0) < 0.05, (np.mean(cs), c0)


# ---------------------------------------------------------------- 29
@check("pair look-ahead: future prices cannot change past signal")
def _():
    cut = 800
    base = PS.signal(P, PX)[:cut]
    from llmsearch.panel import Pair
    rng = np.random.default_rng(6)
    fa = pd.DataFrame({c: P.a.__getattribute__(k) for c, k in
                       (("Open", "open"), ("High", "high"), ("Low", "low"),
                        ("Close", "close"), ("Volume", "volume"))}, index=P.index)
    fb = pd.DataFrame({c: P.b.__getattribute__(k) for c, k in
                       (("Open", "open"), ("High", "high"), ("Low", "low"),
                        ("Close", "close"), ("Volume", "volume"))}, index=P.index)
    for f in (fa, fb):
        tail = rng.normal(0.01, 0.05, len(f) - cut)
        newc = float(f["Close"].iloc[cut - 1]) * np.cumprod(1 + tail)
        for col, mult in (("Close", 1.0), ("Open", 1.0), ("High", 1.01), ("Low", 0.99)):
            f.iloc[cut:, f.columns.get_loc(col)] = newc * mult
    got = PS.signal(Pair.build(fa, fb, 252, "alt"), PX)[:cut]
    assert np.array_equal(base, got, equal_nan=True)


# ---------------------------------------------------------------- 30
@check("pair: signal stays in [0,1] and the strategy is always fully invested")
def _():
    sig = PS.signal(P, PX)
    assert sig.min() >= 0.0 and sig.max() <= 1.0 and np.isfinite(sig).all()
    _, _, _, pos = backtest.pnl(P, sig, 5.0)
    assert np.allclose(pos + (1.0 - pos), 1.0)


# --------------------------------------------------- exogenous conditioning
def _frame(c, idx):
    return pd.DataFrame({"Open": c, "High": c * 1.01, "Low": c * 0.99,
                         "Close": c, "Volume": 1e6}, index=idx)


def synth_exog_pair(n=1200, seed=60):
    """A pair whose spread is genuinely driven by an exogenous series."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2004-01-01", periods=n)
    yld = 3.0 + np.cumsum(rng.normal(0, 0.02, n))
    dy = np.diff(yld, prepend=yld[0])
    mkt = rng.normal(0.0004, 0.010, n)
    # The link must be PREDICTIVE, not contemporaneous: positions are lagged one
    # bar, so a spread driven by the same bar's yield move is unexploitable and
    # would make this a test of nothing.
    sp = 0.5 * np.roll(dy, 1) + rng.normal(0, 0.002, n)   # yesterday's rate move
    sp[0] = 0.0
    from llmsearch.panel import Pair
    return Pair.build(_frame(100 * np.cumprod(1 + mkt + sp), idx),
                      _frame(100 * np.cumprod(1 + mkt - sp), idx), 252, "exog",
                      fx=_frame(np.abs(yld) + 0.5, idx))


XP = synth_exog_pair()
XS = ValueGrowthRotation()
XX = np.array([np.mean(b) for b in XS.bounds])


# ---------------------------------------------------------------- 31
@check("exog: the conditioning series is aligned and readable, and never in P&L")
def _():
    assert XP.x is not None and len(XP.x) == len(XP)
    assert (XP.x.index == XP.index).all()
    sig = np.random.default_rng(1).random(len(XP))
    tot, act, _, pos = backtest.pnl(XP, sig, 0.0)
    assert np.allclose(act, pos * (XP.ret_a - XP.ret_b), atol=1e-15)
    assert np.allclose(tot, pos * XP.ret_a + (1 - pos) * XP.ret_b, atol=1e-15)


# ---------------------------------------------------------------- 32
@check("exog look-ahead: a rewritten future of the conditioning series is inert")
def _():
    """Runs for every exogenous-conditioned strategy, not just one -- a new
    strategy is exactly when a leak gets introduced, so the audit is a loop over
    the registry rather than a test that has to be remembered and copied."""
    from llmsearch.panel import Pair
    cut = 800
    for cls in (ValueGrowthRotation, MomLowVolRotation, GoldBtcRotation):
        s = cls()
        xv = np.array([np.mean(b) for b in s.bounds])
        base = s.signal(XP, xv)[:cut]
        fx = _frame(XP.x.close.copy(), XP.index)
        rng = np.random.default_rng(7)
        fx.iloc[cut:, fx.columns.get_loc("Close")] = 3.0 + np.cumsum(
            rng.normal(0, 0.5, len(fx) - cut))
        alt = Pair.build(_frame(XP.a.close, XP.index), _frame(XP.b.close, XP.index),
                         252, f"alt-{s.name}", fx=fx)
        assert np.array_equal(base, s.signal(alt, xv)[:cut], equal_nan=True), s.name


# ---------------------------------------------------------------- 32b
@check("every strategy: signal is finite, inside its own declared bounds")
def _():
    from llmsearch.panel import Pair
    for cls, d in ((BtcVolRegime, D), (IuseMonthlyTrend, D),
                   (EwCwRotation, P), (ValueGrowthRotation, XP),
                   (MomLowVolRotation, XP), (GoldBtcRotation, XP)):
        s = cls()
        for seed in range(6):
            rng = np.random.default_rng(seed)
            lo = np.array([b[0] for b in s.bounds])
            hi = np.array([b[1] for b in s.bounds])
            sig = s.signal(d, lo + rng.random(len(lo)) * (hi - lo))
            assert np.isfinite(sig).all(), f"{s.name} produced non-finite values"
            assert sig.min() >= s.lo - 1e-12 and sig.max() <= s.hi + 1e-12, s.name


# ---------------------------------------------------------------- 33
@check("exog surrogates: the conditioning series survives the null untouched")
def _():
    rng = np.random.default_rng(8)
    for mode in ("signflip", "blockboot"):
        s = validate.surrogate(XP, mode, rng, block=3)
        assert s.x is not None, f"{mode} dropped the exogenous series"
        assert np.array_equal(s.x.close, XP.x.close), f"{mode} altered it"
        assert (s.x.index == XP.index).all()


# ---------------------------------------------------------------- 34
@check("exog null actually severs the link: a known-good rule dies on surrogates")
def _():
    rng = np.random.default_rng(9)

    def mech(d):
        """The rule the synthetic data was built to reward: hold leg A when the
        yield rose last bar. Not a fitted vector -- an arbitrary parameter
        vector would prove nothing about the null, only about the parameters."""
        dy = np.diff(d.x.close, prepend=d.x.close[0])
        return (dy > 0).astype(float)

    real = backtest.score(XP, mech(XP), 0.0, min_turn=0.0)
    sur = np.array([backtest.score(s, mech(s), 0.0, min_turn=0.0)
                    for s in (validate.surrogate(XP, "signflip", rng, block=3)
                              for _ in range(25))])
    sur = sur[np.isfinite(sur)]
    assert real > 1.0, f"mechanism rule should work on data built for it: {real}"
    assert real > np.nanmax(sur), (real, np.nanmax(sur))
    assert abs(np.nanmean(sur)) < 0.5, np.nanmean(sur)


# ---------------------------------------------------------------- 35
@check("cache cannot confuse a series with its own absolute value")
def _():
    tools.clear_cache()
    tools.set_context(XP.tok)
    dy = tools.roc(XP.x.close, 5)
    a, b = tools.pctile(dy, 60), tools.pctile(tools.absv(dy), 60)
    assert not np.allclose(a[~np.isnan(a)][-50:], b[~np.isnan(b)][-50:]), \
        "pctile(x) and pctile(|x|) returned the same array -- fingerprint collision"
    tools.clear_cache()
    b2 = tools.pctile(tools.absv(dy), 60)
    assert np.array_equal(b, b2, equal_nan=True), "cached and uncached disagree"


if __name__ == "__main__":
    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    sys.exit(1 if FAIL else 0)
