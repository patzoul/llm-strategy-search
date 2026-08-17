"""Driver: fit -> blocked CV -> null bars -> holdout, for one strategy.

    python run.py btc
    python run.py iuse
    python run.py btc --budget fast --splits 20 --nulls 8

The order is fixed and the holdout is read exactly once, at step 7. Nothing
before that point is given a panel that contains it.
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import pandas as pd

from llmsearch import backtest, data, fit, validate
from llmsearch.panel import Pair, Panel
from strategies.btc_regime import BtcVolRegime
from strategies.ew_cw_rotation import EwCwRotation
from strategies.gold_btc import GoldBtcRotation
from strategies.iuse_monthly import IuseMonthlyTrend
from strategies.mom_lowvol import MomLowVolRotation
from strategies.small_large import SmallLargeRotation
from strategies.value_growth import ValueGrowthRotation

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")

CONFIGS = {
    "btc": dict(
        strategy=BtcVolRegime,
        signal_ticker="BTC-USD",
        trade_ticker=None,              # traded directly
        start="2014-09-17",
        split="2022-12-31",             # everything after this is the holdout
        freq="D", ppy=252, cost_bps=15.0, block=21, spike=0.5,
        kill="holdout Sharpe net of costs < 0.4, or worse than buy&hold on both "
             "Sharpe and maxDD, or CV score inside either null bar",
    ),
    "ewcw": dict(
        strategy=EwCwRotation,
        kind="pair",
        signal_ticker="RSP",            # leg A: equal weight  (weight = the signal)
        ticker_b="SPY",                 # leg B: cap weight    (weight = 1 - signal)
        leg_a="equal wt", leg_b="cap wt",
        trade_ticker=None,              # UCITS trade-leg check is a separate script
        start="2003-05-01",             # RSP inception
        split="2016-12-31",
        freq="M", ppy=12, cost_bps=5.0, block=3, spike=0.25,
        kill="holdout information ratio vs cap-weight <= 0, or CV score inside "
             "either null bar, or the holdout Sharpe fails to beat all three "
             "fixed benchmarks (100% CW, 100% EW, 50/50)",
    ),
    "valgro": dict(
        strategy=ValueGrowthRotation,
        kind="pair",
        signal_ticker="IVE",            # leg A: S&P 500 Value   (weight = the signal)
        ticker_b="IVW",                 # leg B: S&P 500 Growth  (weight = 1 - signal)
        leg_a="value", leg_b="growth",
        exog_ticker="^TNX",             # conditioning only: 10y Treasury yield
        exog_spike=0.60,                # a yield series moves far more than a price one
        trade_ticker=None,
        start="2000-05-26",             # IVE/IVW inception
        split="2014-12-31",
        freq="M", ppy=12, cost_bps=5.0, block=3, spike=0.25,
        kill="holdout information ratio vs growth <= 0, or CV score inside "
             "either null bar, or the holdout Sharpe fails to beat all three "
             "fixed benchmarks (100% growth, 100% value, 50/50)",
    ),
    "momlv": dict(
        strategy=MomLowVolRotation,
        kind="pair",
        signal_ticker="MTUM",           # leg A: MSCI USA Momentum   (weight = signal)
        ticker_b="USMV",                # leg B: MSCI USA Min Vol    (weight = 1 - signal)
        leg_a="momentum", leg_b="min vol",
        exog_ticker="SPY",              # conditioning only: market state
        trade_ticker=None,
        start="2013-04-18",             # MTUM inception (USMV starts 2011-10)
        split="2020-12-31",
        freq="M", ppy=12, cost_bps=5.0, block=3, spike=0.25,
        kill="holdout information ratio vs min-vol <= 0, or CV score inside "
             "either null bar, or the holdout Sharpe fails to beat all three "
             "fixed benchmarks (100% min vol, 100% momentum, 50/50)",
    ),
    "goldbtc": dict(
        strategy=GoldBtcRotation,
        kind="pair",
        signal_ticker="BTC-USD",        # leg A: bitcoin  (weight = the signal)
        ticker_b="GLD",                 # leg B: gold     (weight = 1 - signal)
        leg_a="bitcoin", leg_b="gold",
        exog_ticker="SPY",              # conditioning only: risk appetite
        trade_ticker=None,
        start="2014-09-17",             # BTC-USD inception on Yahoo
        split="2021-12-31",
        freq="D", ppy=252, cost_bps=15.0, block=21, spike=0.5,
        kill="holdout information ratio vs gold <= 0, or CV score inside either "
             "null bar, or the holdout Sharpe fails to beat all three fixed "
             "benchmarks (100% gold, 100% bitcoin, 50/50) AND the risk-matched "
             "constant-weight blend",
    ),
    "smlg": dict(
        strategy=SmallLargeRotation,
        kind="pair",
        signal_ticker="IWM",            # leg A: Russell 2000  (weight = the signal)
        ticker_b="IWB",                 # leg B: Russell 1000  (weight = 1 - signal)
        leg_a="small", leg_b="large",
        exog_ticker="SPY",              # conditioning only: risk appetite
        trade_ticker=None,
        start="2000-05-26",             # IWM/IWB inception
        split="2014-12-31",
        freq="M", ppy=12, cost_bps=5.0, block=3, spike=0.25,
        kill="holdout information ratio vs large caps <= 0, or CV score inside "
             "either null bar, or the holdout Sharpe fails to beat all three "
             "fixed benchmarks (100% large, 100% small, 50/50) AND the "
             "risk-matched constant-weight blend",
    ),
    "iuse": dict(
        strategy=IuseMonthlyTrend,
        signal_ticker="SPY",            # the thing being timed: the S&P 500, long history
        trade_ticker="IUSE.L",          # the thing actually bought: EUR-hedged UCITS ETF
        start="1993-01-29",
        split="2013-12-31",
        freq="M", ppy=12, cost_bps=10.0, block=3, spike=0.25,
        kill="holdout Sharpe net of costs below buy&hold's, or CV score inside "
             "either null bar, or the fitted rule collapses to a permanent core "
             "holding (floor at its cap with no timing left)",
    ),
}


def load_state(path: str) -> dict:
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def save_state(path: str, st: dict) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(st, f, default=str)
    os.replace(tmp, path)          # atomic: a kill mid-write cannot corrupt the state


def hr(t=""):
    print("\n" + "=" * 78, flush=True)
    if t:
        print(t)
        print("=" * 78, flush=True)


def align_exec(sig_index, sig_values, target_daily: pd.DataFrame):
    """Apply a signal decided on one series to a different, later-closing one.

    SPY's month-end close is set at 21:00 London, after IUSE.L has closed, so
    the earliest executable price is IUSE.L's *next* close. Returns a panel of
    execution prices and the position held over each execution bar, with every
    decision date strictly before its own execution date.
    """
    px = target_daily["Close"]
    dec, ex = [], []
    for d in sig_index:
        i = px.index.searchsorted(d, side="right")
        if i < len(px):
            dec.append(d)
            ex.append(px.index[i])
    ex = pd.DatetimeIndex(ex)
    keep = ~ex.duplicated(keep="first")
    dec = pd.DatetimeIndex(np.asarray(dec)[keep])
    ex = ex[keep]
    assert len(ex) and (dec < ex).all(), "execution must be strictly after the decision"
    vals = pd.Series(sig_values, index=sig_index).loc[dec].to_numpy()
    frame = px.reindex(ex).to_frame("Close")
    return Panel.from_frame(frame, 12.0, target_daily.attrs.get("ticker", "traded")), vals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("asset", choices=list(CONFIGS))
    ap.add_argument("--budget", default="normal", choices=list(fit.BUDGETS))
    ap.add_argument("--splits", type=int, default=30)
    ap.add_argument("--nulls", type=int, default=12)
    ap.add_argument("--null-splits", type=int, default=0,
                    help="0 = same as --splits, which keeps the bar comparable")
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    null_splits = a.null_splits or a.splits

    cfg = CONFIGS[a.asset]
    st = cfg["strategy"]()
    ppy, cost = cfg["ppy"], cfg["cost_bps"]
    t0 = time.time()

    hr(f"{st.name}  |  signal={cfg['signal_ticker']}  "
       f"trade={cfg['trade_ticker'] or cfg['signal_ticker']}  "
       f"freq={cfg['freq']}  cost={cost:.0f}bps")
    print(f"KILL CRITERION (fixed before anything ran):\n  {cfg['kill']}")

    # ---------------------------------------------------------------- 1 data
    hr("1. data integrity")
    is_pair = cfg.get("kind") == "pair"
    raw = data.load(cfg["signal_ticker"], start=cfg["start"])
    data.print_report(data.integrity_report(raw, cfg["signal_ticker"], spike=cfg["spike"]))
    frame = data.to_monthly(raw) if cfg["freq"] == "M" else raw

    frame_b = None
    if is_pair:
        raw_b = data.load(cfg["ticker_b"], start=cfg["start"])
        data.print_report(data.integrity_report(raw_b, cfg["ticker_b"], spike=cfg["spike"]))
        frame_b = data.to_monthly(raw_b) if cfg["freq"] == "M" else raw_b
        common = frame.index.intersection(frame_b.index)
        print(f"  aligned on {len(common)} common bars "
              f"(dropped {len(frame)-len(common)}/{len(frame_b)-len(common)})")
        frame, frame_b = frame.loc[common], frame_b.loc[common]
        frame.attrs["ticker"] = cfg["signal_ticker"]
        frame_b.attrs["ticker"] = cfg["ticker_b"]

    frame_x = None
    if cfg.get("exog_ticker"):
        raw_x = data.load(cfg["exog_ticker"], start=cfg["start"])
        data.print_report(data.integrity_report(raw_x, cfg["exog_ticker"],
                                                spike=cfg.get("exog_spike", 0.25)))
        fx = data.to_monthly(raw_x) if cfg["freq"] == "M" else raw_x
        common = frame.index.intersection(fx.index)
        # A conditioning series should cover the pair, not truncate it. Losing a
        # large share of bars means the exogenous data does not span the sample --
        # fail loudly rather than silently fitting on whatever remains.
        lost = 1.0 - len(common) / max(len(frame), 1)
        if lost > 0.20:
            raise SystemExit(
                f"{cfg['exog_ticker']} covers only {len(common)} of {len(frame)} bars "
                f"({lost:.0%} lost). Its range is {fx.index[0].date()}..{fx.index[-1].date()} "
                f"against the pair's {frame.index[0].date()}..{frame.index[-1].date()}.")
        frame, frame_b, frame_x = frame.loc[common], frame_b.loc[common], fx.loc[common]
        frame_x.attrs["ticker"] = cfg["exog_ticker"]
        print(f"  conditioning on {cfg['exog_ticker']} (not tradeable, no P&L "
              f"contribution); {len(common)} bars after alignment")

    def build(f, fb, nm):
        if not is_pair:
            return Panel.from_frame(f, ppy, nm)
        fx = None if frame_x is None else frame_x.loc[f.index]
        return Pair.build(f, fb, ppy, nm, fx=fx)

    trade_daily = None
    if cfg["trade_ticker"]:
        trade_daily = data.load(cfg["trade_ticker"], start=cfg["start"])
        trade_daily.attrs["ticker"] = cfg["trade_ticker"]
        data.print_report(data.integrity_report(trade_daily, cfg["trade_ticker"],
                                                spike=cfg["spike"]))

    split = pd.Timestamp(cfg["split"])
    m_is = frame.index <= split
    ins = build(frame[m_is], frame_b[m_is] if is_pair else None,
                cfg["signal_ticker"] + "-IS")
    min_bars = 40 if cfg["freq"] == "M" else 250
    if len(ins) < min_bars:
        raise SystemExit(
            f"in-sample window is {len(ins)} bars, below the {min_bars} needed to fit "
            f"{len(st.names)} parameters over blocked quarters. Check the integrity "
            f"report above -- a short panel usually means a series did not load in full.")
    print(f"\n  in-sample : {ins.index[0].date()} -> {ins.index[-1].date()}  "
          f"({len(ins)} bars)")
    n_hold = int((frame.index > split).sum())
    print(f"  holdout   : {frame[frame.index > split].index[0].date()} -> "
          f"{frame.index[-1].date()}  ({n_hold} bars)  [not loaded until step 7]")

    # ------------------------------------------------------- 2 benchmark, IS
    hr("2. benchmarks on the in-sample window")
    if is_pair:
        la = cfg.get("leg_a", "leg A")
        lb = cfg.get("leg_b", "leg B")
        benches = {f"100% {cfg['signal_ticker']} ({la})": 1.0,
                   f"100% {cfg['ticker_b']} ({lb})": 0.0, "50/50 fixed": 0.5}
    else:
        benches = {"buy & hold": 1.0}
    bh_ins = validate.buy_and_hold(ins, list(benches.values())[-1 if is_pair else 0])
    m_bh_ins = None
    for lbl, w in benches.items():
        r = validate.buy_and_hold(ins, w)
        m = backtest.metrics(r.ret, ppy, r.pos, r.turnover)
        if m_bh_ins is None:
            m_bh_ins = m
        print(f"  {lbl + ' (IS)':<26s} {backtest.fmt(m)}")

    # ------------------------------------------------------------ 3 blocked CV
    hr(f"3. blocked-quarter CV in-sample ({a.splits} splits, 75/25, budget={a.budget})")
    state_path = os.path.join(RESULTS, f"{a.asset}.state.json")
    state = load_state(state_path)
    sig_cfg = f"{a.splits}|{a.budget}|{a.seed}|{cost}|{cfg['split']}"
    if state.get("cfg") != sig_cfg:
        state = {"cfg": sig_cfg}       # settings changed -> nothing carried over is valid

    if "cv" in state:
        c = state["cv"]
        cv = validate.CV(train=np.array(c["train"], float), test=np.array(c["test"], float),
                         xs=[np.array(v, float) for v in c["xs"]])
        print("  [resumed from checkpoint]")
    else:
        cv = validate.blocked_cv(st, ins, cost, n_splits=a.splits, seed=a.seed,
                                 budget=a.budget)
        state["cv"] = {"train": cv.train.tolist(), "test": cv.test.tolist(),
                       "xs": [list(map(float, v)) for v in cv.xs]}
        save_state(state_path, state)
    print(f"  train Sharpe  mean {np.nanmean(cv.train):+.3f}  median {np.nanmedian(cv.train):+.3f}")
    print(f"  TEST  Sharpe  mean {cv.mean_test:+.3f}  median {cv.median_test:+.3f}  "
          f"sd {np.nanstd(cv.test):.3f}  positive on {cv.frac_positive:.0%} of splits")
    print(f"  optimism (train - test): {np.nanmean(cv.train) - cv.mean_test:+.3f}")

    # ----------------------------------------------------------- 4 null bars
    hr(f"4. null bars -- the identical fit-and-CV pipeline on structureless data "
       f"({a.nulls} surrogates x {null_splits} splits)")

    def prog(mode, i, n, s):
        print(f"     {mode} {i}/{n}: {s:+.3f}", flush=True)
        state.setdefault("null", {}).setdefault(mode, []).append(float(s))
        save_state(state_path, state)

    nulls = {}
    for mode in ("signflip", "blockboot"):
        done = list(state.get("null", {}).get(mode, []))[:a.nulls]
        if done:
            print(f"     [{mode}: {len(done)}/{a.nulls} resumed from checkpoint]", flush=True)
            state.setdefault("null", {})[mode] = done
        nb = validate.null_bar(st, ins, cost, mode, n_null=a.nulls,
                               n_splits=null_splits, seed=a.seed, budget=a.budget,
                               block=cfg["block"], progress=prog, resume=done)
        p = validate.null_pvalue(cv.mean_test, nb)
        nulls[mode] = dict(mean=nb.mean, p95=nb.bar_p95, max=nb.bar_max, pval=p,
                           scores=[round(float(v), 4) for v in nb.scores])
        verdict = ("CLEARS the max" if cv.mean_test > nb.bar_max else
                   "clears p95 only" if cv.mean_test > nb.bar_p95 else
                   "INSIDE THE NULL")
        print(f"  {mode:<10s} null mean {nb.mean:+.3f}  p95 {nb.bar_p95:+.3f}  "
              f"max {nb.bar_max:+.3f}  ->  real {cv.mean_test:+.3f} :: {verdict}  (p={p:.3f})")

    # ------------------------------------------------- 5 deployed parameters
    hr("5. parameters deployed (median over the CV splits that generalised)")
    x = validate.consensus(cv)
    print(f"  {st.describe(x)}")
    print(f"  (single full-IS fit, for contrast:\n   "
          f"{st.describe(fit.fit(st, ins, cost, seed=a.seed, budget=a.budget).x)})")

    run = backtest.run_pair if is_pair else backtest.run_panel

    hr("6. in-sample performance with the deployed parameters")
    res_ins = run(ins, st.signal(ins, x), cost)
    m_ins = backtest.metrics(res_ins.ret, ppy, res_ins.pos, res_ins.turnover)
    print(f"  {'strategy (IS)':<26s} {backtest.fmt(m_ins)}")
    for lbl, w in benches.items():
        r = validate.buy_and_hold(ins, w)
        print(f"  {lbl + ' (IS)':<26s} "
              f"{backtest.fmt(backtest.metrics(r.ret, ppy, r.pos, r.turnover))}")

    # ------------------------------------------------------------ 7 HOLDOUT
    hr("7. HOLDOUT -- read once, nothing is tuned after this point")
    full = build(frame, frame_b, cfg["signal_ticker"])
    sig_full = st.signal(full, x)
    hold = np.asarray(full.index > split)

    res_h = run(full, sig_full, cost)
    m_h = backtest.metrics(res_h.ret[hold], ppy, res_h.pos[hold], res_h.turnover[hold])
    m_bh = None
    print(f"  {'strategy':<26s} {backtest.fmt(m_h)}")
    for lbl, w in benches.items():
        r = validate.buy_and_hold(full, w)
        m = backtest.metrics(r.ret[hold], ppy, r.pos[hold], r.turnover[hold])
        if m_bh is None:
            m_bh = m
        print(f"  {lbl:<26s} {backtest.fmt(m)}")

    if is_pair:
        _, act, _, _ = backtest.pnl(full, sig_full, cost)
        a_h = act[hold]
        ir = a_h.mean() / a_h.std(ddof=0) * np.sqrt(ppy)
        print(f"\n  active return vs 100% {cfg['ticker_b']} (CW): "
              f"{(1+a_h).prod()**(ppy/len(a_h))-1:+.2%}/yr   IR {ir:+.2f}")

    m_tr = m_bhtr = None
    if trade_daily is not None:
        tp, pos_ex = align_exec(full.index, sig_full, trade_daily)
        res_tr = backtest.run(tp.close, pos_ex, cost, ppy, ret=tp.ret, index=tp.index)
        ht = np.asarray(tp.index > split)
        m_tr = backtest.metrics(res_tr.ret[ht], ppy, res_tr.pos[ht], res_tr.turnover[ht])
        bh_tr = validate.buy_and_hold(tp)
        m_bhtr = backtest.metrics(bh_tr.ret[ht], ppy, bh_tr.pos[ht], bh_tr.turnover[ht])
        print(f"\n  on {cfg['trade_ticker']} (the instrument actually bought, "
              f"executed at the first close after each decision):")
        print(f"  {'strategy':<26s} {backtest.fmt(m_tr)}")
        print(f"  {'buy & hold':<26s} {backtest.fmt(m_bhtr)}")

    n_trials = a.splits + a.nulls * null_splits * 2 + 1
    dsr = validate.deflated_sharpe(m_h["Sharpe"], m_h["n"], n_trials, ppy=ppy)
    print(f"\n  deflated Sharpe on the holdout: {dsr:.3f}  "
          f"(n_trials={n_trials} DE fits, n_obs={m_h['n']})")

    # ------------------------------------------------- 8 robustness + costs
    hr("8. parameter robustness on the holdout -- plateau or spike?")
    base = m_h["Sharpe"]
    frag = []
    for i, nm in enumerate(st.names):
        lo, hi = st.bounds[i]
        row = []
        for f in (-0.20, -0.10, 0.10, 0.20):
            xx = x.copy()
            xx[i] = np.clip(x[i] + f * (hi - lo), lo, hi)
            rr = run(full, st.signal(full, xx), cost)
            row.append(backtest.metrics(rr.ret[hold], ppy)["Sharpe"])
        spread = float(np.nanmax(row) - np.nanmin(row))
        if spread > 0.6:
            frag.append(nm)
        print(f"  {nm:<10s} base {base:+.2f} | " + " ".join(f"{v:+.2f}" for v in row) +
              f"   spread {spread:.2f}" + ("   <-- fragile" if spread > 0.6 else ""))

    hr("9. cost sensitivity (holdout)")
    costs = {}
    for mult in (0.5, 1.0, 2.0, 5.0):
        rr = run(full, sig_full, cost * mult)
        mm = backtest.metrics(rr.ret[hold], ppy)
        costs[cost * mult] = mm["Sharpe"]
        print(f"  {cost*mult:>5.1f} bps   Sharpe {mm['Sharpe']:+.2f}   CAGR {mm['CAGR']:+.2%}")

    # --------------------------------------------------------------- output
    os.makedirs(RESULTS, exist_ok=True)
    out = dict(asset=a.asset, strategy=st.name, params=st.decode(x),
               cv_mean_test=cv.mean_test, cv_median_test=cv.median_test,
               cv_frac_positive=cv.frac_positive,
               cv_train_mean=float(np.nanmean(cv.train)), nulls=nulls,
               is_metrics=m_ins, is_bh=m_bh_ins, holdout=m_h, holdout_bh=m_bh,
               holdout_traded=m_tr, holdout_traded_bh=m_bhtr, dsr=dsr,
               n_trials=n_trials, fragile_params=frag, cost_sensitivity=costs,
               cost_bps=cost, split=cfg["split"], budget=a.budget,
               splits=a.splits, nulls_n=a.nulls, null_splits=null_splits,
               runtime_s=round(time.time() - t0, 1))
    p = os.path.join(RESULTS, f"{a.asset}.json")
    with open(p, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nwrote {p}   ({out['runtime_s']:.0f}s)")


if __name__ == "__main__":
    main()
