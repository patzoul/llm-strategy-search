"""Report from a checkpoint, without waiting for the null bars to finish.

    python report.py btc

Steps 5-9 of `run.py` -- deployed parameters, the holdout read, robustness and
cost sensitivity -- depend only on the cross-validation, not on the surrogate
null bars. The nulls are the expensive part (hundreds of DE fits) and they
answer a different question: whether the CV score is distinguishable from what
the search produces on noise. So once `run.py` has checkpointed the CV, this
prints everything else immediately, and reports the null bars at whatever
completeness they have reached.

Reading the holdout early does not leak: fitting is finished by the time the CV
checkpoint exists, and nothing here feeds back into it.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

from llmsearch import backtest, data, validate
from llmsearch.panel import Panel
from run import CONFIGS, RESULTS, align_exec, hr


def main(asset: str):
    cfg = CONFIGS[asset]
    st = cfg["strategy"]()
    ppy, cost = cfg["ppy"], cfg["cost_bps"]
    split = pd.Timestamp(cfg["split"])

    path = os.path.join(RESULTS, f"{asset}.state.json")
    if not os.path.exists(path):
        sys.exit(f"no checkpoint at {path} -- run.py has not reached one yet")
    state = json.load(open(path))
    if "cv" not in state:
        sys.exit("checkpoint has null-bar scores but no CV yet -- "
                 "run.py is still in step 3; try again once it reaches step 4")

    c = state["cv"]
    cv = validate.CV(train=np.array(c["train"], float), test=np.array(c["test"], float),
                     xs=[np.array(v, float) for v in c["xs"]])

    raw = data.load(cfg["signal_ticker"], start=cfg["start"])
    frame = data.to_monthly(raw) if cfg["freq"] == "M" else raw
    ins = Panel.from_frame(frame[frame.index <= split], ppy, cfg["signal_ticker"] + "-IS")
    full = Panel.from_frame(frame, ppy, cfg["signal_ticker"])
    hold = np.asarray(full.index > split)

    hr(f"{st.name} -- report from checkpoint")
    print(f"  CV ({len(cv.test)} splits): train {np.nanmean(cv.train):+.3f}  "
          f"TEST {cv.mean_test:+.3f}  positive on {cv.frac_positive:.0%}  "
          f"optimism {np.nanmean(cv.train) - cv.mean_test:+.3f}")

    hr("null bars (at current completeness)")
    for mode, scores in state.get("null", {}).items():
        s = np.array(scores, float)
        if not len(s):
            continue
        nb = validate.NullBar(mode=mode, scores=s)
        verdict = ("CLEARS the max" if cv.mean_test > nb.bar_max else
                   "clears p95 only" if cv.mean_test > nb.bar_p95 else
                   "INSIDE THE NULL")
        print(f"  {mode:<10s} n={len(s):<3d} mean {nb.mean:+.3f}  p95 {nb.bar_p95:+.3f}  "
              f"max {nb.bar_max:+.3f}  ->  real {cv.mean_test:+.3f} :: {verdict}  "
              f"(p={validate.null_pvalue(cv.mean_test, nb):.3f})")

    hr("deployed parameters (median over the CV splits that generalised)")
    x = validate.consensus(cv)
    print(f"  {st.describe(x)}")

    hr("in-sample vs holdout")
    res_i = backtest.run_panel(ins, st.signal(ins, x), cost)
    m_i = backtest.metrics(res_i.ret, ppy, res_i.pos, res_i.turnover)
    bh_i = validate.buy_and_hold(ins)
    print(f"  {'strategy (IS)':<24s} {backtest.fmt(m_i)}")
    print(f"  {'buy & hold (IS)':<24s} "
          f"{backtest.fmt(backtest.metrics(bh_i.ret, ppy, bh_i.pos, bh_i.turnover))}")

    sig = st.signal(full, x)
    res_h = backtest.run_panel(full, sig, cost)
    m_h = backtest.metrics(res_h.ret[hold], ppy, res_h.pos[hold], res_h.turnover[hold])
    bh = validate.buy_and_hold(full)
    m_bh = backtest.metrics(bh.ret[hold], ppy, bh.pos[hold], bh.turnover[hold])
    print(f"\n  on {cfg['signal_ticker']} (holdout, read once):")
    print(f"  {'strategy':<24s} {backtest.fmt(m_h)}")
    print(f"  {'buy & hold':<24s} {backtest.fmt(m_bh)}")

    m_tr = None
    if cfg["trade_ticker"]:
        td = data.load(cfg["trade_ticker"], start=cfg["start"])
        td.attrs["ticker"] = cfg["trade_ticker"]
        tp, pos_ex = align_exec(full.index, sig, td)
        rt = backtest.run(tp.close, pos_ex, cost, ppy, ret=tp.ret, index=tp.index)
        ht = np.asarray(tp.index > split)
        m_tr = backtest.metrics(rt.ret[ht], ppy, rt.pos[ht], rt.turnover[ht])
        bt = validate.buy_and_hold(tp)
        print(f"\n  on {cfg['trade_ticker']} (executed at the first close after each decision):")
        print(f"  {'strategy':<24s} {backtest.fmt(m_tr)}")
        print(f"  {'buy & hold':<24s} "
              f"{backtest.fmt(backtest.metrics(bt.ret[ht], ppy, bt.pos[ht], bt.turnover[ht]))}")

    n_null = sum(len(v) for v in state.get("null", {}).values())
    n_trials = len(cv.test) + n_null * len(cv.test) + 1
    print(f"\n  deflated Sharpe on the holdout: "
          f"{validate.deflated_sharpe(m_h['Sharpe'], m_h['n'], n_trials, ppy=ppy):.3f}"
          f"   (n_trials={n_trials})")

    hr("year by year (holdout)")
    df = pd.DataFrame({"s": res_h.ret, "b": bh.ret, "p": res_h.pos}, index=full.index)[hold]
    print("  year    strategy   buy&hold      diff   avg exposure")
    for y, g in df.groupby(df.index.year):
        s, b = (1 + g.s).prod() - 1, (1 + g.b).prod() - 1
        print(f"  {y}   {100*s:+8.1f}%  {100*b:+8.1f}%  {100*(s-b):+8.1f}%      {g.p.mean():.2f}")

    hr("parameter robustness on the holdout")
    base = m_h["Sharpe"]
    for i, nm in enumerate(st.names):
        lo, hi = st.bounds[i]
        row = []
        for f in (-0.20, -0.10, 0.10, 0.20):
            xx = x.copy()
            xx[i] = np.clip(x[i] + f * (hi - lo), lo, hi)
            rr = backtest.run_panel(full, st.signal(full, xx), cost)
            row.append(backtest.metrics(rr.ret[hold], ppy)["Sharpe"])
        sp = float(np.nanmax(row) - np.nanmin(row))
        print(f"  {nm:<10s} base {base:+.2f} | " + " ".join(f"{v:+.2f}" for v in row) +
              f"   spread {sp:.2f}" + ("   <-- fragile" if sp > 0.6 else ""))

    hr("cost sensitivity (holdout)")
    for mult in (0.5, 1.0, 2.0, 5.0):
        mm = backtest.metrics(backtest.run_panel(full, sig, cost * mult).ret[hold], ppy)
        print(f"  {cost*mult:>5.1f} bps   Sharpe {mm['Sharpe']:+.2f}   CAGR {mm['CAGR']:+.2%}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "btc")
