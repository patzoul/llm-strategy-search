"""Apply the fitted EW/CW rule to the UCITS pair a UK investor can actually buy.

    python tradeleg.py

RSP and SPY are US-domiciled and closed to UK retail under PRIIPs. The LSE
equivalents are XDEW.L (Xtrackers S&P 500 Equal Weight, USD) and CSPX.L (iShares
Core S&P 500, USD). Both are USD-quoted, so no FX is introduced.

The signal is still computed on RSP/SPY -- those are the indices being timed and
they have 23 years of history against XDEW.L's 12. What this script tests is
whether the rule survives on the actual instruments: different domicile, worse
liquidity, withholding-tax drag, and a decision struck at the US close that can
only be executed at the next LSE close.

EWSP.L is deliberately not used: its Yahoo series shows 26% annualised vol and a
+34% daily print for an S&P 500 tracker, which is a data defect, not a fund.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

from llmsearch import backtest, data, validate
from llmsearch.panel import Pair
from run import CONFIGS, RESULTS
from strategies.ew_cw_rotation import EwCwRotation

EW_UCITS, CW_UCITS = "XDEW.L", "CSPX.L"


def main():
    cfg = CONFIGS["ewcw"]
    st = EwCwRotation()
    ppy, cost = cfg["ppy"], cfg["cost_bps"]

    sp = os.path.join(RESULTS, "ewcw.state.json")
    if not os.path.exists(sp):
        raise SystemExit("run.py ewcw has not checkpointed a CV yet")
    c = json.load(open(sp))["cv"]
    cv = validate.CV(train=np.array(c["train"], float), test=np.array(c["test"], float),
                     xs=[np.array(v, float) for v in c["xs"]])
    x = validate.consensus(cv)
    print(f"deployed params: {st.describe(x)}\n")

    # -- signal on the US indices (long history) ---------------------------
    us = {t: data.to_monthly(data.load(t, start=cfg["start"])) for t in ("RSP", "SPY")}
    common = us["RSP"].index.intersection(us["SPY"].index)
    us_pair = Pair.build(us["RSP"].loc[common], us["SPY"].loc[common], ppy, "US")
    sig = pd.Series(st.signal(us_pair, x), index=us_pair.index)

    # -- execute on the UCITS pair, one LSE close later --------------------
    lse = {t: data.load(t, start="2014-01-01") for t in (EW_UCITS, CW_UCITS)}
    for t, d in lse.items():
        rep = data.integrity_report(d, t, spike=0.25)
        data.print_report(rep)
    idx = lse[EW_UCITS].index.intersection(lse[CW_UCITS].index)

    dec, ex = [], []
    for dt in sig.index:
        i = idx.searchsorted(dt, side="right")
        if i < len(idx):
            dec.append(dt)
            ex.append(idx[i])
    ex_idx = pd.DatetimeIndex(ex)
    keep = ~ex_idx.duplicated(keep="first")
    dec = pd.DatetimeIndex(np.asarray(dec)[keep])
    ex_idx = ex_idx[keep]
    assert (dec < ex_idx).all(), "execution must be strictly after the decision"

    fa = lse[EW_UCITS].loc[ex_idx, ["Close"]]
    fb = lse[CW_UCITS].loc[ex_idx, ["Close"]]
    fa.attrs["ticker"], fb.attrs["ticker"] = EW_UCITS, CW_UCITS
    tp = Pair.build(fa, fb, ppy, "UCITS")
    w = sig.loc[dec].to_numpy()

    start = max(tp.index[0], pd.Timestamp(cfg["split"]))
    m = np.asarray(tp.index > start)
    print(f"\nUCITS pair {EW_UCITS}/{CW_UCITS}, {tp.index[m][0].date()} -> "
          f"{tp.index[-1].date()}  ({m.sum()} months)")

    res = backtest.run_pair(tp, w, cost)
    rows = [("strategy", backtest.metrics(res.ret[m], ppy, res.pos[m], res.turnover[m]))]
    for lbl, wt in ((f"100% {EW_UCITS} (EW)", 1.0), (f"100% {CW_UCITS} (CW)", 0.0),
                    ("50/50 fixed", 0.5)):
        r = validate.buy_and_hold(tp, wt)
        rows.append((lbl, backtest.metrics(r.ret[m], ppy, r.pos[m], r.turnover[m])))
    for lbl, mm in rows:
        print(f"  {lbl:<22s} {backtest.fmt(mm)}")

    _, act, _, _ = backtest.pnl(tp, w, cost)
    a = act[m]
    print(f"\n  active vs 100% CW: {(1+a).prod()**(ppy/len(a))-1:+.2%}/yr   "
          f"IR {a.mean()/a.std(ddof=0)*np.sqrt(ppy):+.2f}")

    # -- does the UCITS pair track the US pair at all? ---------------------
    # The two panels are indexed on different calendars -- US month-end closes
    # versus the LSE close one day later -- so they must be joined on the month,
    # not the date, or every value lines up against a NaN.
    us_sp = pd.Series(us_pair.ret_a - us_pair.ret_b,
                      index=us_pair.index.to_period("M"))
    lse_sp = pd.Series(tp.ret_a - tp.ret_b, index=tp.index.to_period("M"))
    j = pd.concat([us_sp.rename("us"), lse_sp.rename("lse")], axis=1).dropna().iloc[1:]
    print(f"\n  monthly EW-CW spread, UCITS vs US ({len(j)} months): "
          f"corr {j.us.corr(j.lse):.3f}  "
          f"(sd {100*j.lse.std():.2f}% vs {100*j.us.std():.2f}%)")


if __name__ == "__main__":
    main()
