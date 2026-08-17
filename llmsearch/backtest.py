"""Signal -> positions -> net returns -> metrics.

Execution convention (the same one everywhere in this repo):

    the signal at bar t is computed from data up to and including the close of
    bar t, and is held over bar t+1.

So `position = shift(signal, 1)`. On the monthly panel a "bar" is a month, which
means: decide on the last trading day of the month, trade at the next available
close, hold for the month. That is what makes the IUSE strategy honestly a
one-month-or-longer strategy rather than a daily one dressed up.

Costs are charged on turnover and are never zero.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class Result:
    ret: np.ndarray          # net strategy returns, per bar
    gross: np.ndarray
    pos: np.ndarray          # position actually held over each bar
    turnover: np.ndarray
    ppy: float
    index: pd.DatetimeIndex | None = None

    @property
    def equity(self) -> np.ndarray:
        return np.cumprod(1.0 + self.ret)


def positions(sig: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pos = np.empty_like(sig)
    pos[0] = 0.0
    pos[1:] = sig[:-1]
    turn = np.abs(np.diff(pos, prepend=0.0))
    return pos, turn


def run(px: np.ndarray, sig: np.ndarray, cost_bps: float, ppy: float,
        ret: np.ndarray | None = None, index=None) -> Result:
    if ret is None:
        ret = np.zeros_like(px, dtype=float)
        ret[1:] = px[1:] / px[:-1] - 1.0
    pos, turn = positions(np.asarray(sig, dtype=float))
    gross = pos * ret
    net = gross - turn * (cost_bps / 1e4)
    return Result(ret=net, gross=gross, pos=pos, turnover=turn, ppy=ppy, index=index)


def run_panel(d, sig, cost_bps: float) -> Result:
    return run(d.close, sig, cost_bps, d.ppy, ret=d.ret, index=d.index)


# --------------------------------------------------------------------------

def metrics(ret: np.ndarray, ppy: float, pos=None, turnover=None) -> dict:
    ret = np.asarray(ret, dtype=float)
    ret = ret[np.isfinite(ret)]
    n = len(ret)
    if n < 2:
        return {k: np.nan for k in
                ("CAGR", "vol", "Sharpe", "Sortino", "maxDD", "turn_py", "expo")} | {"n": n}
    eq = np.cumprod(1.0 + ret)
    yrs = n / ppy
    cagr = eq[-1] ** (1.0 / yrs) - 1.0 if eq[-1] > 0 else -1.0
    sd = ret.std(ddof=0)
    vol = sd * np.sqrt(ppy)
    sharpe = (ret.mean() / sd * np.sqrt(ppy)) if sd > 0 else np.nan
    neg = ret[ret < 0]
    dn = neg.std(ddof=0) if len(neg) > 1 else np.nan
    sortino = (ret.mean() / dn * np.sqrt(ppy)) if dn and dn > 0 else np.nan
    mdd = float((eq / np.maximum.accumulate(eq) - 1.0).min())
    out = {"n": n, "CAGR": float(cagr), "vol": float(vol), "Sharpe": float(sharpe),
           "Sortino": float(sortino), "maxDD": mdd}
    out["turn_py"] = float(np.nansum(turnover) / yrs) if turnover is not None else np.nan
    # `expo` is *average* exposure, not the fraction of bars with a non-zero
    # position. A strategy with a permanent core holding of 0.2% is technically
    # "in the market" 100% of the time, which tells you nothing about how much
    # risk it was actually carrying.
    out["expo"] = float(np.mean(np.abs(np.asarray(pos)))) if pos is not None else np.nan
    return out


def fmt(m: dict) -> str:
    return (f"CAGR {m['CAGR']:>7.2%}  vol {m['vol']:>6.2%}  Sharpe {m['Sharpe']:>5.2f}  "
            f"maxDD {m['maxDD']:>7.2%}  turn/y {m['turn_py']:>5.1f}  "
            f"avgExp {m['expo']:>5.1%}  n={m['n']}")


def is_pair(d) -> bool:
    return hasattr(d, "ret_b")


def pnl(d, sig: np.ndarray, cost_bps: float):
    """Net total return, net active return and traded notional, per bar.

    Single asset: position `sig` in the asset, the rest in cash. Active return
    is the same thing (cash earns nothing here).

    Pair: weight `sig` on leg A and 1-sig on leg B, always fully invested. A
    switch trades *both* legs, so the notional traded is 2x the weight change --
    getting this wrong halves the cost of the only thing the strategy does.
    Active return is measured against holding leg B throughout, which reduces
    exactly to `w * spread`: the common market factor cancels, leaving only what
    the rotation rule controls.
    """
    pos, turn = positions(np.asarray(sig, dtype=float))
    if is_pair(d):
        turn = 2.0 * turn
        cost = turn * (cost_bps / 1e4)
        total = pos * d.ret_a + (1.0 - pos) * d.ret_b - cost
        active = pos * (d.ret_a - d.ret_b) - cost
    else:
        cost = turn * (cost_bps / 1e4)
        total = active = pos * d.ret - cost
    return total, active, turn, pos


def run_pair(d, sig: np.ndarray, cost_bps: float) -> Result:
    total, _, turn, pos = pnl(d, sig, cost_bps)
    return Result(ret=total, gross=total + turn * (cost_bps / 1e4), pos=pos,
                  turnover=turn, ppy=d.ppy, index=d.index)


def score(d, sig: np.ndarray, cost_bps: float, mask: np.ndarray | None = None,
          min_turn: float = 0.25) -> float:
    """The single number the optimiser maximises: net Sharpe on the scored bars.

    `mask` selects which bars count (used by the blocked-quarter CV so that a
    strategy is fitted on the training quarters and scored on the held-out ones
    without ever re-slicing the price series -- slicing would silently change
    every rolling window and make the two runs incomparable).

    For a pair this is the **information ratio of the active return** against
    holding leg B, not the total Sharpe. Total Sharpe would be dominated by the
    shared equity beta, which the rotation rule cannot influence -- optimising
    it means fitting noise in a series that is ~95% common factor. The IR
    isolates the part the strategy is actually responsible for, and it is the
    quantity the surrogate nulls are built to have zero expectation of.

    A strategy that never trades gets -inf rather than a flattering NaN.
    """
    _, r, turn, _ = pnl(d, sig, cost_bps)
    if mask is not None:
        r, turn = r[mask], turn[mask]
    r = r[np.isfinite(r)]
    if len(r) < 20:
        return -np.inf
    yrs = len(r) / d.ppy
    if turn.sum() / max(yrs, 1e-9) < min_turn:
        return -np.inf          # degenerate: effectively buy-and-hold or always flat
    sd = r.std(ddof=0)
    if not np.isfinite(sd) or sd <= 0:
        return -np.inf
    s = r.mean() / sd * np.sqrt(d.ppy)
    return float(s) if np.isfinite(s) else -np.inf
