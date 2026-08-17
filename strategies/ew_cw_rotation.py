"""Equal-weight vs cap-weight S&P 500 rotation -- month-end, long-only.

The signal is the weight on the EQUAL-weight index; the remainder goes to
cap-weight. Always 100% invested in US large caps -- this strategy never decides
how much equity to hold, only which version of the same 500 companies.

MECHANISM:
The two indices hold identical constituents, so the entire spread between them
is a bet on breadth and concentration:
  * equal weight carries a structural small-size and value tilt relative to
    cap weight, and rebalances against winners every quarter;
  * it therefore wins when market breadth is wide and loses badly in
    concentration regimes, where a handful of mega caps carry the index (the
    2023-24 AI episode being the extreme case);
  * the spread trends -- concentration regimes persist for years, not weeks --
    which is what a relative-momentum leg is for;
  * but concentration also mean-reverts over much longer horizons, which pulls
    the other way and is what the reversion leg is for;
  * and in market stress equal weight has the higher drawdown, being tilted to
    smaller and more levered names, so volatility is a reason to sit in cap
    weight regardless of what the trend leg says.

The nonlinearity the article asks for is that the two legs are *blended* by a
fitted weight and then *gated* by the volatility regime -- the stress gate
multiplies rather than adds, so a violent tape overrides an attractive trend.

FREQUENCY: month-end bars, so a position can change at most once a month.

WHY THE SCORE IS AN INFORMATION RATIO: both legs are ~95% the same market. The
strategy's total Sharpe is therefore mostly the equity risk premium, which no
rotation rule creates or destroys. `backtest.score` scores a pair on the active
return against holding cap-weight throughout, which reduces exactly to
`w * (ew - cw)` -- the part the rule is responsible for.

VENUE: fitted on RSP/SPY for the history (2003-). A UK-based holder cannot
buy US-domiciled ETFs; the UCITS implementation is XDEW.L (Xtrackers S&P 500
Equal Weight, USD) against CSPX.L, both LSE-listed, and is checked separately as
a trade-leg holdout.
"""

from llmsearch.spec import Strategy


class EwCwRotation(Strategy):
    name = "ew_cw_rotation"
    bars = "M"
    lo, hi = 0.0, 1.0          # weight on equal-weight; remainder to cap-weight

    def param_space(self):
        return {
            "mom_n":    ("int",   1,   18),    # relative-momentum lookback, months
            "mom_th":   ("float", 0.20, 0.80), # momentum percentile that favours EW
            "rev_n":    ("int",  12,   72),    # long-horizon reversion window, months
            "rev_w":    ("float", 0.0,  1.0),  # blend: 0 = pure momentum, 1 = pure reversion
            "vol_n":    ("int",   2,   18),    # realised-vol window, months
            "calm_th":  ("float", 0.10, 0.90), # calmness percentile that permits the EW tilt
            "look":     ("int",  24,  144),    # window all percentiles are taken over
            "gw":       ("float", 0.02, 0.30), # soft-gate width, in percentile units
        }

    def structure(self, d, p, t):
        ratio = d.ratio          # EW price / CW price -- the only series that matters
        mkt = d.b.ret            # cap-weight returns, as the market-stress proxy

        # leg A -- relative momentum: has EW been beating CW lately?
        mom = t.gate(t.pctile(t.roc(ratio, p["mom_n"]), p["look"]),
                     p["mom_th"], p["gw"])

        # leg B -- long-horizon reversion: is EW historically cheap vs CW?
        # A low percentile of ratio-vs-its-own-average means EW is depressed,
        # which the reversion view reads as a reason to own it, hence 1 - x.
        stretch = ratio / t.sma(ratio, p["rev_n"]) - 1.0
        rev = 1.0 - t.pctile(stretch, p["look"])

        core = t.blend(rev, mom, p["rev_w"])       # rev_w*rev + (1-rev_w)*mom

        # gate -- in a violent tape, sit in cap-weight whatever the legs say
        calm = 1.0 - t.pctile(t.rvol(mkt, p["vol_n"]), p["look"])
        stress_ok = t.gate(calm, p["calm_th"], p["gw"])

        return core * stress_ok
