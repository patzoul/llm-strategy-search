"""Small cap vs large cap rotation -- month-end, long-only.

The signal is the weight on SMALL CAPS (IWM, Russell 2000); the remainder goes
to large caps (IWB, Russell 1000). Same index provider, same methodology, and
the two segments together are the Russell 3000 -- so this is a true like-for-like
size split rather than two funds that happen to hold different things.

MECHANISM:
The size premium itself is largely gone. Fama-French SMB has been close to zero
since the 1980s, and the spread here is +1.15%/yr over 26 years at 1.25x the
volatility -- not a premium worth harvesting statically. What remains is
cyclicality, and it is well documented:

  * small caps are higher beta and more operationally levered, so they lead in
    early-cycle recoveries and lag in downturns;
  * they are more credit-sensitive -- smaller firms carry more floating-rate
    debt and have poorer access to bond markets -- so tightening credit hits
    them first;
  * they lose badly in concentration regimes, when a handful of mega caps carry
    the index, which describes most of 2015-2024;
  * and the relative valuation gap between the two mean-reverts over multi-year
    cycles.

So the structure pairs a momentum leg with a long-horizon cheapness leg, gated
by market volatility: small caps are the risk-on expression, and a violent tape
is a reason to be in large caps whatever the other legs say.

A NOTE ON THE CONDITIONING SERIES: credit spreads would be the better-motivated
conditioner -- they are the most direct channel by which the cycle reaches small
caps. HYG only starts in April 2007, which would cost a third of the sample, so
market volatility on SPY is used instead as the risk-appetite proxy. That is a
deliberate trade of mechanism quality for statistical power, made because the
shorter-sample tests in this repo have been the least conclusive ones.

HONEST NOTE ON NOVELTY: this is the momentum + reversion + volatility-gate shape
already used for the equal-weight/cap-weight and value/growth rotations, both of
which failed. It is a fair test of whether the size spread has exploitable
structure, but it is not an independent test of a new idea, and the prior going
in should be low. The structures available from a causal price-only indicator
library are not unlimited.
"""

from llmsearch.spec import Strategy


class SmallLargeRotation(Strategy):
    name = "small_large_rotation"
    bars = "M"
    lo, hi = 0.0, 1.0          # weight on small caps; remainder to large caps

    def param_space(self):
        return {
            "mom_n":    ("int",   1,   18),    # relative-momentum lookback, months
            "mom_th":   ("float", 0.20, 0.80), # percentile that favours small caps
            "rev_n":    ("int",  12,   72),    # long-horizon reversion window, months
            "rev_w":    ("float", 0.0,  1.0),  # 0 = pure momentum, 1 = pure reversion
            "vol_n":    ("int",   2,   18),    # market realised-vol window, months
            "calm_th":  ("float", 0.10, 0.90), # calmness percentile small caps require
            "look":     ("int",  24,  144),    # window all percentiles are taken over
            "gw":       ("float", 0.02, 0.30), # soft-gate width, in percentile units
        }

    def structure(self, d, p, t):
        ratio = d.ratio          # small-cap price / large-cap price
        mkt_r = d.exog("SPY").ret   # risk-appetite proxy, not tradeable

        # leg A -- have small caps been beating large lately?
        mom = t.gate(t.pctile(t.roc(ratio, p["mom_n"]), p["look"]),
                     p["mom_th"], p["gw"])

        # leg B -- are small caps historically depressed against large? A low
        # percentile of the stretch reads as cheap, hence 1 - x.
        stretch = ratio / t.sma(ratio, p["rev_n"]) - 1.0
        rev = 1.0 - t.pctile(stretch, p["look"])

        core = t.blend(rev, mom, p["rev_w"])

        # gate -- small caps are the risk-on expression; a violent tape sends
        # the book to large caps whatever the legs say.
        calm = t.gate(1.0 - t.pctile(t.rvol(mkt_r, p["vol_n"]), p["look"]),
                      p["calm_th"], p["gw"])

        return core * calm
