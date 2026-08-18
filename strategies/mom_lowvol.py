"""Momentum vs minimum-volatility rotation -- month-end, long-only.

The signal is the weight on MOMENTUM (MTUM); the remainder goes to minimum
volatility (USMV). Always 100% invested in US large caps: this strategy chooses
which factor, never how much equity.

MECHANISM -- the best-documented of the rotations tested in this repo:
Momentum does not fail randomly. Daniel & Moskowitz ("Momentum Crashes") show
its expected return turns sharply negative under a specific, *observable*
conjunction: the market is in a drawdown AND volatility is high. The reason is
mechanical -- after a sustained decline the momentum portfolio is implicitly
short the beaten-down high-beta names, so when the market rebounds violently it
is run over. The 2009 and 2020-21 momentum crashes both fit this shape.

Minimum volatility is close to the mirror image: it lags in strong risk-on
rallies and holds up in drawdowns, which is exactly when momentum is exposed.

So the rule is a conjunction, not a sum: hold momentum only when the market is
near its highs AND calm. Either condition failing sends the book to minimum
volatility. That multiplicative gate is the nonlinearity, and it is the part of
the structure with genuine prior support rather than a fitted convenience.

THE MARKET STATE IS EXOGENOUS. Drawdown and volatility are measured on SPY, read
as a conditioning variable only -- it contributes nothing to P&L, and the
surrogate nulls leave it in its original time order so that randomising the
factor spread severs its link to the market state. Clearing the null therefore
means the crash-condition mechanism carries information, not merely that one
factor beat the other.

SAMPLE WARNING: USMV starts in October 2011 and MTUM in April 2013, so the
common history is ~160 months -- by far the shortest of the strategies here. Two
things follow. First, the in-sample window is only ~93 months, which forces
short lookback windows: a 48-month percentile would spend half the sample in
burn-in. The parameter ranges below are capped accordingly, so the strategy
cannot express slow regime views even if they exist. Second, a ~68-month holdout
gives a standard error on an annualised Sharpe of roughly sqrt(12/68) ~ 0.42, so
only very large differences are detectable. A null result here is weaker
evidence of absence than in the longer-sample tests.
"""

from llmsearch.spec import Strategy


class MomLowVolRotation(Strategy):
    name = "mom_lowvol_rotation"
    bars = "M"
    lo, hi = 0.0, 1.0          # weight on MOMENTUM; remainder to min-volatility

    def param_space(self):
        # Windows are deliberately short: see the SAMPLE WARNING above. Maximum
        # burn-in is look + max(mom_n, vol_n) = 24 + 12 = 36 of ~93 in-sample
        # months, which is already more than a third of the fitting window.
        return {
            "mom_n":    ("int",   1,   12),    # factor relative-momentum lookback, months
            "mom_th":   ("float", 0.20, 0.80), # percentile that favours momentum
            "vol_n":    ("int",   2,   12),    # market realised-vol window, months
            "calm_th":  ("float", 0.10, 0.90), # calmness percentile required for momentum
            "dd_th":    ("float", 0.10, 0.90), # drawdown percentile required for momentum
            "look":     ("int",  12,   24),    # window all percentiles are taken over
            "gw":       ("float", 0.02, 0.30), # soft-gate width, in percentile units
        }

    def structure(self, d, p, t):
        ratio = d.ratio          # momentum price / min-vol price
        spy = d.exog("SPY")      # market state proxy, not tradeable here
        mkt, mkt_r = spy.close, spy.ret

        # -- has momentum been beating min-vol lately? --------------------
        rel = t.gate(t.pctile(t.roc(ratio, p["mom_n"]), p["look"]),
                     p["mom_th"], p["gw"])

        # -- the momentum-crash condition ---------------------------------
        # near the highs: drawdown ranked against its own recent history,
        # so 1 = at a peak, 0 = deepest drawdown in the window.
        near_high = t.gate(t.pctile(t.drawdown(mkt), p["look"]), p["dd_th"], p["gw"])
        calm = t.gate(1.0 - t.pctile(t.rvol(mkt_r, p["vol_n"]), p["look"]),
                      p["calm_th"], p["gw"])

        # Conjunction, not a sum: momentum needs BOTH conditions. Either one
        # failing routes the book to minimum volatility.
        return rel * near_high * calm
