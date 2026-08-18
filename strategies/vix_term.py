"""Equity exposure driven by the implied-volatility term structure -- daily.

The signal is the fraction of the book held in SPY; the remainder sits in cash.
This is a "how much equity" strategy, not a "which asset" one, and that alone
makes it different in kind from the six rotations and the two trend overlays
that preceded it.

WHY THIS IS A GENUINELY DIFFERENT MECHANISM:
Every prior strategy in this repo derived its signal from realised price
history -- returns, moving averages, realised volatility, drawdowns, yields.
This one reads the *option market's forward expectation*, which is information
those signals do not contain. The VIX at today's close is a forecast of the next
thirty days; realised volatility is a measurement of the last n. They are not
the same quantity and they diverge in ways that have been studied for decades.

Two documented signals, both forward-looking:

  * TERM STRUCTURE. VIX3M is 3-month implied volatility, VIX is 30-day. In calm
    markets the curve is in contango (VIX3M > VIX) because near-dated options
    are cheap. When it inverts into backwardation the option market is pricing
    imminent stress, and inversion has historically led rather than lagged
    realised drawdowns. That leading property is the whole point: a realised-vol
    gate can only react after the damage, this one can react before.

  * VOLATILITY RISK PREMIUM. Implied volatility sits above subsequently realised
    volatility on average -- the compensation sellers demand for bearing
    variance risk. The premium is the clearest evidence that risk-taking is
    being paid. When it compresses or inverts, the compensation for holding
    equity risk has thinned.

The two multiply: exposure requires an upward-sloping curve AND a healthy
premium. Either condition failing cuts the book toward cash. `floor` lets the
optimiser keep a permanent core holding and time only the remainder -- if it
pushes the floor high, that is informative in itself.

CAUSALITY: both index levels are struck at the same close as the SPY price they
are read alongside, and positions are held from the *next* bar, so nothing here
sees its own future. The look-ahead invariant covers it directly.

DATA: ^VIX3M begins 2006-07-17 and Yahoo's copy currently ends a month behind
SPY, so the usable sample is ~5,030 daily bars over twenty years -- the largest
daily sample in this repo. It spans 2008, the 2018 volatility blow-up, 2020 and
2022, which are the episodes the mechanism exists for.

A NOTE ON THE NULL: this is long/flat on a rising asset, so the sign-flip null
will be the weak bar again -- it deletes the equity drift the strategy collects
for free. The block bootstrap is the one that matters, and it leaves ^VIX and
^VIX3M in their original time order so that scrambling SPY severs the link
between implied volatility and equity returns. That is precisely the hypothesis
under test.
"""

from llmsearch.spec import Strategy


class VixTermStructure(Strategy):
    name = "vix_term_structure"
    bars = "D"
    lo, hi = 0.0, 1.0          # fraction of the book in equities; rest is cash

    def param_space(self):
        return {
            "ts_n":    ("int",   1,   20),    # smoothing of the term-structure ratio, days
            "ts_th":   ("float", 0.10, 0.90), # curve-slope percentile exposure requires
            "rv_n":    ("int",   5,   60),    # realised-vol window for the premium, days
            "vrp_th":  ("float", 0.10, 0.90), # premium percentile exposure requires
            "look":    ("int", 120,  750),    # window all percentiles are taken over
            "gw":      ("float", 0.02, 0.30), # soft-gate width, in percentile units
            "floor":   ("float", 0.0,  0.50), # permanent core equity holding
        }

    def structure(self, d, p, t):
        vix = d.exog("^VIX").close        # 30-day implied volatility, index points
        v3m = d.exog("^VIX3M").close      # 3-month implied volatility, index points

        # -- term structure: contango is calm, backwardation is priced stress --
        slope = t.sma(v3m / vix - 1.0, p["ts_n"])
        contango = t.gate(t.pctile(slope, p["look"]), p["ts_th"], p["gw"])

        # -- volatility risk premium: implied minus subsequently-realised ------
        # The 100 is a unit conversion, not a fitted magnitude: VIX is quoted in
        # percentage points while rvol returns an annualised fraction.
        realised = 100.0 * t.rvol(d.ret, p["rv_n"], 252.0)
        vrp = vix - realised
        premium = t.gate(t.pctile(vrp, p["look"]), p["vrp_th"], p["gw"])

        # Conjunction: an upward curve AND a paid premium. Either failing cuts
        # the book toward cash.
        timed = contango * premium
        return p["floor"] + (1.0 - p["floor"]) * timed
