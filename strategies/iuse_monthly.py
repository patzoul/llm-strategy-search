"""IUSE -- month-end decisions, long/flat, minimum holding period one month.

MECHANISM:
Two effects, both long-documented on the S&P 500 and both weak individually:
  * absolute (time-series) momentum -- a 12-ish month positive return and a
    price above its own long average filter out the slow bear markets. It has
    historically cut drawdowns far more than it raised returns.
  * the low-volatility effect in *time* -- realised vol is strongly
    autocorrelated, and the compensation per unit of risk is higher when vol is
    low than when it is high, so scaling exposure by where vol sits in its own
    history raises risk-adjusted return without a directional forecast.
The nonlinearity is that the vol scaling only bites when the trend gate is on;
the two multiply rather than sum.

FREQUENCY: the panel handed to this strategy is month-end bars, so every window
below is in *months* and a position can only change once a month. The one-month
minimum holding period is structural, not a constraint bolted on afterwards.

The `floor` parameter lets the optimiser keep a permanent core holding and time
only the remainder. If it wants to, that is informative: it means the timing
overlay is only worth applying to part of the position.

WHAT IUSE ACTUALLY IS: IUSE.L is the iShares S&P 500 EUR-Hedged UCITS ETF (Acc),
quoted in EUR on the LSE. The USD exposure is hedged to EUR, so its EUR return
tracks the S&P 500 in USD less the USD-EUR rate differential -- but a
sterling-based holder still carries EUR/GBP. The signal is therefore computed on
the underlying S&P 500 (long history, and it is what is being timed) and the
fitted rule is applied to the actual IUSE.L price series as a second holdout.
"""

from llmsearch.spec import Strategy


class IuseMonthlyTrend(Strategy):
    name = "iuse_monthly_trend"
    bars = "M"
    lo, hi = 0.0, 1.0          # long/flat, no leverage

    def param_space(self):
        return {
            "trend_n":  ("int",   2,   18),    # trend MA, months
            "trend_w":  ("float", 0.002, 0.08),# soft-gate width on the price/MA ratio
            "mom_n":    ("int",   1,   15),    # absolute-momentum lookback, months
            "mom_th":   ("float", -0.06, 0.10),# return threshold that counts as positive
            "vol_n":    ("int",   3,   24),    # realised-vol window, months
            "vol_look": ("int",  24,  200),    # history the vol percentile is taken over
            "vol_beta": ("float", 0.0, 1.0),   # how hard vol scales exposure (0 = not at all)
            "floor":    ("float", 0.0, 0.50),  # permanent core holding
        }

    def structure(self, d, p, t):
        c, r = d.close, d.ret

        # gate 1 -- price above its own moving average
        trend = t.gate(c / t.sma(c, p["trend_n"]) - 1.0, 0.0, p["trend_w"])

        # gate 2 -- absolute momentum over the last mom_n months
        mom = t.gate(t.roc(c, p["mom_n"]), p["mom_th"], p["trend_w"])

        # scale -- calm months get more exposure than violent ones
        calm = 1.0 - t.pctile(t.rvol(r, p["vol_n"]), p["vol_look"])
        scale = (1.0 - p["vol_beta"]) + p["vol_beta"] * calm

        timed = trend * mom * scale
        return p["floor"] + (1.0 - p["floor"]) * timed
