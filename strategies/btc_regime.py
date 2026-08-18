"""BTC-USD -- daily decisions, long/flat.

MECHANISM (state it before fitting anything, per the house rule):
Bitcoin has a well-documented time-series momentum effect -- long holding
periods of persistent drift punctuated by 70-80% drawdowns -- and the thing that
distinguishes the good momentum periods from the bad ones is the volatility
regime. Trend-following pays when realised vol is in the calm part of its own
history and whipsaws when it is in the violent part; in violent regimes the
short-horizon behaviour flips to mean reversion (forced-liquidation cascades
overshoot and snap back). On top of that sits a slow trend filter whose only job
is to be in cash for the multi-month bear phases.

That is exactly the "trend when calm, mean-revert when volatile" nonlinearity
the article argues LLMs should be asked to *propose*, with the numbers left to
the optimiser.

CONSTRAINT: long/flat only. The intended venue is UK retail, where the FCA bans
crypto derivatives, so there is no short leg to backtest honestly.

Every indicator is turned into a rolling percentile before it meets a threshold.
That is deliberate: a raw threshold fitted on 2015 BTC (annualised vol ~80%,
price $300) means nothing in 2025. A percentile threshold is comparable across
both. The only bare constants below are 0.0 as "above its own average" and 0.0
as "hard step" -- structural choices, not tuned magnitudes.
"""

from llmsearch.spec import Strategy


class BtcVolRegime(Strategy):
    name = "btc_vol_regime"
    bars = "D"
    lo, hi = 0.0, 1.0          # long/flat

    def param_space(self):
        return {
            "fast":     ("int",   5,   60),    # fast MA, days
            "slow":     ("int",  30,  250),    # slow MA + the cash filter, days
            "tr_th":    ("float", 0.20, 0.80), # trend percentile to call it a trend
            "rsi_n":    ("int",   4,   30),    # RSI lookback, days
            "rsi_lo":   ("float", 0.05, 0.50), # RSI percentile that counts as a dip
            "vol_n":    ("int",   7,   60),    # realised-vol window, days
            "vol_look": ("int", 120,  750),    # window all percentiles are taken over
            "calm_th":  ("float", 0.20, 0.85), # calmness percentile splitting the regimes
            "gw":       ("float", 0.02, 0.25), # soft-gate width, in percentile units
        }

    def structure(self, d, p, t):
        c, r = d.close, d.ret

        # regime: where does today's realised vol sit in its own history?
        calm = 1.0 - t.pctile(t.rvol(r, p["vol_n"]), p["vol_look"])
        w_trend = t.gate(calm, p["calm_th"], p["gw"])        # 1 = calm -> trend leg

        # leg A, for calm regimes: fast-over-slow trend, percentile-thresholded
        tr_raw = t.sma(c, p["fast"]) / t.sma(c, p["slow"]) - 1.0
        trend = t.gate(t.pctile(tr_raw, p["vol_look"]), p["tr_th"], p["gw"])

        # leg B, for violent regimes: buy the washed-out RSI
        rsi_p = t.pctile(t.rsi(c, p["rsi_n"]), p["vol_look"])
        mrev = t.gate(p["rsi_lo"] - rsi_p, 0.0, p["gw"])

        core = t.blend(trend, mrev, w_trend)

        # hard filter: never hold below the slow average, whatever the legs say
        cash = t.gate(c / t.sma(c, p["slow"]) - 1.0, 0.0, 0.0)

        return core * cash
