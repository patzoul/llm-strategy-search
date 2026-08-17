"""Bitcoin vs gold rotation -- daily decisions, long-only, always invested.

The signal is the weight on BITCOIN; the remainder goes to gold. The book is
always fully invested in one or the other, so this never decides how much hard
asset to hold, only which one.

MECHANISM:
Both are non-sovereign stores of value with no cash flow, and both are bought
for broadly the same reason -- but they sit at opposite ends of the risk
spectrum, and that is what makes the switch meaningful rather than cosmetic:

  * gold is the defensive leg: ~16% annualised vol, a stress and real-rate
    asset that holds up when risk appetite collapses;
  * bitcoin is the offensive leg: ~66% annualised vol, behaving as a high-beta
    liquidity and risk-appetite asset, with repeated 70-80% drawdowns;
  * their daily correlation over the common sample is 0.090 -- effectively
    independent, which is precisely why rotating between them can do something
    that rotating between two equity styles cannot.

The dominant fact to exploit is not subtle: bitcoin spends long stretches in
multi-month bear phases during which gold is simply the better asset to own. So
the primary lever is a slow trend filter on bitcoin itself. Secondary is broad
risk appetite -- bitcoin's correlation with equities rose sharply after 2020, so
a violent equity tape is information about bitcoin even though it says nothing
about gold.

The three conditions multiply rather than sum: bitcoin must be above its own
slow average, AND beating gold lately, AND the equity tape must be calm. Any one
failing routes the book to gold. That conjunction is the nonlinearity.

WHY DAILY, NOT MONTHLY: the common history starts at bitcoin's Yahoo inception
in September 2014, giving only ~143 months. A monthly test would be the thinnest
in this repo -- thinner than the momentum/min-vol test already flagged as
underpowered. Daily bars on the NYSE calendar give ~2,996 observations instead,
which is a properly powered test. Bitcoin trades continuously while gold does
not, so bars are intersected onto NYSE sessions and bitcoin's Monday return
correctly spans the weekend it was held through.

READ THE VOLATILITY COLUMN, NOT THE WEIGHT. This is a weight-based rotation
between assets whose volatilities differ by 4.1x. With near-zero correlation, a
50/50 *weight* portfolio is about 94% bitcoin *risk*. So "50% in bitcoin" does
not mean balanced, and a strategy that lowers its average bitcoin weight will
cut volatility for reasons that have nothing to do with skill. The fixed-weight
benchmarks and the risk-matched constant-weight comparison exist to make that
visible; a Sharpe improvement that merely reflects holding less bitcoin is not
an edge.

VENUE: bitcoin spot at IBKR is US-only, so a UK holder needs a different venue
for the bitcoin leg, with different spreads than the 15bp assumed here.
"""

from llmsearch.spec import Strategy


class GoldBtcRotation(Strategy):
    name = "gold_btc_rotation"
    bars = "D"
    lo, hi = 0.0, 1.0          # weight on BITCOIN; remainder to gold

    def param_space(self):
        return {
            "trend_n":  ("int",  20,  250),    # bitcoin's own trend filter, days
            "mom_n":    ("int",   5,  120),    # BTC-vs-gold relative momentum, days
            "mom_th":   ("float", 0.20, 0.80), # percentile that counts as beating gold
            "vol_n":    ("int",  10,   90),    # equity realised-vol window, days
            "calm_th":  ("float", 0.10, 0.90), # calmness percentile risk appetite requires
            "look":     ("int", 120,  400),    # window all percentiles are taken over
            "gw":       ("float", 0.02, 0.30), # soft-gate width, in percentile units
        }

    def structure(self, d, p, t):
        btc = d.a.close          # bitcoin price
        ratio = d.ratio          # bitcoin / gold
        mkt_r = d.x.ret          # SPY returns, risk-appetite proxy (not tradeable)

        # -- primary: is bitcoin in an uptrend at all? --------------------
        # A hard step. The 0.0s are structural ("above its own average",
        # "no soft transition"), not fitted magnitudes.
        btc_trend = t.gate(btc / t.sma(btc, p["trend_n"]) - 1.0, 0.0, 0.0)

        # -- has bitcoin been beating gold lately? ------------------------
        rel = t.gate(t.pctile(t.roc(ratio, p["mom_n"]), p["look"]),
                     p["mom_th"], p["gw"])

        # -- is risk appetite intact? -------------------------------------
        calm = t.gate(1.0 - t.pctile(t.rvol(mkt_r, p["vol_n"]), p["look"]),
                      p["calm_th"], p["gw"])

        # Conjunction: any one condition failing sends the book to gold.
        return btc_trend * rel * calm
