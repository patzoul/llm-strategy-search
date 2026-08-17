"""S&P 500 Value vs Growth rotation -- month-end, long-only, rate-conditioned.

The signal is the weight on VALUE; the remainder goes to growth. Always 100%
invested in US large caps: this strategy never decides how much equity to hold,
only which style. IVE and IVW together span the S&P 500, so the entire spread
between them is a style bet, not a market bet.

MECHANISM:
Unlike the equal-weight/cap-weight spread, value-versus-growth has a documented
*exogenous* driver, and it is the reason this strategy is worth a separate test:

  * growth stocks are long-duration equity -- more of their value sits in distant
    cash flows -- so their relative price is rate-sensitive. Falling yields
    lengthen the discount tail and favour growth; rising yields compress it and
    favour value. This was the dominant style driver through 2010-2024, and the
    2022 value reversal coincided exactly with the fastest tightening cycle in
    forty years;
  * the spread also mean-reverts on multi-year horizons as the relative
    valuation gap between the two baskets opens and closes;
  * and it trends over intermediate horizons -- style regimes persist for years.

Three prior strategies in this repo failed using price history alone. The point
of the rate leg is to test an actual economic mechanism rather than re-run the
same momentum-and-reversion shape on a different spread.

THE NONLINEARITY: the rate view does not simply add to the price view. It takes
over only when rates are actually *moving* -- a genuine regime switch, gated on
the magnitude of the recent yield change. In a quiet rate environment the rule
falls back on the price structure; in a violent one it follows duration.

THE EXOGENOUS SERIES IS NOT TRADEABLE. ^TNX is the 10-year Treasury yield, read
only as a conditioning variable. It contributes nothing to P&L, and the
surrogate nulls leave it in its original time order precisely so that
randomising the style spread severs its link to rates -- which is the
hypothesis being tested.

VENUE: IVE/IVW are US-domiciled and closed to UK retail under PRIIPs. UCITS
equivalents exist (IUVL.L / IUGL.L, MSCI USA Value / Growth) but track different
indices, so the trade-leg check is a separate question from whether the signal
works at all.
"""

from llmsearch.spec import Strategy


class ValueGrowthRotation(Strategy):
    name = "value_growth_rotation"
    bars = "M"
    lo, hi = 0.0, 1.0          # weight on VALUE; remainder to growth

    def param_space(self):
        return {
            "mom_n":    ("int",   1,   18),    # style relative-momentum lookback, months
            "mom_th":   ("float", 0.20, 0.80), # momentum percentile that favours value
            "rev_n":    ("int",  12,   72),    # long-horizon reversion window, months
            "rev_w":    ("float", 0.0,  1.0),  # 0 = pure momentum, 1 = pure reversion
            "rate_n":   ("int",   1,   18),    # yield-change lookback, months
            "rate_th":  ("float", 0.20, 0.80), # yield-change percentile that favours value
            "conf_th":  ("float", 0.10, 0.90), # how big a rate move before rates take over
            "look":     ("int",  24,  144),    # window all percentiles are taken over
            "gw":       ("float", 0.02, 0.30), # soft-gate width, in percentile units
        }

    def structure(self, d, p, t):
        ratio = d.ratio          # value price / growth price
        yld = d.x.close          # 10-year Treasury yield, level

        # -- price view -------------------------------------------------
        # leg A: has value been beating growth lately?
        mom = t.gate(t.pctile(t.roc(ratio, p["mom_n"]), p["look"]),
                     p["mom_th"], p["gw"])
        # leg B: is value historically depressed against growth? A low
        # percentile of the stretch reads as cheap, hence 1 - x.
        stretch = ratio / t.sma(ratio, p["rev_n"]) - 1.0
        rev = 1.0 - t.pctile(stretch, p["look"])
        price_view = t.blend(rev, mom, p["rev_w"])

        # -- rate view --------------------------------------------------
        # rising yields compress the growth discount tail -> favour value
        dy = t.roc(yld, p["rate_n"])
        rate_view = t.gate(t.pctile(dy, p["look"]), p["rate_th"], p["gw"])

        # -- regime switch ----------------------------------------------
        # rates only take over when they are genuinely moving; otherwise the
        # price structure decides. This is the multiplicative switch rather
        # than a flat sum of the two views.
        conf = t.gate(t.pctile(t.absv(dy), p["look"]), p["conf_th"], p["gw"])
        return t.blend(rate_view, price_view, conf)
