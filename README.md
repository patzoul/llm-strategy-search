# llm-strategy-search

An implementation of the framework in Vincent Maciejewski's *"Why LLMs can't
trade, and how to use them in trading"*, applied to two strategies: **Bitcoin**
(daily decisions) and **IUSE** (month-end decisions, minimum one-month holding
period).

## What the article actually argues

An LLM cannot tell a real market effect from a pattern that happens to look good
in historical data. So do not ask it what to trade. Use it as a **structure
generator inside an optimisation loop**, and split the labour three ways:

| who | does what |
|---|---|
| the LLM | proposes the *shape* of a strategy — which indicators, how they combine, which regime switches. It declares parameter **ranges** and never writes a number. |
| the optimiser | differential evolution finds the parameter **values**. |
| the statistics | blocked cross-validation and a null bar decide whether any of it survived. |

That division is enforced here in code, not by convention: `param_space()`
returns ranges, `structure()` receives already-decoded parameters, and there is
no path by which a strategy can hard-code a magnitude.

## Setup

```bash
pip install -r requirements.txt
```

`bottleneck` is optional but strongly recommended — it backs the
rolling-percentile fast path, and without it the null bars take far longer.
`tools.py` falls back to a pure-numpy implementation, and `selfcheck.py` asserts
the two agree to 1e-10.

Price data is downloaded on demand into `data_cache/`, which is **not** tracked:
a fresh clone re-fetches from Yahoo. Yahoo restates adjusted closes, so figures
may drift slightly from those published in `results/`. Every run prints an
integrity report — date range, bar count, annualised vol, extreme returns — so
any drift is visible rather than silent.

## What is here

```
llmsearch/
  tools.py      causal indicator library — SMA, EMA, ROC, RSI, realised vol,
                z-score/Bollinger, breakout, rolling percentile, drawdown,
                plus the gate/blend combination primitives
  panel.py      the read-only numpy views strategies see: Panel (one asset)
                and Pair (two legs plus an optional exogenous conditioning
                series that carries no P&L)
  spec.py       the strategy contract: param_space() + structure() -> signal
  backtest.py   signal -> position (lagged one bar) -> net return -> metrics;
                pair switches trade both legs, so cost is on 2x the weight change
  fit.py        differential evolution over the declared ranges
  validate.py   blocked-quarter CV, surrogate null bars, deflated Sharpe
  data.py       loading with an integrity report at the boundary
strategies/
  btc_regime.py       BTC-USD, daily, long/flat
  iuse_monthly.py     S&P 500 signal -> IUSE.L, monthly, long/flat
  ew_cw_rotation.py   RSP / SPY, equal weight vs cap weight, monthly
  value_growth.py     IVE / IVW, conditioned on the 10y yield, monthly
  mom_lowvol.py       MTUM / USMV, conditioned on market state, monthly
run.py          the nine-step driver
report.py       print holdout/robustness/costs from a checkpoint, without
                waiting for the null bars to finish
tradeleg.py     apply the fitted EW/CW rule to the UCITS pair a UK investor
                can actually buy
selfcheck.py    36 invariants — run this first
report.html     the written write-up of all five results
results/        run logs, fitted parameters, full surrogate score arrays
```

```bash
python selfcheck.py
```

```bash
python run.py momlv --splits 30 --nulls 40
```

Runs checkpoint after every surrogate, so an interrupted run resumes rather than
restarts — and a resumed run is bit-identical to an uninterrupted one, because
skipped surrogates are still generated to advance the RNG stream.

## The validation protocol

1. **Blocked-quarter CV.** Fit on a random 75% of calendar quarters, score on
   the withheld 25%, repeated many times. Quarters rather than random days, so a
   test bar is not sandwiched between its own training neighbours.
2. **Null bar.** Run the *identical* fit-and-CV pipeline on surrogate series
   with no exploitable structure, and record how good a CV score the procedure
   reaches on noise. The real score must beat that. This is the honest answer to
   "how many things did you try" — it measures the whole search, not one
   backtest.
3. **Holdout.** A date-forward slice, loaded once, at step 7, with no tuning
   afterwards.

Then parameter robustness (plateau or spike?), cost sensitivity, and a deflated
Sharpe as a cross-check.

### Where this departs from the article

* **Two null families, not one.** The article sign-flips returns. For a
  long-only strategy on an asset with a large positive drift, sign-flipping
  removes the drift as well as the structure, which makes the bar too easy — the
  strategy gets credit merely for being long a rising asset. So a **stationary
  block bootstrap** runs alongside it: it preserves the drift *and* the
  volatility clustering and destroys only the ordering. A strategy has to clear
  both bars.
* **Consensus parameters, not the best split's.** Deploying the parameters from
  the best-scoring CV split would be selecting on exactly what the CV is
  supposed to measure. The deployed vector is the per-coordinate median across
  the splits that generalised.
* **Integer parameters are quantised** to at most 40 levels across their range.
  A 300-day and a 301-day moving average are not distinguishable on eight years
  of data; letting the optimiser choose between them is pure overfitting
  surface. (It also makes the indicator cache hit, worth about 10x.)
* **Costs are never zero** and are stress-tested at 0.5x / 1x / 2x / 5x.
* **The traded instrument is separated from the timed one** for IUSE — see
  below.

## The two strategies

### BTC — `btc_vol_regime`, daily, long/flat

Mechanism: Bitcoin has persistent time-series momentum punctuated by 70–80%
drawdowns, and what separates the good momentum periods from the bad ones is the
volatility regime. Trend-following pays when realised vol sits in the calm part
of its own history and whipsaws when it is violent; in violent regimes the
short-horizon behaviour flips towards mean reversion. On top sits a slow trend
filter whose only job is to be in cash for the multi-month bear phases.

That is the article's "trend when calm, mean-revert when volatile" nonlinearity.

Every indicator is converted to a rolling percentile before meeting a threshold.
A raw threshold fitted on 2015 BTC (80% annualised vol, $300 price) means
nothing in 2025; a percentile threshold is comparable across both.

**Long/flat only.** The venue is UK retail and the FCA bans crypto derivatives for
retail, so there is no short leg that could honestly be backtested.

### IUSE — `iuse_monthly_trend`, month-end, long/flat

Mechanism: absolute momentum plus a trend filter (historically a drawdown
reducer far more than a return raiser), multiplied by a volatility-percentile
scaler (realised vol is strongly autocorrelated, and compensation per unit of
risk is higher when vol is low). The two multiply rather than sum — that is the
nonlinearity.

The panel handed to the strategy is **month-end bars**, so every window is in
months and a position can change at most once a month. The one-month minimum
holding period is structural, not a constraint bolted on afterwards.

**What IUSE actually is:** `IUSE.L` is the *iShares S&P 500 EUR-Hedged UCITS ETF
(Acc)*, quoted in **EUR** on the LSE. Two consequences:

* The signal is computed on the **S&P 500** (via SPY, total-return adjusted,
  from 1993) because that is what is being timed and it has three decades of
  history rather than IUSE.L's fifteen years.
* The fitted rule is then applied to the **actual IUSE.L price series** as a
  second holdout, executed at the first IUSE.L close *after* each decision —
  SPY's month-end close is struck at 21:00 London, after IUSE.L has already
  closed, so its own month-end close is not an executable price. The driver
  asserts every decision date is strictly before its execution date.
* For a sterling-based holder the USD exposure is hedged to EUR but EUR/GBP is
  not hedged. If the intent was unhedged USD S&P 500 exposure, `CSPX.L` or
  `IUSA.L` is the instrument, and the same signal applies unchanged.

## Results — six strategies, none validated

| strategy | CV test | sign-flip null (p) | block-boot null (p) | holdout |
|---|---|---|---|---|
| BTC daily vol-regime | +0.780 | 0.095 | 0.190 | Sharpe 0.96 vs risk-matched 0.86 |
| IUSE monthly trend | +0.673 | 0.038 | 0.346 | lost to buy&hold in all 13 years |
| EW vs CW rotation | +0.088 | 0.231 | 0.654 | active −1.56%/yr, IR −0.49 |
| Value vs growth (rate-conditioned) | −0.061 | 0.714 | 0.619 | active −0.51%/yr, IR −0.04 |
| Momentum vs min-vol (crash-conditioned) | **+1.010** | **0.049** | **0.073** | Sharpe 0.89 beats all benchmarks, IR +0.40 (t=0.94) |
| Bitcoin vs gold, daily | +0.912 | 0.065 | **0.452** | Sharpe 0.70 vs 0.91 risk-matched, 1.07 for gold alone |

### Average exposure is a misleading summary

The bitcoin/gold rotation lost 24.5% in 2022 while averaging a **3%** bitcoin
weight, in a year gold was flat. Not a bug: it held a full-size bitcoin position
on exactly 8 days, and bitcoin averaged **−3.5%** on those days against −0.33%
across the year — including 82% weight on 8 November, the day FTX collapsed. A
strategy that is usually flat and occasionally all-in is not well described by
its mean weight. Report the position-weighted return of the leg alongside it.

### The null bar must be sized before you look at it

Momentum vs min-vol is the cautionary tale. At **20** surrogates it cleared both
null maxima with p=0.048 — a clean pass on every pre-registered gate. But 0.048
is 1/21, the *resolution floor* of a 20-surrogate test: the real score beating
every draw cannot produce a smaller number no matter how strong it is.

Extending to **40** surrogates, with the target fixed in advance:

| | n=20 | n=40 |
|---|---|---|
| sign-flip | 0 beat, p=0.048 | 1 beat, **p=0.049** |
| block bootstrap | 0 beat, p=0.048 | 2 beat, **p=0.073** |

Both p-values got *worse* with more evidence, and the block-bootstrap maximum
went from +0.993 to +1.638 — far above the real +1.010. A genuine effect
sharpens as surrogates accumulate; a resolution artifact degrades. Choose
`--nulls` from the p-value you need to resolve (n=20 can never show better than
0.048; n=40 gets to 0.024) and fix it *before* seeing the result.

## Earlier results — four strategies, four kills

| strategy | CV test | sign-flip null (max / p) | block-boot null (max / p) | holdout |
|---|---|---|---|---|
| BTC daily vol-regime | +0.780 | +0.829 / 0.095 | +1.241 / 0.190 | Sharpe 0.96 vs risk-matched 0.86 |
| IUSE monthly trend | +0.673 | +0.639 / 0.038 | +1.075 / 0.346 | lost to buy&hold in all 13 years |
| EW vs CW rotation | +0.088 | +0.360 / 0.231 | +0.749 / 0.654 | active −1.56%/yr, IR −0.49 |
| Value vs growth (rate-conditioned) | **−0.061** | +0.566 / 0.714 | +0.903 / 0.619 | active −0.51%/yr, IR −0.04 |

Every one is inside its block-bootstrap null — the bar the article does not use.
For the two rotations the null *mean* exceeds the real cross-validated score: the
average structureless surrogate beats the real data.

Recurring findings across all four:

* **The sign-flip null is too weak.** IUSE cleared it at p=0.038 and would have
  been reported as a pass. Flipping return signs removes the asset's drift along
  with its structure, which flatters any long-only strategy on a rising asset.
* **Spreads do not persist at monthly horizons.** EW−CW autocorrelation is +0.113
  at lag 1 and +0.003 at lag 3; value−growth is +0.155 and +0.012. There is no
  trend to follow at a frequency a monthly rule can act on.
* **Both style spreads flip sign between the two windows** — EW beat CW by
  +1.94%/yr in-sample and lost 2.83%/yr out; value beat growth by +2.00%/yr
  in-sample and lost 5.25%/yr out. Anything fitted on the first window learned
  the wrong prior for the second.
* **The optimiser discarded the mean-reversion leg both times** (`rev_w` fitted
  to 0.089 and 0.001) and pinned `look` and `gw` to their lower bounds. A
  parameter jammed against the edge of its declared box is a sign the structure
  cannot express what the search is chasing.
* **The opportunity is real; the rules capture none of it.** Perfect-foresight
  oracles earn +6.25%/yr (EW/CW) and +12.12%/yr (value/growth) in active return.

## Detailed results (2026-08-12)

Both fail their pre-registered kill criteria. The decisive test in each case is
the **block-bootstrap null**, which is the bar the article does not use.

| | CV test Sharpe | sign-flip null (mean / max / p) | block-boot null (mean / max / p) |
|---|---|---|---|
| BTC daily | +0.780 | +0.292 / +0.829 / **0.095** | +0.399 / **+1.241** / **0.190** |
| IUSE monthly | +0.673 | +0.031 / +0.639 / 0.038 | +0.501 / **+1.075** / **0.346** |

Run the identical fit-and-CV pipeline on surrogates that keep each asset's drift
and volatility clustering but destroy the time ordering, and it produces
cross-validated scores *at least as good as the real data* 19% of the time for
BTC and 35% for IUSE. Neither CV score is evidence of an effect.

IUSE clears the sign-flip bar (p=0.038) and would have been reported as a pass
had only the article's null been run. That is the single most important finding
here: for a long-only strategy on an asset with a large positive drift,
sign-flipping removes the drift along with the structure and sets the bar far too
low.

Holdouts, for completeness:

| holdout | CAGR | vol | Sharpe | maxDD | avg exp |
|---|---|---|---|---|---|
| BTC strategy 2023–26 | 21.4% | 22.8% | 0.96 | −20.7% | 44% |
| BTC constant 44% (risk-matched) | 14.1% | 17.1% | 0.86 | −27.0% | 44% |
| BTC buy & hold | 29.3% | 38.7% | 0.86 | −53.1% | 100% |
| IUSE strategy 2014–26 (on SPY) | 7.9% | 10.7% | 0.77 | −22.5% | 82% |
| IUSE buy & hold | 13.8% | 14.5% | 0.97 | −23.9% | 100% |

IUSE lost to buy-and-hold in all 13 holdout years, including 2022 — the one bear
market, the scenario the rule exists for — where it averaged 32% exposure and
still lost more than the index by de-risking after the fall.

BTC's holdout looks good, but against the risk-matched constant-44% benchmark
the edge is 0.96 vs 0.86 Sharpe, and it comes almost entirely from one event:
2025–26, when BTC fell 32% and the strategy was flat. Deflated Sharpe 0.145
across 1,026 DE fits. `vol_n`'s ±20% neighbours score +1.24/+1.34 against the
deployed +0.96, so the holdout number is a draw from a wide distribution.

Raw output: `results/{btc,iuse}.{log,json}`; resumable state in `*.state.json`.

## Reading the output

Numbers to distrust, in order:

* a CV score inside either null bar — the search found noise;
* a large train-minus-test optimism gap;
* a parameter whose ±20% neighbours collapse (a spike, not a plateau);
* a holdout Sharpe that only survives at 0.5x costs;
* a strategy that beats buy-and-hold on Sharpe purely by being out of the market
  during one crash — check `inMkt` and which years drove it.

`selfcheck.py` must pass before any of this means anything. The look-ahead test
rewrites the entire future of the price series and asserts the signal history up
to the cut is byte-identical; if it fails, every result here is void.
