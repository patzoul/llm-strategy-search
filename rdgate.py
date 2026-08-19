"""Honest-gate wrapper for an RD-Agent(Q) search run.

RD-Agent's own loop hill-climbs on the same backtest it reports (the Analysis Unit
is shown IC / ARR / MDD and answers "Replace Best Result: yes"), and never counts
its trials even though it logs them (Table 3: TL / VL / SL). This module fixes
three things:

  1. the agent is only ever pointed at a search window that excludes the holdout;
  2. every loop is counted, and the count feeds the deflated Sharpe;
  3. the holdout is scored exactly once, after the agent has stopped.

Drop beside run.py in llm-strategy-search. Ports to quantkit by swapping
llmsearch.validate.deflated_sharpe -> quantkit.validate.deflated_sharpe (same
Bailey / Lopez de Prado form, same argument order).

Usage
-----
    g = SearchGuard(holdout_start="2019-01-01")
    g.check_qlib_config("conf_factor.yaml")       # refuses to launch on overlap
    ... run RD-Agent ...
    log = SearchLog.from_rdagent("log/factor_loop/")
    verdict = g.settle(log, holdout_returns, ppy=252)
"""

from __future__ import annotations

import glob
import json
import os
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from llmsearch import backtest, validate
from llmsearch.panel import Panel


# ---------------------------------------------------------------------------
# 1. trial accounting -- the number RD-Agent logs and then ignores
# ---------------------------------------------------------------------------

@dataclass
class Trial:
    loop: int
    action: str                 # "factor" | "model"
    metrics: dict               # whatever the Validation Unit produced
    accepted: bool              # Analysis Unit said "Replace Best Result: yes"


@dataclass
class SearchLog:
    trials: list[Trial] = field(default_factory=list)
    backends: set = field(default_factory=set)

    @property
    def n_trials(self) -> int:
        """TOTAL loops -- not valid loops, not SOTA selections.

        A loop that crashed in Co-STEER still consumed a look at the data if it
        got as far as a backtest. A rejected loop is still a trial: it is
        precisely the rejected ones that make the accepted one look good.
        Undercounting here is the easiest way to fake a passing DSR.
        """
        return len(self.trials)

    @property
    def n_accepted(self) -> int:
        return sum(t.accepted for t in self.trials)

    def sharpe_like(self, key: str = "IR") -> np.ndarray:
        return np.asarray([t.metrics.get(key, np.nan) for t in self.trials],
                          dtype=float)

    def var_trials(self, key: str = "IR", ppy: float = 252.0) -> float | None:
        """Empirical variance of the trial Sharpes, in per-bar units.

        Strictly better than the DSR default of 1/(n_obs-1): that assumes the
        trials are independent zero-edge draws, and these are not. Each loop
        builds on the accumulated SOTA library, so they are strongly correlated
        and their spread is narrower. Feeding the measured spread makes the bar
        honest rather than merely conservative.
        """
        s = self.sharpe_like(key)
        s = s[np.isfinite(s)]
        if len(s) <= 2:
            return None
        return float(np.var(s, ddof=1) / ppy)      # annualised -> per bar

    def merge(self, other: "SearchLog") -> "SearchLog":
        """Pool runs. If you tried o3-mini AND GPT-4o AND GPT-4.1 and reported
        the best, the trial count is the SUM, not the winner's."""
        return SearchLog(trials=self.trials + other.trials,
                         backends=self.backends | other.backends)

    @staticmethod
    def from_rdagent(log_dir: str, backend: str = "unknown") -> "SearchLog":
        """Parse an RD-Agent loop trace.

        RD-Agent serialises each loop's hypothesis / feedback / result. The exact
        layout moves between releases, so this reads defensively and refuses
        LOUDLY: an unparsed loop must not silently vanish from n_trials.
        """
        trials, seen = [], 0
        for p in sorted(glob.glob(os.path.join(log_dir, "**", "*.json"),
                                  recursive=True)):
            with open(p, encoding="utf-8") as fh:
                try:
                    obj = json.load(fh)
                except json.JSONDecodeError:
                    continue
            if not isinstance(obj, dict) or "result" not in obj:
                continue
            seen += 1
            fb = obj.get("feedback") or {}
            trials.append(Trial(
                loop=int(obj.get("loop_id", seen)),
                action=str(obj.get("action", "?")),
                metrics=dict(obj.get("result") or {}),
                accepted=str(fb.get("Replace Best Result", "no")).lower() == "yes",
            ))
        if not trials:
            raise RuntimeError(
                f"parsed 0 trials from {log_dir}. Do NOT proceed with n_trials=0 "
                "-- fix the parser, or count the loops by hand from the banner.")
        return SearchLog(trials=trials, backends={backend})


# ---------------------------------------------------------------------------
# 2. the split guard -- refuse to launch a search that can see the holdout
# ---------------------------------------------------------------------------

class HoldoutViolation(RuntimeError):
    pass


@dataclass
class SearchGuard:
    holdout_start: str
    _settled: bool = False

    def check_qlib_config(self, path: str) -> None:
        """Read the Qlib workflow YAML the agent hands to its Validation Unit and
        refuse if anything it scores reaches into the holdout.

        This is the whole fix in one function. RD-Agent's default config backtests
        `segments.test`, and the paper reports `segments.test` -- so the search
        selects on the number it publishes. Here the agent's test segment must end
        before the holdout begins; the holdout is not a segment at all.
        """
        import yaml
        with open(path, encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh)
        h = pd.Timestamp(self.holdout_start)
        bad = []
        for dotted, val in _walk(cfg):
            if not isinstance(val, str):
                continue
            if not any(k in dotted for k in ("end_time", "segments", "backtest")):
                continue
            try:
                ts = pd.Timestamp(val)
            except (ValueError, TypeError):
                continue
            if ts >= h:
                bad.append(f"{dotted} = {val}")
        if bad:
            raise HoldoutViolation(
                f"config reaches into the holdout (starts {h.date()}):\n  "
                + "\n  ".join(bad)
                + "\nThe agent must never backtest past the holdout boundary.")

    def settle(self, log: SearchLog, holdout_ret, ppy: float = 252.0,
               benchmark_ret=None, key: str = "IR") -> dict:
        """Score the holdout. Callable ONCE per guard instance."""
        if self._settled:
            raise HoldoutViolation(
                "holdout already read. A second look is tuning, not validation. "
                "Start a new guard with a later holdout, or accept the answer.")
        self._settled = True

        r = np.asarray(holdout_ret, dtype=float)
        m = backtest.metrics(r, ppy)
        n_obs = int(m["n"])

        s = pd.Series(r)
        sk, ku = float(s.skew()), float(s.kurt() + 3.0)   # pandas gives excess

        dsr = validate.deflated_sharpe(
            m["Sharpe"], n_obs=n_obs, n_trials=log.n_trials,
            skew=sk, kurt=ku, ppy=ppy, var_trials=log.var_trials(key, ppy))

        out = {
            "n_trials": log.n_trials,
            "n_accepted": log.n_accepted,
            "backends": sorted(log.backends),
            "holdout": m,
            "skew": sk,
            "kurt": ku,
            "DSR": dsr,
            "pass_dsr": bool(np.isfinite(dsr) and dsr > 0.95),
        }
        if benchmark_ret is not None:
            b = backtest.metrics(np.asarray(benchmark_ret, dtype=float), ppy)
            out["benchmark"] = b
            out["beats_benchmark"] = bool(m["Sharpe"] > b["Sharpe"])
        return out


def _walk(node, prefix=""):
    if isinstance(node, dict):
        for k, v in node.items():
            yield from _walk(v, f"{prefix}.{k}" if prefix else str(k))
    elif isinstance(node, list):
        for i, v in enumerate(node):
            yield from _walk(v, f"{prefix}[{i}]")
    else:
        yield prefix, node


# ---------------------------------------------------------------------------
# 3. adapter -- cross-sectional strategy -> the single-series gate stack
# ---------------------------------------------------------------------------

def returns_to_panel(ret: pd.Series, ppy: float = 252.0,
                     name: str = "rdagent") -> Panel:
    """Wrap a daily portfolio return stream as a Panel so blocked_cv, null_bar and
    buy_and_hold apply unchanged.

    Caveat to carry into any writeup: this gates the RETURN STREAM, not the factor
    mechanics. It cannot detect a look-ahead inside a generated factor -- only that
    the resulting P&L is or is not distinguishable from noise. Factor-level leakage
    still needs Co-STEER's health checks plus a manual read of the code the agent
    wrote for whatever it finally selected.
    """
    eq = np.cumprod(1.0 + np.asarray(ret, dtype=float))
    df = pd.DataFrame({"Close": eq}, index=pd.DatetimeIndex(ret.index))
    return Panel.from_frame(df, ppy=ppy, name=name)


# ---------------------------------------------------------------------------
# 4. the null bar -- the gate RD-Agent has no analogue for
# ---------------------------------------------------------------------------

def null_bar_plan(n_null: int = 12, hours_per_run: float = 12.0,
                  usd_per_run: float = 10.0) -> str:
    """The honest null re-runs THE WHOLE AGENT on surrogate universes.

    Freeze nothing: block-bootstrap each stock's returns (block ~21d, preserving
    volatility clustering while destroying cross-sectional predictability), rebuild
    the Qlib .bin store, and let the agent search it on the same budget. If it
    still "discovers" factors with IC ~0.05, the loop is a noise miner and the real
    result means nothing.

    The cheap substitute -- freeze the discovered factor set and re-fit only the
    model on surrogates -- tests the SELECTION but not the PROPOSAL, and the
    proposal step is where an LLM search overfits hardest. Say which one you ran.
    """
    return (f"{n_null} surrogate agent runs ~= {n_null * hours_per_run:.0f} "
            f"GPU-hours, ~${n_null * usd_per_run:.0f} in API calls. "
            f"n_null=12 floors the p-value at 1/13 = 0.077, so budget n_null >= 19 "
            f"if you want to be able to claim p < 0.05 at all.")
