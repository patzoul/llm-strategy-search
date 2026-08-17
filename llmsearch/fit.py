"""Parameter fitting by differential evolution.

The division of labour from the article: the structure is written by hand (by
the LLM), the *numbers* are found here. DE rather than a grid because the
objective is non-smooth (hard gates, quantised integer windows) and 8-9
dimensional, where a grid is either coarse or unaffordable.

`budget` is a named profile rather than raw DE knobs so that the fit used inside
the null bar is provably the *same* fit used on the real data. If the null were
given a cheaper optimiser the bar would be too low, which is the single easiest
way to fool yourself with this method.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import differential_evolution

from . import backtest

BUDGETS = {
    "fast":   dict(maxiter=25, popsize=10, tol=0.02),
    "normal": dict(maxiter=60, popsize=15, tol=0.01),
    "deep":   dict(maxiter=150, popsize=25, tol=0.005),
}


@dataclass
class Fit:
    x: np.ndarray
    train_score: float
    n_evals: int


def fit(strategy, d, cost_bps: float, mask: np.ndarray | None = None,
        seed: int = 0, budget: str = "normal") -> Fit:
    """Maximise net Sharpe over the masked bars. Returns the best parameter vector."""
    counter = [0]

    def obj(x):
        counter[0] += 1
        try:
            sig = strategy.signal(d, x)
        except Exception:
            return 1e6
        s = backtest.score(d, sig, cost_bps, mask)
        return -s if np.isfinite(s) else 1e6

    res = differential_evolution(
        obj, strategy.bounds, seed=seed, polish=False, init="sobol",
        mutation=(0.4, 1.0), recombination=0.8, updating="immediate",
        **BUDGETS[budget])
    return Fit(x=np.asarray(res.x, dtype=float), train_score=-float(res.fun),
               n_evals=counter[0])
