"""The strategy contract.

This is the whole point of the article's framework: the LLM writes `structure()`
and declares `param_space()`, and it *never writes a number*. Every constant a
strategy would want is a declared range, and the optimiser picks the value.

A strategy is therefore a pure function of (panel, parameter vector) -> signal
in [lo, hi], with no state, no fitted objects and no numbers.

Integer parameters are quantised onto at most `MAX_LEVELS` values across their
declared range. That is not only a speed trick (it makes the indicator cache
hit, which is worth ~10x): a 300-day and a 301-day moving average are not
distinguishable on 8 years of data, so letting the optimiser choose between them
is pure overfitting surface.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import tools

MAX_LEVELS = 40


@dataclass
class Param:
    kind: str      # "int" or "float"
    lo: float
    hi: float

    @property
    def step(self) -> int:
        if self.kind != "int":
            return 1
        span = int(round(self.hi - self.lo))
        return max(1, int(np.ceil(span / MAX_LEVELS)))

    def decode(self, u: float) -> float:
        v = float(np.clip(u, self.lo, self.hi))
        if self.kind != "int":
            return v
        k = int(round((v - self.lo) / self.step))
        return int(min(self.hi, self.lo + k * self.step))


class Strategy:
    """Subclass, set `name`/`bars`, implement `param_space` and `structure`."""

    name = "unnamed"
    bars = "D"          # "D" daily decisions, "M" month-end decisions
    lo, hi = 0.0, 1.0   # signal bounds; long-only by default

    # -- to be provided by the subclass ------------------------------------

    def param_space(self) -> dict[str, tuple]:
        raise NotImplementedError

    def structure(self, d, p: dict, t) -> np.ndarray:
        raise NotImplementedError

    # -- plumbing ----------------------------------------------------------

    def __init__(self):
        self._params = {k: Param(*v) for k, v in self.param_space().items()}

    @property
    def params(self) -> dict[str, Param]:
        return self._params

    @property
    def bounds(self) -> list[tuple[float, float]]:
        return [(p.lo, p.hi) for p in self._params.values()]

    @property
    def names(self) -> list[str]:
        return list(self._params.keys())

    def decode(self, x) -> dict:
        if isinstance(x, dict):
            return x
        return {n: p.decode(v) for (n, p), v in zip(self._params.items(), x)}

    def signal(self, d, x) -> np.ndarray:
        """Full pipeline: decode -> structure -> clip. Never call structure directly."""
        tools.set_context(d.tok)
        sig = self.structure(d, self.decode(x), tools)
        return tools.clip_signal(sig, self.lo, self.hi)

    def describe(self, x) -> str:
        return ", ".join(f"{k}={v:.3g}" if isinstance(v, float) else f"{k}={v}"
                         for k, v in self.decode(x).items())
