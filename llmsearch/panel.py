"""The immutable numpy view of a price series that strategies actually see.

Strategies get a `Panel`, not a DataFrame, for two reasons: it is the numpy
representation the inner loop needs, and it is read-only, so a strategy cannot
accidentally mutate the data it is being scored on.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

_TOK = [0]


def next_token() -> int:
    _TOK[0] += 1
    return _TOK[0]


@dataclass(frozen=True)
class Panel:
    index: pd.DatetimeIndex
    close: np.ndarray
    high: np.ndarray
    low: np.ndarray
    open: np.ndarray
    volume: np.ndarray
    ret: np.ndarray          # simple returns, ret[0] = 0
    ppy: float
    name: str = ""
    tok: int = 0

    def __len__(self):
        return len(self.close)

    @staticmethod
    def from_frame(df: pd.DataFrame, ppy: float, name: str = "") -> "Panel":
        c = np.ascontiguousarray(df["Close"].to_numpy(dtype=float))
        r = np.zeros_like(c)
        r[1:] = c[1:] / c[:-1] - 1.0

        def col(k):
            return (np.ascontiguousarray(df[k].to_numpy(dtype=float))
                    if k in df.columns else c.copy())

        arrs = [c, col("High"), col("Low"), col("Open"), col("Volume"), r]
        for a in arrs:
            a.flags.writeable = False
        return Panel(index=pd.DatetimeIndex(df.index), close=arrs[0], high=arrs[1],
                     low=arrs[2], open=arrs[3], volume=arrs[4], ret=arrs[5],
                     ppy=ppy, name=name or df.attrs.get("ticker", ""),
                     tok=next_token())

    def slice(self, mask: np.ndarray) -> "Panel":
        """A new panel over a boolean mask, with its own cache token.

        Only used for reporting. Fitting never slices -- it masks which bars are
        *scored*, because slicing would change every rolling window and make the
        in-sample and out-of-sample signals incomparable.
        """
        idx = np.asarray(mask, dtype=bool)
        df = pd.DataFrame({"Close": self.close[idx], "High": self.high[idx],
                           "Low": self.low[idx], "Open": self.open[idx],
                           "Volume": self.volume[idx]}, index=self.index[idx])
        return Panel.from_frame(df, self.ppy, self.name)

    def series(self, a: np.ndarray) -> pd.Series:
        return pd.Series(np.asarray(a), index=self.index)


@dataclass(frozen=True)
class Pair:
    """Two aligned assets for a rotation strategy.

    The signal is the weight on `a`; `b` gets the remainder. Both legs are
    always fully invested, so this is long-only and always in the market -- the
    strategy chooses *which* index, never how much equity.

    `spread` is the thing the strategy is actually forecasting: everything else
    in the two return streams is the common market factor, which no rotation
    rule can influence.
    """

    a: Panel
    b: Panel
    ppy: float
    name: str = ""
    tok: int = 0
    x: "Panel | None" = None    # optional exogenous context (e.g. a yield series)

    def __len__(self):
        return len(self.a)

    @property
    def index(self) -> pd.DatetimeIndex:
        return self.a.index

    @property
    def ret_a(self) -> np.ndarray:
        return self.a.ret

    @property
    def ret_b(self) -> np.ndarray:
        return self.b.ret

    @property
    def close(self) -> np.ndarray:
        return self.a.close

    @property
    def ret(self) -> np.ndarray:
        """Relative return of a over b -- what a rotation rule earns per unit of tilt."""
        return self.a.ret - self.b.ret

    @property
    def ratio(self) -> np.ndarray:
        return self.a.close / self.b.close

    @staticmethod
    def build(fa: pd.DataFrame, fb: pd.DataFrame, ppy: float, name: str = "",
              fx: pd.DataFrame | None = None) -> "Pair":
        """`fx` is an optional exogenous conditioning series (not tradeable).

        It is aligned onto the same bars but is never part of the P&L -- the
        strategy may read it, and the surrogate nulls deliberately leave it
        untouched so that randomising the pair breaks its relationship with the
        exogenous series. Resampling it alongside the pair would preserve that
        relationship and test the wrong hypothesis.
        """
        idx = fa.index.intersection(fb.index)
        if fx is not None:
            idx = idx.intersection(fx.index)
        a = Panel.from_frame(fa.loc[idx], ppy, fa.attrs.get("ticker", "A"))
        b = Panel.from_frame(fb.loc[idx], ppy, fb.attrs.get("ticker", "B"))
        x = Panel.from_frame(fx.loc[idx], ppy, fx.attrs.get("ticker", "X")) if fx is not None else None
        return Pair(a=a, b=b, ppy=ppy, name=name or f"{a.name}/{b.name}",
                    tok=next_token(), x=x)

    def with_legs(self, fa: pd.DataFrame, fb: pd.DataFrame, name: str) -> "Pair":
        """Replace the tradeable legs, carrying the exogenous series through unchanged."""
        idx = fa.index
        a = Panel.from_frame(fa, self.ppy, self.a.name)
        b = Panel.from_frame(fb, self.ppy, self.b.name)
        return Pair(a=a, b=b, ppy=self.ppy, name=name, tok=next_token(), x=self.x)
