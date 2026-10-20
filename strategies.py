"""
Quoting strategies.

Every strategy returns, for each simulation, how far below the mid to bid
(delta_b) and how far above the mid to offer (delta_a).

Avellaneda-Stoikov (2008)
-------------------------
A market maker with exponential utility (risk aversion gamma) holding q shares
values the stock at its *reservation price*

    r = s - q * gamma * sigma^2 * (T - t)

below the mid when long, above it when short: an extra share is worth less to
someone already carrying too many. The optimal total spread is

    delta_a + delta_b = gamma * sigma^2 * (T - t) + (2 / gamma) * ln(1 + gamma / k)

and quotes are centred on r rather than on s. The first term is an inventory-risk
premium that shrinks to zero at the close; the second is the pure
order-flow term (as gamma -> 0 it tends to 2 / k, the spread that maximises
expected profit per unit time when risk does not matter).

Centring on r *skews* the quotes: when long, both quotes move down, so the
market maker is more likely to sell and less likely to buy, pulling inventory
back to zero without giving up the spread.

Symmetric benchmark (from the paper)
------------------------------------
Same spread, but centred on the mid. Same expected spread capture, no inventory
control, so it serves as the control for the skew.

Toxicity-aware wrapper
----------------------
Watches the mid-price `horizon` steps after each of its own fills (the
"markout"). In this model the informed trader's jump follows immediately, so a
1-step horizon is enough; in real markets desks measure markouts over seconds to
minutes. Uninformed fills have zero expected markout; informed fills lose
about jump_size. The running average markout per fill is therefore an estimate
of the adverse-selection cost per fill, and the wrapper widens both quotes by
that amount (the Glosten-Milgrom break-even logic). It learns the cost online
from its own fills; it is not told which traders are informed.
"""

from __future__ import annotations

import numpy as np

from .market import MarketParams


class Strategy:
    name = "strategy"

    def reset(self, n_sims: int, params: MarketParams) -> None:
        self.params = params

    def quotes(self, t: float, s: np.ndarray, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

    def observe(self, step: int, n_bid: np.ndarray, n_ask: np.ndarray, s_before: np.ndarray, s_after: np.ndarray) -> None:
        """Called after every step with the fills and the mid-price before and after it."""


class AvellanedaStoikov(Strategy):
    def __init__(self, gamma: float, sigma: float | None = None, k: float | None = None):
        self.gamma, self._sigma, self._k = gamma, sigma, k
        self.name = f"Inventory (gamma={gamma:g})"

    def reset(self, n_sims, params):
        super().reset(n_sims, params)
        self.sigma = params.sigma if self._sigma is None else self._sigma
        self.k = params.k if self._k is None else self._k

    def spread(self, t: float) -> float:
        g, tau = self.gamma, self.params.T - t
        return g * self.sigma**2 * tau + (2.0 / g) * np.log1p(g / self.k)

    def reservation_offset(self, t: float, q: np.ndarray) -> np.ndarray:
        """r - s."""
        return -q * self.gamma * self.sigma**2 * (self.params.T - t)

    def quotes(self, t, s, q):
        half = 0.5 * self.spread(t)
        shift = self.reservation_offset(t, q)
        return half - shift, half + shift   # (delta_b, delta_a)


class Symmetric(AvellanedaStoikov):
    """Avellaneda-Stoikov spread, centred on the mid instead of the reservation price."""

    def __init__(self, gamma: float, sigma: float | None = None, k: float | None = None):
        super().__init__(gamma, sigma, k)
        self.name = f"Symmetric (gamma={gamma:g})"

    def quotes(self, t, s, q):
        half = np.full_like(s, 0.5 * self.spread(t))
        return half, half


class FixedSpread(Strategy):
    def __init__(self, half_spread: float):
        self.half = half_spread
        self.name = f"Fixed half-spread {half_spread:g}"

    def quotes(self, t, s, q):
        h = np.full_like(s, self.half)
        return h, h


class ExtraSpread(Strategy):
    """Base strategy with both quotes pushed out by a constant amount."""

    def __init__(self, base: Strategy, extra: float):
        self.base, self.extra = base, extra
        self.name = f"{base.name} + {extra:g}"

    def reset(self, n_sims, params):
        super().reset(n_sims, params)
        self.base.reset(n_sims, params)

    def quotes(self, t, s, q):
        db, da = self.base.quotes(t, s, q)
        return db + self.extra, da + self.extra


class ToxicityAware(Strategy):
    """Base strategy widened by an online estimate of adverse-selection cost per fill."""

    def __init__(self, base: Strategy, horizon: int = 1, learning_rate: float = 0.05, initial_cost: float = 0.0):
        self.base, self.horizon, self.lr, self.initial_cost = base, horizon, learning_rate, initial_cost
        self.name = f"{base.name} + toxicity filter"

    def reset(self, n_sims, params):
        super().reset(n_sims, params)
        self.base.reset(n_sims, params)
        self.cost = np.full(n_sims, self.initial_cost)   # estimated adverse move per fill
        self._pending = []                               # (n_bid, n_ask, mid at fill) per step
        self._extra_sum, self._n_quotes = np.zeros(n_sims), 0

    def quotes(self, t, s, q):
        db, da = self.base.quotes(t, s, q)
        extra = np.maximum(self.cost, 0.0)
        self._extra_sum += extra
        self._n_quotes += 1
        return db + extra, da + extra

    @property
    def average_extra(self) -> np.ndarray:
        """Average amount each quote was widened by, per simulation."""
        return self._extra_sum / max(self._n_quotes, 1)

    def observe(self, step, n_bid, n_ask, s_before, s_after):
        self._pending.append((n_bid, n_ask, s_before))
        if len(self._pending) < self.horizon:
            return
        b, a, s_fill = self._pending.pop(0)
        n = b + a
        has = n > 0
        if not has.any():
            return
        move = s_after - s_fill
        # Loss per fill: sold (ask filled) and price rose, or bought (bid filled) and price fell.
        markout = (a * move - b * move)[has] / n[has]
        self.cost[has] += self.lr * (markout - self.cost[has])
