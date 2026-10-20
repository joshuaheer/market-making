"""
Market model.

Mid-price
---------
Following Avellaneda & Stoikov (2008), the mid-price is an arithmetic random
walk: each step of length dt it moves up or down by sigma * sqrt(dt) with equal
probability. Optionally it also jumps by +/- jump_size when an informed trader
arrives (see below).

Uninformed order flow
---------------------
Liquidity traders arrive at random. If the market maker quotes its ask delta_a
above the mid, the chance of being lifted in a step of length dt is

    P(fill) = lambda(delta_a) * dt,   lambda(delta) = A * exp(-k * delta)

and likewise for the bid. Quoting tighter fills more often but earns less per
fill; that trade-off sets the optimal spread. A is the overall order-flow
intensity, k how quickly flow dies off as quotes move away from the mid.

Informed order flow (adverse selection)
---------------------------------------
With rate informed_rate per unit time, an informed trader arrives who knows the
mid-price is about to jump by +/- jump_size. They buy at the ask if the price is
about to rise above it (delta_a < jump_size), or sell at the bid if it is about
to fall below it. Either way the jump then happens. A market maker hit by an
informed trader loses jump_size - delta on that fill: this is adverse selection,
the reason real market makers widen spreads when flow looks "toxic".

Units follow the paper: T = 1 trading session, prices in dollars.
"""

from __future__ import annotations

from dataclasses import dataclass, replace


@dataclass(frozen=True)
class MarketParams:
    s0: float = 100.0          # starting mid-price
    T: float = 1.0             # session length
    sigma: float = 2.0         # mid-price volatility (dollars per sqrt(unit time))
    dt: float = 0.005          # time step
    A: float = 140.0           # order-arrival intensity at zero distance from the mid
    k: float = 1.5             # decay of arrival intensity with distance
    informed_rate: float = 0.0  # informed-trader arrivals per unit time
    jump_size: float = 0.0     # size of the price move informed traders know about

    @property
    def n_steps(self) -> int:
        return int(round(self.T / self.dt))

    @property
    def total_variance_rate(self) -> float:
        """Variance per unit time from diffusion plus jumps: sigma^2 + rate * jump^2."""
        return self.sigma**2 + self.informed_rate * self.jump_size**2

    def with_(self, **changes) -> "MarketParams":
        return replace(self, **changes)
