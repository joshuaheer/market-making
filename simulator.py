"""
Monte Carlo simulator for a single market maker, vectorised across simulations.

Each step:
1. The strategy posts bid = s - delta_b and ask = s + delta_a.
2. Uninformed buyers lift the ask with probability A exp(-k delta_a) dt; sellers
   hit the bid with probability A exp(-k delta_b) dt.
3. With probability informed_rate * dt an informed trader arrives knowing the
   next price jump, and trades if it is profitable for them.
4. The mid moves by +/- sigma sqrt(dt), plus the jump if one was due.

P&L accounting (exact, not approximate)
---------------------------------------
Wealth = cash + q * s. Its change splits cleanly into two sources:

* Spread capture: every fill earns its distance from the mid (delta_a on a
  sale, delta_b on a purchase). This is the market maker's "edge".
* Inventory P&L: q * (change in mid) on whatever position is held. This is the
  risk, split further into the diffusion part and the jump part. With informed
  traders, the jump part is mostly adverse selection: the jumps arrive right
  after an informed trader has dumped the wrong side of the trade on you.

    total P&L = spread capture + inventory P&L (diffusion) + inventory P&L (jumps)

holds to floating-point precision on every path; a test checks it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .market import MarketParams
from .strategies import Strategy


@dataclass
class SimResult:
    strategy: str
    pnl: np.ndarray
    final_q: np.ndarray
    spread_capture: np.ndarray
    inventory_pnl_diffusion: np.ndarray
    inventory_pnl_jump: np.ndarray
    n_fills: np.ndarray
    n_informed_fills: np.ndarray
    avg_spread: float
    max_abs_q: np.ndarray
    paths: dict = field(default_factory=dict)

    @property
    def inventory_pnl(self) -> np.ndarray:
        return self.inventory_pnl_diffusion + self.inventory_pnl_jump

    def summary(self) -> dict:
        std = self.pnl.std(ddof=1)
        return {
            "strategy": self.strategy,
            "avg_spread": self.avg_spread,
            "mean_pnl": float(self.pnl.mean()),
            "std_pnl": float(std),
            "sharpe": float(self.pnl.mean() / std) if std > 0 else float("nan"),
            "mean_final_q": float(self.final_q.mean()),
            "std_final_q": float(self.final_q.std(ddof=1)),
            "mean_max_abs_q": float(self.max_abs_q.mean()),
            "fills": float(self.n_fills.mean()),
            "informed_fills": float(self.n_informed_fills.mean()),
            "spread_capture": float(self.spread_capture.mean()),
            "inventory_pnl_diffusion": float(self.inventory_pnl_diffusion.mean()),
            "inventory_pnl_jump": float(self.inventory_pnl_jump.mean()),
        }


def simulate(params: MarketParams, strategy: Strategy, n_sims: int = 1000, seed=None,
             keep_paths: int = 0) -> SimResult:
    """Run n_sims independent sessions. keep_paths > 0 stores mid, quotes and inventory for that many sessions."""
    rng = np.random.default_rng(seed)
    p = params
    n_steps = p.n_steps
    strategy.reset(n_sims, p)

    s = np.full(n_sims, p.s0)
    q = np.zeros(n_sims)
    cash = np.zeros(n_sims)
    spread_capture = np.zeros(n_sims)
    inv_diff = np.zeros(n_sims)
    inv_jump = np.zeros(n_sims)
    n_fills = np.zeros(n_sims)
    n_informed = np.zeros(n_sims)
    max_abs_q = np.zeros(n_sims)
    spread_sum = 0.0
    step_move = p.sigma * np.sqrt(p.dt)

    if keep_paths:
        rec = {key: np.empty((keep_paths, n_steps + 1)) for key in ("mid", "bid", "ask", "q")}

    for i in range(n_steps):
        t = i * p.dt
        delta_b, delta_a = strategy.quotes(t, s, q)
        spread_sum += float(np.mean(delta_a + delta_b))

        prob_a = np.clip(p.A * np.exp(-p.k * delta_a) * p.dt, 0.0, 1.0)
        prob_b = np.clip(p.A * np.exp(-p.k * delta_b) * p.dt, 0.0, 1.0)
        n_ask = (rng.random(n_sims) < prob_a).astype(float)
        n_bid = (rng.random(n_sims) < prob_b).astype(float)

        jump = np.zeros(n_sims)
        if p.informed_rate > 0:
            arrives = rng.random(n_sims) < p.informed_rate * p.dt
            direction = np.where(rng.random(n_sims) < 0.5, 1.0, -1.0)
            jump = arrives * direction * p.jump_size
            inf_ask = (arrives & (direction > 0) & (delta_a < p.jump_size)).astype(float)
            inf_bid = (arrives & (direction < 0) & (delta_b < p.jump_size)).astype(float)
            n_ask += inf_ask
            n_bid += inf_bid
            n_informed += inf_ask + inf_bid

        if keep_paths:
            rec["mid"][:, i], rec["q"][:, i] = s[:keep_paths], q[:keep_paths]
            rec["bid"][:, i], rec["ask"][:, i] = (s - delta_b)[:keep_paths], (s + delta_a)[:keep_paths]

        # Trades happen at the posted quotes.
        cash += n_ask * (s + delta_a) - n_bid * (s - delta_b)
        q += n_bid - n_ask
        spread_capture += n_ask * delta_a + n_bid * delta_b
        n_fills += n_ask + n_bid
        np.maximum(max_abs_q, np.abs(q), out=max_abs_q)

        # Mid-price moves; the held inventory gains or loses.
        diffusion = np.where(rng.random(n_sims) < 0.5, step_move, -step_move)
        inv_diff += q * diffusion
        inv_jump += q * jump
        s_before = s
        s = s + diffusion + jump
        strategy.observe(i, n_bid, n_ask, s_before, s)

    if keep_paths:
        rec["mid"][:, -1], rec["q"][:, -1] = s[:keep_paths], q[:keep_paths]
        rec["bid"][:, -1] = rec["ask"][:, -1] = np.nan
        rec["t"] = np.arange(n_steps + 1) * p.dt

    pnl = cash + q * s
    return SimResult(strategy.name, pnl, q.copy(), spread_capture, inv_diff, inv_jump, n_fills, n_informed,
                     spread_sum / n_steps, max_abs_q, rec if keep_paths else {})
