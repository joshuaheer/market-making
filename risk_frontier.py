"""
Experiment 2: the risk-return trade-off, and how the quotes respond to inventory.

Sweep risk aversion gamma from 0.001 to 3. Higher gamma means wider spreads (fewer,
safer fills) and stronger inventory skew. Plotting mean profit against its
standard deviation traces out a frontier: the trader picks a point on it.

Expected: the inventory strategy dominates the symmetric one at every gamma
(similar or slightly lower profit, much lower risk), so its Sharpe ratio is higher.
"""

import numpy as np
import matplotlib.pyplot as plt

from mmlab import AvellanedaStoikov, MarketParams, Symmetric, simulate
from experiments import _style as st

GAMMAS = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0]
LABELLED = {0.01: "0.01", 0.1: "0.1", 1.0: "1", 3.0: "3"}


def run(quick=False):
    st.apply()
    params = MarketParams()
    n = 2_000 if quick else 10_000
    rows = []
    for g in GAMMAS:
        inv = simulate(params, AvellanedaStoikov(g), n, seed=21).summary()
        sym = simulate(params, Symmetric(g), n, seed=21).summary()
        rows.append({"gamma": g, "inv": inv, "sym": sym})

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.4))
    for key, colour, marker, label in (("sym", st.ORANGE, "s", "Symmetric"), ("inv", st.BLUE, "o", "Inventory")):
        x = [r[key]["std_pnl"] for r in rows]
        y = [r[key]["mean_pnl"] for r in rows]
        ax1.plot(x, y, color=colour, marker=marker, label=label)
    for r in rows:
        if r["gamma"] in LABELLED:
            ax1.annotate(f"gamma {LABELLED[r['gamma']]}", (r["inv"]["std_pnl"], r["inv"]["mean_pnl"]),
                         textcoords="offset points", xytext=(8, -4), fontsize=8, color=st.INK_2)
    ax1.set(title="Risk vs return as risk aversion rises", xlabel="Std of session P&L ($)",
            ylabel="Mean session P&L ($)")
    ax1.set_ylim(0, 75)
    ax1.legend(loc="lower right")

    strat = AvellanedaStoikov(0.1)
    strat.reset(1, params)
    q = np.arange(-6, 7)
    db, da = strat.quotes(0.0, np.full(q.shape, params.s0), q.astype(float))
    ax2.axhline(0, color=st.INK_2, lw=0.8)
    ax2.plot(q, da, color=st.ORANGE, marker="s", label="Ask - mid")
    ax2.plot(q, -db, color=st.BLUE, marker="o", label="Bid - mid")
    ax2.plot(q, strat.reservation_offset(0.0, q), color=st.INK_2, ls="--", lw=1.2, label="Reservation price - mid")
    ax2.set(title="Quotes skew against inventory (gamma = 0.1, t = 0)", xlabel="Inventory held",
            ylabel="Distance from mid ($)")
    ax2.legend(loc="upper right")
    path = st.save(fig, "02_risk_frontier.png")
    return {"figure": path.name, "rows": rows}


if __name__ == "__main__":
    for r in run()["rows"]:
        print(r["gamma"], {k: round(r["inv"][k], 2) for k in ("mean_pnl", "std_pnl", "sharpe", "avg_spread")},
              {k: round(r["sym"][k], 2) for k in ("mean_pnl", "std_pnl", "sharpe")})
