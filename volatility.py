"""
Experiment 4: "What happens if volatility doubles?" (and what if you get it wrong?)

Part A: the true vol is 1, 2, 4 or 8, and the market maker knows it. The
inventory term of the spread, gamma * sigma^2 * (T - t), scales with variance, so
quotes widen and fills fall as vol rises.

Part B: the true vol is 4, but the market maker's model uses a different number.
Underestimating vol under-prices inventory risk: quotes are too tight and too
weakly skewed, so the position wanders and P&L becomes far more volatile.
Overestimating it wastes edge by quoting too wide.
"""

import numpy as np
import matplotlib.pyplot as plt

from mmlab import AvellanedaStoikov, MarketParams, simulate
from experiments import _style as st

GAMMA = 0.1
TRUE_VOLS = [1, 2, 4, 8]
ASSUMED = [1, 2, 3, 4, 6, 8]
TRUE_FOR_B = 4


def run(quick=False):
    st.apply()
    n = 2_000 if quick else 10_000
    part_a = []
    for sigma in TRUE_VOLS:
        s = simulate(MarketParams(sigma=sigma), AvellanedaStoikov(GAMMA), n, seed=40 + sigma).summary()
        part_a.append({"sigma": sigma, **s})
    part_b = []
    for assumed in ASSUMED:
        s = simulate(MarketParams(sigma=TRUE_FOR_B), AvellanedaStoikov(GAMMA, sigma=assumed), n, seed=77).summary()
        part_b.append({"assumed": assumed, **s})

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.4))
    ax1.plot(TRUE_VOLS, [r["mean_pnl"] for r in part_a], color=st.BLUE, marker="o", label="Mean P&L")
    ax1.plot(TRUE_VOLS, [r["std_pnl"] for r in part_a], color=st.ORANGE, marker="s", label="Std of P&L")
    for r in part_a:
        ax1.annotate(f"spread {r['avg_spread']:.1f}", (r["sigma"], r["mean_pnl"]), textcoords="offset points",
                     xytext=(6, 6), fontsize=8, color=st.INK_2)
    ax1.set(title="Higher vol: wider quotes, less profit", xlabel="True mid-price volatility (sigma)",
            ylabel="$ per session")
    ax1.set_xticks(TRUE_VOLS)
    ax1.set_ylim(0, 80)
    ax1.legend()

    ax2.plot(ASSUMED, [r["mean_pnl"] for r in part_b], color=st.BLUE, marker="o", label="Mean P&L")
    ax2.plot(ASSUMED, [r["std_pnl"] for r in part_b], color=st.ORANGE, marker="s", label="Std of P&L")
    ax2.axvline(TRUE_FOR_B, color=st.INK_2, lw=0.8, ls=":")
    ax2.set(title=f"Using the wrong vol (true sigma = {TRUE_FOR_B})", xlabel="Sigma assumed by the quoting model",
            ylabel="$ per session")
    ax2.set_ylim(0, 80)
    ax2.legend()
    path = st.save(fig, "04_volatility.png")
    return {"figure": path.name, "part_a": part_a, "part_b": part_b, "true_for_b": TRUE_FOR_B}


if __name__ == "__main__":
    out = run()
    for r in out["part_a"]:
        print("true", r["sigma"], {k: round(r[k], 2) for k in ("avg_spread", "mean_pnl", "std_pnl", "sharpe", "fills", "std_final_q")})
    for r in out["part_b"]:
        print("assumed", r["assumed"], {k: round(r[k], 2) for k in ("avg_spread", "mean_pnl", "std_pnl", "sharpe", "std_final_q")})
