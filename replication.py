"""
Experiment 1: reproduce Table 1 of Avellaneda & Stoikov (2008).

Set-up from the paper: s0 = 100, T = 1, sigma = 2, dt = 0.005, A = 140, k = 1.5,
for risk aversion gamma = 0.1, 0.01 and 1. Two strategies with the same spread:
"inventory" (quotes centred on the reservation price) and "symmetric" (centred on
the mid). The paper ran 1,000 sessions; we run 10,000 to tighten our estimates.

Expected: the inventory strategy gives up a little mean profit for a much smaller
spread of outcomes, and ends the day with far less inventory.
"""

import numpy as np
import matplotlib.pyplot as plt

from mmlab import AvellanedaStoikov, MarketParams, Symmetric, simulate
from experiments import _style as st

# Published results: (avg spread, profit, std profit, final q, std final q).
PAPER = {
    (0.1, "Inventory"): (1.49, 65.0, 6.6, 0.08, 2.9), (0.1, "Symmetric"): (1.49, 68.4, 12.7, 0.26, 8.4),
    (0.01, "Inventory"): (1.35, 68.6, 8.7, 0.12, 5.1), (0.01, "Symmetric"): (1.35, 68.8, 12.8, 0.09, 8.7),
    (1.0, "Inventory"): (3.02, 31.4, 5.0, 0.02, 1.7), (1.0, "Symmetric"): (3.02, 44.0, 11.0, 0.00, 5.1),
}


def run(quick=False):
    st.apply()
    params = MarketParams()
    n = 2_000 if quick else 10_000
    rows = []
    results = {}
    for gamma in (0.1, 0.01, 1.0):
        for label, cls in (("Inventory", AvellanedaStoikov), ("Symmetric", Symmetric)):
            res = simulate(params, cls(gamma), n, seed=int(gamma * 1000) + len(label))
            results[gamma, label] = res
            s = res.summary()
            rows.append({"gamma": gamma, "strategy": label, "ours": (s["avg_spread"], s["mean_pnl"], s["std_pnl"],
                         s["mean_final_q"], s["std_final_q"]), "paper": PAPER[gamma, label],
                         "se_mean": s["std_pnl"] / np.sqrt(n), "sharpe": s["sharpe"]})

    # One illustrative session, same random numbers for both strategies.
    inv = simulate(params, AvellanedaStoikov(0.1), 1, seed=11, keep_paths=1)
    sym = simulate(params, Symmetric(0.1), 1, seed=11, keep_paths=1)
    t = inv.paths["t"]

    fig = plt.figure(figsize=(14, 4.6))
    grid = fig.add_gridspec(2, 2, width_ratios=[1.5, 1], height_ratios=[1.6, 1], hspace=0.35, wspace=0.18)
    ax_path = fig.add_subplot(grid[0, 0])
    ax_q = fig.add_subplot(grid[1, 0], sharex=ax_path)
    ax_hist = fig.add_subplot(grid[:, 1])

    ax_path.plot(t, inv.paths["mid"][0], color=st.INK_2, lw=1.2, label="Mid-price")
    ax_path.plot(t, inv.paths["ask"][0], color=st.ORANGE, lw=1, label="Ask (inventory strategy)")
    ax_path.plot(t, inv.paths["bid"][0], color=st.BLUE, lw=1, label="Bid (inventory strategy)")
    ax_path.set(title="One session at gamma = 0.1: quotes and inventory", ylabel="Price ($)")
    lo, hi = ax_path.get_ylim()
    ax_path.set_ylim(lo, hi + 0.35 * (hi - lo))
    ax_path.legend(ncol=3, loc="upper left")
    plt.setp(ax_path.get_xticklabels(), visible=False)

    ax_q.step(t, sym.paths["q"][0], color=st.ORANGE, lw=1.5, where="post", label="Symmetric")
    ax_q.step(t, inv.paths["q"][0], color=st.BLUE, lw=1.5, where="post", label="Inventory")
    ax_q.axhline(0, color=st.INK_2, lw=0.8)
    ax_q.set(xlabel="Time (fraction of session)", ylabel="Inventory")
    ax_q.legend(ncol=2, loc="lower left")

    bins = np.linspace(20, 110, 61)
    ax_hist.hist(results[0.1, "Symmetric"].pnl, bins=bins, color=st.ORANGE, alpha=0.75, label="Symmetric")
    ax_hist.hist(results[0.1, "Inventory"].pnl, bins=bins, color=st.BLUE, alpha=0.75, label="Inventory")
    ax_hist.set(title="Session P&L, gamma = 0.1", xlabel="Profit per session ($)", ylabel="Sessions")
    ax_hist.legend()
    path = st.save(fig, "01_replication.png")
    return {"figure": path.name, "rows": rows, "n_sims": n}


def table_markdown(rows):
    head = ("| gamma | Strategy | Spread | Profit | Std(profit) | Final q | Std(final q) | Sharpe |\n"
            "|---:|---|---|---|---|---|---|---:|")
    lines = [head]
    for r in rows:
        o, p = r["ours"], r["paper"]
        cells = [f"{o[0]:.2f} ({p[0]:.2f})", f"{o[1]:.1f} ({p[1]:.1f})", f"{o[2]:.1f} ({p[2]:.1f})",
                 f"{o[3]:+.2f} ({p[3]:+.2f})", f"{o[4]:.1f} ({p[4]:.1f})"]
        lines.append(f"| {r['gamma']:g} | {r['strategy']} | " + " | ".join(cells) + f" | {r['sharpe']:.1f} |")
    return "\n".join(lines)


if __name__ == "__main__":
    out = run()
    print("ours (paper)")
    print(table_markdown(out["rows"]))
