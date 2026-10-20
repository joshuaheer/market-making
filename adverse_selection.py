"""
Experiment 3: adverse selection. What happens when some traders know more than you?

Informed traders arrive at a given rate per session, each knowing the mid is about
to jump by $2. They trade only when it pays them, which means you lose on every
fill with them. Three ways to quote:

* Avellaneda-Stoikov as before (gamma = 0.1), unaware of the informed flow.
* The same, plus a toxicity filter: it measures the markout of its own fills
  (how far the price moves against it right after) and widens both quotes by the
  running average. It is never told which traders were informed.
* A best fixed widening, found by trying every widening from 0 to $2.50 with
  hindsight. This is an upper benchmark that a real trader could not know in advance.

Expected: the unaware quoter's profit collapses as informed flow grows; the filter
recovers most of it. The filter widens to the *break-even* level (where informed
losses roughly equal what the widening costs in lost spread), which is not quite
the profit-maximising level: informed traders stop trading altogether once the
spread exceeds their edge, a discontinuity that an average-markout rule doesn't see.
"""

import numpy as np
import matplotlib.pyplot as plt

from mmlab import AvellanedaStoikov, ExtraSpread, MarketParams, ToxicityAware, simulate
from experiments import _style as st

JUMP = 2.0
RATES = [0, 10, 20, 35, 50, 75, 100]
EXTRAS = np.round(np.arange(0.0, 2.51, 0.125), 3)
FOCUS_RATE = 50
GAMMA = 0.1


def run(quick=False):
    st.apply()
    n = 1_000 if quick else 5_000
    rows = []
    sweeps = {}
    for rate in RATES:
        params = MarketParams(informed_rate=rate, jump_size=JUMP)
        seed = 300 + rate
        naive = simulate(params, AvellanedaStoikov(GAMMA), n, seed=seed)
        filt_strategy = ToxicityAware(AvellanedaStoikov(GAMMA))
        filt = simulate(params, filt_strategy, n, seed=seed)
        sweep = [(e, simulate(params, ExtraSpread(AvellanedaStoikov(GAMMA), e), n, seed=seed)) for e in EXTRAS]
        sweeps[rate] = sweep
        best_extra, best = max(sweep, key=lambda item: item[1].pnl.mean())
        rows.append({
            "rate": rate, "naive": naive.summary(), "filter": filt.summary(), "best": best.summary(),
            "best_extra": float(best_extra), "filter_avg_extra": float(filt_strategy.average_extra.mean()),
            "naive_res": naive, "filter_res": filt,
        })

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 4.6))
    x = RATES
    for key, colour, marker, label in (("naive", st.ORANGE, "s", "Unaware (A-S)"),
                                       ("filter", st.BLUE, "o", "A-S + toxicity filter"),
                                       ("best", st.AQUA, "^", "Best fixed widening (hindsight)")):
        ax1.plot(x, [r[key]["mean_pnl"] for r in rows], color=colour, marker=marker, label=label)
    ax1.axhline(0, color=st.INK_2, lw=0.8)
    ax1.set(title="Profit as informed flow grows", xlabel=f"Informed traders per session (jump ${JUMP:g})",
            ylabel="Mean session P&L ($)")
    ax1.legend()

    focus = next(r for r in rows if r["rate"] == FOCUS_RATE)
    sweep = sweeps[FOCUS_RATE]
    ax2.plot([e for e, _ in sweep], [res.pnl.mean() for _, res in sweep], color=st.INK_2, marker="o", ms=4,
             label="Fixed widening")
    ax2.axvline(focus["filter_avg_extra"], color=st.BLUE, lw=1.5, ls="--",
                label=f"Filter's average widening (${focus['filter_avg_extra']:.2f})")
    ax2.axvline(focus["best_extra"], color=st.AQUA, lw=1.5, ls=":",
                label=f"Best widening (${focus['best_extra']:.2f})")
    ax2.set(title=f"How wide to quote? ({FOCUS_RATE} informed)",
            xlabel="Extra half-spread on both quotes ($)", ylabel="Mean session P&L ($)")
    ax2.legend(loc="lower right")

    labels = ["Spread\ncapture", "Diffusion\nP&L", "Jump P&L\n(adverse sel.)", "Total"]
    width = 0.38
    pos = np.arange(len(labels))
    for offset, key, colour, label in ((-width / 2, "naive", st.ORANGE, "Unaware"), (width / 2, "filter", st.BLUE, "With filter")):
        s = focus[key]
        values = [s["spread_capture"], s["inventory_pnl_diffusion"], s["inventory_pnl_jump"], s["mean_pnl"]]
        bars = ax3.bar(pos + offset, values, width=width * 0.92, color=colour, label=label)
        for bar, v in zip(bars, values):
            ax3.annotate("0" if abs(v) < 0.5 else f"{v:+.0f}", (bar.get_x() + bar.get_width() / 2, v), textcoords="offset points",
                         xytext=(0, 4 if v >= 0 else -12), ha="center", fontsize=8, color=st.INK)
    ax3.axhline(0, color=st.INK_2, lw=0.8)
    ax3.set_xticks(pos, labels)
    ax3.grid(axis="x", visible=False)
    ax3.set(title=f"Where the money goes ({FOCUS_RATE} informed)", ylabel="Mean $ per session")
    ax3.legend()
    path = st.save(fig, "03_adverse_selection.png")

    for r in rows:
        r.pop("naive_res"), r.pop("filter_res")
    return {"figure": path.name, "rows": rows, "jump": JUMP, "focus_rate": FOCUS_RATE}


if __name__ == "__main__":
    out = run()
    for r in out["rows"]:
        print(r["rate"], round(r["naive"]["mean_pnl"], 1), round(r["filter"]["mean_pnl"], 1),
              round(r["best"]["mean_pnl"], 1), "best extra", r["best_extra"], "filter extra", round(r["filter_avg_extra"], 2),
              "informed fills", round(r["naive"]["informed_fills"], 1), round(r["filter"]["informed_fills"], 1))
