# Market-Making Simulator: Inventory Risk and Adverse Selection

A Python simulator of a market maker who continuously quotes a bid and an ask, earns the spread, and has to manage two risks: **holding too much inventory** when the price moves, and **trading with people who know more than you do**.

It implements the Avellaneda-Stoikov (2008) optimal quoting model, reproduces the paper's published results, and then extends the model with informed traders and a quoter that detects toxic order flow from its own fills.

![Adverse selection results](figures/03_adverse_selection.png)

## Headline findings

| Question | Result |
|---|---|
| Does the implementation match the paper? | Reproduces Avellaneda & Stoikov's Table 1: 28 of the 30 published figures match within two standard errors, e.g. at γ = 0.1, profit **64.9 (paper 65.0)**, std **6.5 (6.6)**, inventory std **2.9 (2.9)**. |
| What does skewing quotes against inventory buy you? | At γ = 0.1 it halves P&L risk (std $13.3 → $6.5) and cuts inventory std from 8.3 to 2.9, for a 5% drop in mean profit, **doubling the Sharpe ratio (5.1 → 10.0)**. The inventory strategy has the higher Sharpe at every risk-aversion level tested. |
| What do informed traders do to a market maker? | With 100 informed traders per session, the unaware quoter goes from **+$65 to −$65** per session. The P&L decomposition shows why: spread capture is unchanged, but jump losses on inventory wipe it out. |
| Can you defend against them without knowing who they are? | A toxicity filter that widens quotes by the average markout of its own fills turns the −$65 into **+$13**. The best fixed widening, chosen with hindsight, makes +$24. The gap is explained below. |
| What if volatility doubles? | From σ = 2 to 4, the optimal spread widens from 1.49 to 2.09, fills fall 21% and profit falls 19%, while risk stays roughly constant: the model absorbs vol by widening, not by taking more risk. |

All numbers are in [RESULTS.md](RESULTS.md).

## What's inside

```
mmlab/
  market.py       market parameters: mid-price volatility, order-flow intensity, informed traders
  strategies.py   Avellaneda-Stoikov, symmetric benchmark, fixed spread, toxicity-aware wrapper
  simulator.py    vectorised Monte Carlo of 10,000 sessions at once, with exact P&L decomposition
experiments/      four scripts, one per question; each saves a figure in figures/
tests/            14 tests against the paper, the formulas and economic intuition
```

## Running it

```bash
pip install -r requirements.txt
python -m experiments.run_all      # all experiments, ~15 seconds; writes RESULTS.md
pytest                             # run the tests
```

## The model

**Mid-price.** A random walk: each step of dt = 0.005 it moves ±σ√dt, with σ = 2 over a session of length T = 1.

**Uninformed order flow.** If you quote your ask δ above the mid, the chance of being filled in a step is A·e^(−kδ)·dt. Quoting tighter fills more often but earns less per fill.

**The Avellaneda-Stoikov quoter.** A market maker with risk aversion γ, holding q shares, values the stock at its *reservation price*:

  r = s − q·γ·σ²·(T − t)

When long, an extra share is worth less to you than the market mid, so r sits below it. The optimal total spread is

  δ_ask + δ_bid = γσ²(T − t) + (2/γ)·ln(1 + γ/k)

and both quotes are centred on r instead of the mid. The first term is an inventory-risk premium that disappears at the close; the second is pure order-flow economics (it tends to 2/k as γ → 0). Centring on r **skews** the quotes: when long, both move down, so you sell more and buy less, pulling inventory back to zero without giving up the spread.

**P&L accounting.** Every session's P&L splits exactly into:
- **Spread capture**: each fill earns its distance from the mid.
- **Inventory P&L**: q × (change in mid) on whatever position you hold, split into the diffusion part and the jump part.

This identity holds to floating-point precision on every path, and a test checks it.

## The experiments

### 1. Reproducing the paper
![](figures/01_replication.png)

10,000 sessions per row versus the paper's 1,000. Cells show **ours (paper)**:

| γ | Strategy | Spread | Profit | Std(profit) | Std(final q) |
|---:|---|---|---|---|---|
| 0.1 | Inventory | 1.49 (1.49) | 64.9 (65.0) | 6.5 (6.6) | 2.9 (2.9) |
| 0.1 | Symmetric | 1.49 (1.49) | 68.0 (68.4) | 13.3 (12.7) | 8.3 (8.4) |
| 0.01 | Inventory | 1.35 (1.35) | 68.5 (68.6) | 8.9 (8.7) | 5.2 (5.1) |
| 0.01 | Symmetric | 1.35 (1.35) | 68.8 (68.8) | 13.4 (12.8) | 8.6 (8.7) |
| 1 | Inventory | 3.03 (3.02) | 31.4 (31.4) | 4.9 (5.0) | 1.6 (1.7) |
| 1 | Symmetric | 3.03 (3.02) | 44.1 (44.0) | 9.7 (11.0) | 5.8 (5.1) |

28 of the 30 figures match within two standard errors, allowing for sampling noise in both the paper's 1,000 sessions and ours. The two exceptions are both in the γ = 1 symmetric row: std(profit) and std(final q). The most likely cause is a detail the paper doesn't specify, such as the order in which fills and price moves are applied within a step; the inventory strategy, which is the paper's subject, matches throughout.

### 2. The risk-return frontier
![](figures/02_risk_frontier.png)

Raising γ trades profit for safety along a frontier. The inventory strategy sits up and to the left of the symmetric one at every γ: similar or slightly lower profit for much less risk. Sharpe peaks around γ = 0.1 (10.0). The right panel shows the mechanism: holding 4 shares at t = 0 moves both quotes down by $1.60, so the ask sits $0.76 *below* the mid and the bid $2.45 below it. The market maker is willing to sell below fair value to cut its risk.

### 3. Adverse selection
Informed traders arrive at random, each knowing the mid is about to jump by $2. They trade only when it pays them, so every fill with them loses money.

The **toxicity filter** never learns who is informed. It watches the **markout** of its own fills (how far the price moves against it straight after each trade) and widens both quotes by the running average. This is Glosten-Milgrom break-even logic, and the same idea trading desks use with real markout data.

| Informed per session | Unaware | With filter | Best fixed widening (hindsight) |
|---:|---:|---:|---:|
| 0 | +$64.7 | +$64.7 | +$64.7 |
| 20 | +$38.9 | +$38.5 | +$40.6 |
| 50 | −$0.5 | +$20.1 | +$25.4 |
| 100 | −$64.8 | +$13.3 | +$24.1 |

**Why the filter falls short of the best fixed widening.** The filter widens to the level where the *average* markout is covered, which is break-even against informed flow. But informed traders stop trading entirely once your half-spread exceeds their $2 edge, so profit jumps discontinuously near that point (middle panel). An average-markout rule doesn't see that cliff. A model-based quoter that estimated the informed edge and compared "absorb it" with "quote outside it" would capture the rest; that is the natural next step.

### 4. Volatility
![](figures/04_volatility.png)

**If vol doubles** (σ 2 → 4): spread widens 1.49 → 2.09, fills fall 97 → 77, profit falls $64.9 → $52.7, and risk barely moves ($6.6 → $6.8). The inventory term γσ²(T − t) scales with *variance*, so the response to vol is non-linear: at σ = 8, profit drops to $21.

**If you mis-estimate vol** (true σ = 4):
- **Underestimating** (assume σ = 1) under-prices inventory risk. Quotes are too tight and too weakly skewed, so inventory wanders (std 4.1 vs 2.0) and P&L risk rises 66% ($11.3 vs $6.8).
- **Overestimating** quotes too wide and gives up edge.

The Sharpe ratio actually peaks at an assumed σ of 3 rather than the true 4. That isn't a bug: Avellaneda-Stoikov maximises exponential utility at a given risk aversion, not the Sharpe ratio.

## What the tests check

- Spread and reservation-price formulas match the paper; the spread tends to 2/k as γ → 0.
- Skewing moves the quotes without changing the total spread; the symmetric strategy ignores inventory.
- Simulated fill rate matches the Poisson intensity A·e^(−kδ); fixed-spread profit equals spread × expected fills.
- The P&L decomposition is exact on every path.
- The paper's γ = 0.1 row is reproduced; the inventory strategy beats symmetric on Sharpe at every γ.
- Informed traders stop trading when the spread exceeds their edge.
- The filter's learned cost converges to (informed share of fills) × jump size, and the filter turns a losing quoter profitable.

## Limitations and next steps

- One unit per fill and no queue position. A full limit order book with competing market makers would make fills depend on where you sit in the queue, not just the distance from the mid.
- The informed traders' edge is a fixed jump that arrives instantly. Real information leaks out gradually, so markouts are measured over seconds to minutes.
- **Model-based toxicity response:** estimate the informed edge from markouts and choose between absorbing it and quoting outside it.
- **Guéant-Lehalle-Fernandez-Tapia (2013):** a closed-form infinite-horizon version with hard inventory limits, closer to how desks quote continuously.
- **Calibration:** fit A and k to real trade data, e.g. from a crypto exchange's public trade feed.

## References

- Avellaneda, M. & Stoikov, S. (2008). High-frequency trading in a limit order book. *Quantitative Finance*, 8(3).
- Glosten, L. & Milgrom, P. (1985). Bid, ask and transaction prices in a specialist market with heterogeneously informed traders. *Journal of Financial Economics*.
- Guéant, O., Lehalle, C.-A. & Fernandez-Tapia, J. (2013). Dealing with the inventory risk: a solution to the market making problem. *Mathematics and Financial Economics*.
