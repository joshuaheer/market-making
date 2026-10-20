"""Tests for the market-making simulator and strategies.  Run with:  pytest"""

import numpy as np
import pytest

from mmlab import AvellanedaStoikov, ExtraSpread, FixedSpread, MarketParams, Symmetric, ToxicityAware, simulate

PARAMS = MarketParams()


# ------------------------------------------------------------------ formulas

def test_spread_formula_matches_paper_at_t0():
    s = AvellanedaStoikov(0.1)
    s.reset(1, PARAMS)
    expected = 0.1 * 4 * 1.0 + (2 / 0.1) * np.log(1 + 0.1 / 1.5)
    assert s.spread(0.0) == pytest.approx(expected)


def test_spread_tends_to_2_over_k_as_risk_aversion_vanishes():
    s = AvellanedaStoikov(1e-8)
    s.reset(1, PARAMS)
    assert s.spread(0.0) == pytest.approx(2 / PARAMS.k, rel=1e-6)


def test_reservation_price_below_mid_when_long_and_quotes_skew():
    s = AvellanedaStoikov(0.1)
    s.reset(3, PARAMS)
    q = np.array([-3.0, 0.0, 3.0])
    db, da = s.quotes(0.0, np.full(3, 100.0), q)
    assert s.reservation_offset(0.0, q)[2] < 0 < s.reservation_offset(0.0, q)[0]
    # Long: ask moves closer to the mid (sell more), bid moves further away (buy less).
    assert da[2] < da[1] < da[0] and db[2] > db[1] > db[0]
    np.testing.assert_allclose(da + db, s.spread(0.0))     # skew moves the quotes, not the spread


def test_symmetric_quotes_ignore_inventory():
    s = Symmetric(0.1)
    s.reset(2, PARAMS)
    db, da = s.quotes(0.0, np.full(2, 100.0), np.array([-5.0, 5.0]))
    np.testing.assert_allclose(db, da)
    np.testing.assert_allclose(db[0], db[1])


# ------------------------------------------------------------------ simulator mechanics

def test_pnl_decomposition_is_exact():
    p = PARAMS.with_(informed_rate=40, jump_size=2.0)
    res = simulate(p, AvellanedaStoikov(0.1), 500, seed=1)
    np.testing.assert_allclose(res.pnl, res.spread_capture + res.inventory_pnl_diffusion + res.inventory_pnl_jump,
                               atol=1e-9)


def test_fill_rate_matches_poisson_intensity():
    half = 1.0
    res = simulate(PARAMS, FixedSpread(half), 4000, seed=2)
    expected = 2 * PARAMS.A * np.exp(-PARAMS.k * half) * PARAMS.T     # both sides, whole session
    assert res.n_fills.mean() == pytest.approx(expected, rel=0.02)


def test_fixed_spread_profit_equals_spread_times_fills():
    # With zero-mean price moves, expected profit is just the spread earned on fills.
    res = simulate(PARAMS, FixedSpread(0.75), 4000, seed=3)
    expected = 0.75 * 2 * PARAMS.A * np.exp(-PARAMS.k * 0.75)
    se = res.pnl.std() / np.sqrt(len(res.pnl))
    assert abs(res.pnl.mean() - expected) < 4 * se + 0.5


def test_same_seed_is_reproducible():
    a = simulate(PARAMS, AvellanedaStoikov(0.1), 50, seed=9).pnl
    b = simulate(PARAMS, AvellanedaStoikov(0.1), 50, seed=9).pnl
    np.testing.assert_array_equal(a, b)


# ------------------------------------------------------------------ economics

def test_reproduces_avellaneda_stoikov_table_gamma_0_1():
    inv = simulate(PARAMS, AvellanedaStoikov(0.1), 5000, seed=4).summary()
    sym = simulate(PARAMS, Symmetric(0.1), 5000, seed=4).summary()
    assert inv["avg_spread"] == pytest.approx(1.49, abs=0.01)
    assert inv["mean_pnl"] == pytest.approx(65.0, abs=1.0)
    assert inv["std_pnl"] == pytest.approx(6.6, abs=0.6)
    assert inv["std_final_q"] == pytest.approx(2.9, abs=0.3)
    assert sym["mean_pnl"] == pytest.approx(68.4, abs=1.0)
    assert sym["std_final_q"] == pytest.approx(8.4, abs=0.6)


def test_inventory_strategy_has_higher_sharpe_than_symmetric():
    for g in (0.01, 0.1, 1.0):
        inv = simulate(PARAMS, AvellanedaStoikov(g), 2000, seed=5).summary()
        sym = simulate(PARAMS, Symmetric(g), 2000, seed=5).summary()
        assert inv["sharpe"] > sym["sharpe"]
        assert inv["std_final_q"] < sym["std_final_q"]


def test_informed_flow_destroys_unaware_quoter_and_filter_recovers():
    p = PARAMS.with_(informed_rate=100, jump_size=2.0)
    naive = simulate(p, AvellanedaStoikov(0.1), 2000, seed=6).summary()
    filt = simulate(p, ToxicityAware(AvellanedaStoikov(0.1)), 2000, seed=6).summary()
    assert naive["mean_pnl"] < 0 < filt["mean_pnl"]
    assert filt["informed_fills"] < naive["informed_fills"]


def test_informed_traders_stop_trading_when_spread_exceeds_their_edge():
    p = PARAMS.with_(informed_rate=100, jump_size=1.0)
    res = simulate(p, FixedSpread(1.01), 500, seed=7)
    assert res.n_informed_fills.sum() == 0


def test_toxicity_filter_learns_markout_close_to_informed_share_times_jump():
    # With a fixed base spread, the expected loss per fill is (informed share of fills) * jump.
    p = PARAMS.with_(informed_rate=50, jump_size=2.0)
    strat = ToxicityAware(FixedSpread(0.75), learning_rate=0.02)
    res = simulate(p, strat, 2000, seed=8)
    share = res.n_informed_fills.sum() / res.n_fills.sum()
    assert strat.cost.mean() == pytest.approx(share * 2.0, rel=0.25)


def test_widening_helps_under_heavy_informed_flow():
    p = PARAMS.with_(informed_rate=75, jump_size=2.0)
    base = simulate(p, AvellanedaStoikov(0.1), 2000, seed=9).pnl.mean()
    wide = simulate(p, ExtraSpread(AvellanedaStoikov(0.1), 1.0), 2000, seed=9).pnl.mean()
    assert wide > base + 20
