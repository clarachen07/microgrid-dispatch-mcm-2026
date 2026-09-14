"""Physical safety, causal wiring, date events and single-settlement tests."""
from types import SimpleNamespace
from dataclasses import replace

import numpy as np
import pytest

from microgrid.data import DataBundle
from microgrid.export import events, period
from microgrid.simulate import Policy, execute_safe, settlement_components, simulate
from microgrid.quantile import JointForecaster, simulate_quantile_q2, solve_contract_48h
from microgrid.verify import verify_ledger


def test_safety_cancels_charging_when_realized_demand_is_high():
    c, d, u, waste, end = execute_safe(10, 30, 0, 15, 0, 6000)
    assert (c, d, u, waste, end) == pytest.approx((0, 0, 20, 0, 6000))


@pytest.mark.parametrize("soc,q,load,pv", [(1200, 0, 2000, 0), (10800, 2000, 0, 0), (6000, 100, 300, 20), (6000, 300, 100, 20)])
def test_safety_respects_power_state_balance_and_no_urgent_charging(soc, q, load, pv):
    c, d, u, w, end = execute_safe(q, load, pv, 3000, 3000, soc)
    assert q + pv + d + u == pytest.approx(load + c + w)
    assert 1200 - 1e-8 <= end <= 10800 + 1e-8
    assert c <= 5000 / 6 and d <= 5000 / 6
    assert c * d == 0 and c * u == 0
    assert end == pytest.approx(soc + .9 * c - d / .9)


def test_event_grouping_preserves_calendar_day_boundary():
    first, second = np.zeros(144), np.zeros(144)
    first[-2:] = [2, 3]
    second[:2] = [4, 5]
    assert events(first) == [(142, 144, 5.)]
    assert events(second) == [(0, 2, 9.)]
    assert period(142, 144) == "23:40-24:00"
    assert period(0, 2) == "00:00-00:20"
    assert events(np.zeros(144)) == []


def test_settlement_is_against_midnight_once_not_revision_chain():
    # An intermediate 140 kWh commitment is later replaced by 120 kWh.
    fees = settlement_components(100., 120., 2., 3., "adjusted")
    assert fees == pytest.approx((200., 60., 0., 30.))
    assert sum(fees) == pytest.approx(290.)
    adjusted = settlement_components(100., 60., 2., 0., "adjusted")
    assert adjusted == pytest.approx((200., 0., -40., 0.))


def test_current_actual_changes_safety_but_not_pre_interval_optimization(monkeypatch):
    import microgrid.simulate as module
    captured = []

    class FixedForecast:
        def __init__(self, data, family="naive", seed=2026):
            pass
        def forecast(self, origin, horizon, **kwargs):
            return SimpleNamespace(net_s=np.full((2, horizon), 100.), price_s=np.ones((2, horizon)), mean_load=np.full(horizon, 600.), mean_pv=np.zeros(horizon), mean_price=np.ones(horizon))
        def diagnostics(self):
            return {}

    def recording_solver(net_s, price_s, initial_soc, **kwargs):
        horizon = np.asarray(net_s).shape[1]
        captured.append((np.array(net_s, copy=True), np.array(price_s, copy=True), initial_soc))
        return {"q": np.full(horizon, 50.), "c": np.full(horizon, 10.), "d": np.zeros(horizon), "soc": np.full(horizon + 1, 6000.)}

    monkeypatch.setattr(module, "ForecastEngine", FixedForecast)
    monkeypatch.setattr(module, "solve_schedule", recording_solver)
    data = DataBundle(np.array([np.datetime64("2025-01-01")]), np.full((1, 144), 600.), np.zeros((1, 144)), np.ones((1, 144)), np.full(144, 600.), np.zeros(144), np.ones(144), np.zeros((1, 4, 24)))
    first, summary = simulate(Policy(), data=data, end_day=1, verbose=False)
    assert verify_ledger(first, summary, "q2", 144)["passed"]
    tampered = {key: value.copy() for key, value in first.items()}
    tampered["base_fee"][0] += 1.0
    result = verify_ledger(tampered, summary, "q2", 144)
    assert not result["passed"]
    assert result["max_fee_error_yuan"] == 1.0
    before = captured[:2]
    captured.clear()
    changed = replace(data, load=data.load.copy())
    changed.load[0, 0] = 1200.
    second, _ = simulate(Policy(), data=changed, end_day=1, verbose=False)
    for original, new in zip(before, captured[:2]):
        for a, b in zip(original, new):
            np.testing.assert_array_equal(a, b)
    assert first["plan"][0] == second["plan"][0]
    assert first["target_charge"][0] == second["target_charge"][0]
    assert first["emergency"][0] == 50.
    assert second["emergency"][0] == 150.
    assert first["charge"][0] == second["charge"][0] == 0.


def test_real_q43_pipeline_is_invariant_to_unavailable_future_inputs():
    """Use the real forecaster, LP solver, controller, and settlement together."""
    days = 3
    slot = np.arange(144)
    prior_load = 4400 + 700 * np.cos(2 * np.pi * slot / 144)
    prior_pv = 5800 * np.maximum(0., np.sin(np.pi * (slot - 36) / 72))
    prior_price = .75 + .3 * np.sin(2 * np.pi * slot / 144)
    load = np.vstack([prior_load * factor for factor in (1., 1.06, .91)])
    pv = np.vstack([prior_pv * factor for factor in (.93, 1.09, .79)])
    price = np.vstack([prior_price * factor for factor in (1.03, .96, 1.12)])
    issued = np.empty((days, 4, 24))
    for day in range(days):
        for release in range(4):
            targets = (release * 36 + 6 * np.arange(1, 25) - 1) % 144
            issued[day, release] = prior_pv[targets] * (1. + .02 * release)
    data = DataBundle(np.datetime64("2025-01-01") + np.arange(days).astype("timedelta64[D]"),
                      load, pv, price, prior_load, prior_pv, prior_price, issued)
    boundary = 144 + 60  # Day 2 at 10:00, between the 06:00 and 12:00 releases.
    changed = replace(data, load=data.load.copy(), pv=data.pv.copy(), price=data.price.copy(),
                      forecasts=data.forecasts.copy())
    changed.load.reshape(-1)[boundary:] *= 1.8
    changed.pv.reshape(-1)[boundary:] *= .15
    changed.price.reshape(-1)[boundary:] *= 2.5
    for day in range(days):
        for release in range(4):
            if day * 144 + release * 36 > boundary:
                changed.forecasts[day, release] *= .3
    policy = Policy(name="q43_causal_test", pv_forecasts=True, variable_price=True,
                    updates=(6, 12, 18), scenarios=12, family="naive")
    first, _ = simulate(policy, data=data, end_day=days, verbose=False)
    second, _ = simulate(policy, data=changed, end_day=days, verbose=False)
    # At the boundary itself the proposed controls are still fixed before the
    # newly changed interval is measured; the actual response may then differ.
    for key in ("plan", "effective", "target_charge", "target_discharge"):
        np.testing.assert_allclose(first[key][:boundary + 1], second[key][:boundary + 1],
                                   rtol=0, atol=1e-9, err_msg=key)
    for key in ("charge", "discharge", "soc_start", "soc_end", "cost", "base_fee",
                "up_fee", "down_fee", "emergency_fee"):
        np.testing.assert_allclose(first[key][:boundary], second[key][:boundary],
                                   rtol=0, atol=1e-9, err_msg=key)
    assert not np.allclose(first["cost"][boundary:], second["cost"][boundary:])


def quantile_test_data(days=10):
    slot = np.arange(144)
    prior_load = 4200 + 500 * np.cos(2 * np.pi * slot / 144)
    prior_pv = 5400 * np.maximum(0., np.sin(np.pi * (slot - 36) / 72))
    load = np.vstack([prior_load * (1 + .01 * day) for day in range(days)])
    pv = np.vstack([prior_pv * (1 - .005 * day) for day in range(days)])
    price = np.ones((days, 144))
    return DataBundle(np.datetime64("2025-01-01") + np.arange(days).astype("timedelta64[D]"),
                      load, pv, price, prior_load, prior_pv, np.ones(144),
                      np.zeros((days, 4, 24)))


def test_quantile_forecaster_never_uses_operating_day_actuals():
    data = quantile_test_data()
    first = JointForecaster(data.load / 6, data.pv / 6,
                            data.prior_load / 6, data.prior_pv / 6)
    changed_load = data.load.copy(); changed_pv = data.pv.copy()
    changed_load[8:] *= 3; changed_pv[8:] = 0
    second = JointForecaster(changed_load / 6, changed_pv / 6,
                             data.prior_load / 6, data.prior_pv / 6)
    for left, right in zip(first.pair(8), second.pair(8)):
        np.testing.assert_allclose(left, right)
    np.testing.assert_allclose(first.residual_margin(8, .8),
                               second.residual_margin(8, .8))


def test_quantile_contract_has_48h_terminal_state_and_limits():
    net = np.full(288, 500.)
    price = np.tile(np.r_[np.ones(144), np.ones(144) * 1.2], 1)
    grid, state = solve_contract_48h(net, price, 7200.)
    assert len(grid) == 288 and len(state) == 289
    assert state[0] == pytest.approx(7200.) and state[-1] == pytest.approx(6000.)
    assert np.min(grid) >= -1e-9
    assert np.min(state) >= 1200 - 1e-7 and np.max(state) <= 10800 + 1e-7


def test_quantile_immediate_feedback_is_physical_and_has_no_revisions():
    data = quantile_test_data(days=2)
    log, summary = simulate_quantile_q2(.8, data=data, end_day=2, verbose=False)
    result = verify_ledger(log, summary, "q2", 2 * 144)
    assert result["passed"]
    assert summary["configuration"]["feedback"] == "immediate_safe"
    assert summary["configuration"]["control_hours"] == 0
    np.testing.assert_array_equal(log["plan"], log["effective"])
    assert np.all(log["issue_index"] == np.repeat([0, 144], 144))


def test_quantile_no_storage_preserves_fixed_inventory():
    data = quantile_test_data(days=2)
    log, summary = simulate_quantile_q2(.8, data=data, end_day=2,
                                        storage=False, verbose=False)
    assert verify_ledger(log, summary, "q2", 2 * 144)["passed"]
    assert np.all(log["charge"] == 0) and np.all(log["discharge"] == 0)
    assert np.all(log["soc_start"] == 6000) and np.all(log["soc_end"] == 6000)
