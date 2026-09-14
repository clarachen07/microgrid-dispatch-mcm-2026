"""Small independently checkable scheduling and settlement examples."""

from types import SimpleNamespace

import numpy as np
import pytest

from microgrid.optimize import (
    OptimizationError,
    adjustment_cost,
    recover_exclusive_dispatch,
    solve_schedule,
)


def test_newsvendor_eighty_percent_quantile():
    net = np.arange(10.0)[:, None]
    answer = solve_schedule(net, np.ones_like(net), 6000, capacity=False)
    assert 7.0 - 1e-7 <= answer["q"][0] <= 8.0 + 1e-7
    assert answer["objective"] == pytest.approx(8.5)


@pytest.mark.parametrize("settlement,expected", [
    ("adjusted", [160.0, 200.0, 320.0]),
    ("full_plan_plus_penalty", [240.0, 200.0, 320.0]),
])
def test_three_settlement_cases(settlement, expected):
    assert adjustment_cost(np.array([60.0, 100.0, 140.0]), 100.0, 2.0, settlement) == pytest.approx(expected)


def test_adjusted_contract_examples_at_one_yuan():
    assert adjustment_cost(80.0, 100.0, 1.0, "adjusted") == pytest.approx(90.0)
    assert adjustment_cost(120.0, 100.0, 1.0, "adjusted") == pytest.approx(130.0)


def test_full_plan_plus_penalty_downward_revision_is_dominated():
    answer = solve_schedule([[60.0]], [[2.0]], 6000, purchase=[140.0], original=[100.0], adjust_mask=[True], capacity=False, settlement="full_plan_plus_penalty")
    # A previous upward revision can be cancelled against the midnight baseline.
    assert answer["q"] == pytest.approx([100.0])
    assert answer["objective"] == pytest.approx(200.0)


def test_adjusted_revision_uses_original_commitment_once():
    answer = solve_schedule([[60.0]], [[2.0]], 6000, purchase=[140.0], original=[100.0], adjust_mask=[True], capacity=False, settlement="adjusted")
    assert answer["q"] == pytest.approx([60.0])
    assert answer["objective"] == pytest.approx(160.0)


def test_recovery_preserves_stored_energy_and_reduces_demand():
    c = np.array([10.0, 5.0, 0.0])
    d = np.array([4.0, 8.0, 2.0])
    cc, dd = recover_exclusive_dispatch(c, d)
    assert 0.9 * cc - dd / 0.9 == pytest.approx(0.9 * c - d / 0.9)
    assert np.max(np.minimum(cc, dd)) < 1e-12
    assert np.all(cc - dd <= c - d + 1e-12)


def test_q1_lp_and_binary_audit_agree():
    kwargs = dict(net_s=[[0.0, 4.0, 4.0, 0.0]], price_s=[[1.0, 3.0, 3.0, 1.0]], initial_soc=6000, terminal_soc=6000, allow_emergency=False)
    lp = solve_schedule(**kwargs)
    mip = solve_schedule(**kwargs, milp=True)
    assert lp["objective"] == pytest.approx(8.0 / 0.81, abs=1e-6)
    assert mip["objective"] == pytest.approx(lp["objective"], abs=1e-6)
    assert lp["soc"][-1] == pytest.approx(6000.0)
    assert lp["emergency"].max() < 1e-7
    assert np.min(lp["soc"]) >= 1200 - 1e-7
    assert np.max(np.minimum(lp["c"], lp["d"])) < 1e-7


def test_mixed_fixed_adjust_and_free_modes():
    answer = solve_schedule([[6.0, 6.0, 6.0]], [[1.0, 1.0, 1.0]], 6000, purchase=[2.0, np.nan, np.nan], original=[np.nan, 4.0, np.nan], mode=["fixed", "adjust", "free"], capacity=False)
    assert answer["q"] == pytest.approx([2.0, 6.0, 6.0])
    assert answer["emergency"] == pytest.approx(np.array([[4.0, 0.0, 0.0]]))
    assert answer["objective"] == pytest.approx(35.0)


def test_infeasible_no_emergency_case_raises():
    with pytest.raises(OptimizationError) as caught:
        solve_schedule([[10.0]], [[1.0]], 6000, purchase=[0.0], capacity=False, allow_emergency=False)
    assert caught.value.status == 2
    assert "message" in caught.value.diagnostics


def test_time_limit_incumbent_is_not_silently_returned(monkeypatch):
    def fake_solver(*args, **kwargs):
        return SimpleNamespace(status=1, message="Time limit reached", x=np.zeros(9), fun=0.0)
    monkeypatch.setattr("microgrid.optimize.linprog", fake_solver)
    with pytest.raises(OptimizationError) as caught:
        solve_schedule([[1.0]], [[1.0]], 6000, time_limit=0.001)
    assert caught.value.status == 1
    assert caught.value.diagnostics["has_incumbent"] is True


def test_invalid_mode_inputs_are_rejected():
    with pytest.raises(ValueError, match="original"):
        solve_schedule([[1.0]], [[1.0]], 6000, mode=["adjust"])
    with pytest.raises(ValueError, match="negative prices"):
        solve_schedule([[1.0]], [[-1.0]], 6000)


@pytest.mark.parametrize("original,expected", [(2.0, 7.0), (8.0, 8.0), (10.0, 9.0)])
def test_adjusted_seventy_ninety_percent_deadband(original, expected):
    answer = solve_schedule(np.arange(11.0)[:, None], np.ones((11, 1)), 6000, original=[original], mode=["adjust"], capacity=False, settlement="adjusted")
    assert answer["q"] == pytest.approx([expected])


def test_soc_input_roundoff_is_clipped_but_real_violations_are_rejected():
    answer = solve_schedule([[0.0]], [[1.0]], 10800 + 7e-12, terminal_soc=10800 + 7e-12)
    assert answer["soc"][0] == 10800
    assert answer["soc"][-1] == pytest.approx(10800)
    with pytest.raises(ValueError, match="initial_soc"):
        solve_schedule([[0.0]], [[1.0]], 10800 + 1e-5)
