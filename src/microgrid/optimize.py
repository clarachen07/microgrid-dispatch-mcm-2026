"""Sparse reference scheduling; operational safety feedback is implemented elsewhere.

All flows are kWh per interval. Scenarios share a reference battery trajectory;
only emergency supply differs. This is not a full multistage stochastic policy.
"""

import time
from typing import Optional

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, linprog, milp as scipy_milp
from scipy.sparse import coo_matrix, vstack


SOC_MIN = 1200.0
SOC_MAX = 10800.0
MAX_INTERVAL_ENERGY = 5000.0 / 6.0


class OptimizationError(RuntimeError):
    """The solver failed to certify an optimum or returned an invalid schedule."""

    def __init__(self, message, status=None, diagnostics=None):
        super().__init__(message)
        self.status = status
        self.diagnostics = diagnostics or {}


def adjustment_cost(q, original, price, settlement="adjusted"):
    """Total final settlement against the ORIGINAL midnight commitment.

    This includes the original purchase charge, not just the adjustment charge.
    Revisions are never summed against successive intermediate commitments.
    """
    if settlement not in ("adjusted", "full_plan_plus_penalty"):
        raise ValueError("settlement must be 'adjusted' or 'full_plan_plus_penalty'")
    q, original, price = np.broadcast_arrays(q, original, price)
    down_sign = -1.0 if settlement == "adjusted" else 1.0
    return price * (
        original + 1.5 * np.maximum(q - original, 0.0)
        + down_sign * 0.5 * np.maximum(original - q, 0.0)
    )


def recover_exclusive_dispatch(c, d, eta=0.9):
    """Remove simultaneous charge/discharge without changing the SOC path.

    AC demand decreases; unrestricted, free unused energy makes recovery feasible.
    The caller must recompute emergency supply and the objective after recovery.
    """
    if not 0.0 < eta <= 1.0:
        raise ValueError("eta must be in (0, 1]")
    c = np.maximum(np.asarray(c, dtype=float), 0.0).copy()
    d = np.maximum(np.asarray(d, dtype=float), 0.0).copy()
    if c.shape != d.shape:
        raise ValueError("charge and discharge must have the same shape")
    delta = np.minimum(c, d / (eta * eta))
    return np.maximum(c - delta, 0.0), np.maximum(d - eta * eta * delta, 0.0)


def _vector(value, horizon, name, required=False):
    if value is None:
        if required:
            raise ValueError("{} is required for the requested mode".format(name))
        return None
    result = np.asarray(value, dtype=float)
    if result.shape != (horizon,):
        raise ValueError("{} must have shape ({},)".format(name, horizon))
    return result


def solve_schedule(
    net_s: np.ndarray,
    price_s: np.ndarray,
    initial_soc: float,
    purchase: Optional[np.ndarray] = None,
    original: Optional[np.ndarray] = None,
    adjust_mask: Optional[np.ndarray] = None,
    terminal_soc: Optional[float] = None,
    eta: float = 0.9,
    capacity: bool = True,
    settlement: str = "adjusted",
    milp: bool = False,
    mode: Optional[np.ndarray] = None,
    allow_emergency: bool = True,
    time_limit: Optional[float] = 60.0,
):
    """Return an optimal finite-horizon reference schedule or raise an error.

    ``net_s`` is load minus PV, shape (K,H); ``price_s`` is (K,H) or (H,).
    Scenarios are equiprobable. ``mode`` optionally specifies 'free', 'fixed',
    or 'adjust' per interval. Without it, purchase=None means free purchases;
    a purchase vector means fixed purchases except where adjust_mask is true.
    An adjust interval is settled against ``original``. Other intervals use
    ordinary p*q, including fixed intervals (constant for this optimization).
    Irrelevant entries of purchase/original may be NaN; used entries must be
    finite and nonnegative. capacity=False disables the battery. Terminal SOC
    is an optional hard constraint. Time-limited incumbents are never silently
    accepted as optimal. Negative electricity prices need a different bounded
    market formulation and are explicitly rejected by this reference model.
    """
    start = time.perf_counter()
    net = np.asarray(net_s, dtype=float)
    if net.ndim == 1:
        net = net[None, :]
    if net.ndim != 2 or not net.shape[0] or not net.shape[1]:
        raise ValueError("net_s must be a nonempty (K,H) array")
    k_count, horizon = net.shape
    try:
        prices = np.broadcast_to(np.asarray(price_s, dtype=float), net.shape)
    except ValueError as exc:
        raise ValueError("price_s must broadcast to the shape of net_s") from exc
    if not np.all(np.isfinite(net)) or not np.all(np.isfinite(prices)):
        raise ValueError("net_s and price_s must contain finite values")
    if np.any(prices < 0):
        raise ValueError("negative prices require bounded purchases and a different recourse formulation")
    if not 0.0 < eta <= 1.0:
        raise ValueError("eta must be in (0, 1]")
    if not isinstance(capacity, (bool, np.bool_)):
        raise ValueError("capacity is a boolean battery-enable switch")
    if settlement not in ("adjusted", "full_plan_plus_penalty"):
        raise ValueError("settlement must be 'adjusted' or 'full_plan_plus_penalty'")
    if time_limit is not None and (not np.isfinite(time_limit) or time_limit <= 0):
        raise ValueError("time_limit must be positive or None")
    soc_input_tolerance = 1e-7
    if not np.isfinite(initial_soc) or not SOC_MIN - soc_input_tolerance <= initial_soc <= SOC_MAX + soc_input_tolerance:
        raise ValueError("initial_soc must be in [1200,10800]")
    initial_soc = float(np.clip(initial_soc, SOC_MIN, SOC_MAX))
    if terminal_soc is not None and (
        not np.isfinite(terminal_soc) or not SOC_MIN - soc_input_tolerance <= terminal_soc <= SOC_MAX + soc_input_tolerance
    ):
        raise ValueError("terminal_soc must be in [1200,10800]")
    if terminal_soc is not None:
        terminal_soc = float(np.clip(terminal_soc, SOC_MIN, SOC_MAX))

    planned = _vector(purchase, horizon, "purchase")
    midnight = _vector(original, horizon, "original")
    if mode is None:
        modes = np.full(horizon, "free" if planned is None else "fixed", dtype="U6")
        if adjust_mask is not None:
            mask = np.asarray(adjust_mask, dtype=bool)
            if mask.shape != (horizon,):
                raise ValueError("adjust_mask must have shape (H,)")
            modes[mask] = "adjust"
    else:
        modes = np.asarray(mode, dtype="U6")
        if modes.shape != (horizon,) or not np.all(np.isin(modes, ["free", "fixed", "adjust"])):
            raise ValueError("mode must contain H values: free, fixed, or adjust")
        if adjust_mask is not None:
            mask = np.asarray(adjust_mask, dtype=bool)
            if mask.shape != (horizon,) or not np.array_equal(mask, modes == "adjust"):
                raise ValueError("adjust_mask conflicts with explicit mode")
    fixed = modes == "fixed"
    adjusted = modes == "adjust"
    if np.any(fixed):
        if planned is None or np.any(~np.isfinite(planned[fixed])) or np.any(planned[fixed] < 0):
            raise ValueError("fixed intervals require finite nonnegative purchase values")
    if np.any(adjusted):
        if midnight is None or np.any(~np.isfinite(midnight[adjusted])) or np.any(midnight[adjusted] < 0):
            raise ValueError("adjust intervals require finite nonnegative original values")

    # q, c, d, S[0:H+1], emergency[K,H], up[A], down[A], binary[H].
    q_idx = np.arange(horizon)
    c_idx = q_idx + horizon
    d_idx = q_idx + 2 * horizon
    soc_idx = np.arange(horizon + 1) + 3 * horizon
    emergency_idx = np.arange(k_count * horizon).reshape(k_count, horizon) + 4 * horizon + 1
    offset = 4 * horizon + 1 + k_count * horizon
    adj_times = np.flatnonzero(adjusted)
    up_idx = np.arange(adj_times.size) + offset
    down_idx = up_idx + adj_times.size
    offset += 2 * adj_times.size
    binary_idx = np.arange(horizon) + offset if milp and capacity else np.array([], dtype=int)
    nvars = offset + binary_idx.size

    objective = np.zeros(nvars)
    mean_price = prices.mean(axis=0)
    objective[q_idx[~adjusted]] = mean_price[~adjusted]
    objective[emergency_idx.ravel()] = (5.0 * prices / k_count).ravel()
    objective[up_idx] = 1.5 * mean_price[adjusted]
    objective[down_idx] = (-0.5 if settlement == "adjusted" else 0.5) * mean_price[adjusted]
    constant_cost = float(np.dot(mean_price[adjusted], midnight[adjusted])) if np.any(adjusted) else 0.0
    lower = np.zeros(nvars)
    upper = np.full(nvars, np.inf)
    upper[c_idx] = upper[d_idx] = MAX_INTERVAL_ENERGY if capacity else 0.0
    lower[soc_idx], upper[soc_idx] = SOC_MIN, SOC_MAX
    lower[soc_idx[0]] = upper[soc_idx[0]] = initial_soc
    if terminal_soc is not None:
        lower[soc_idx[-1]] = upper[soc_idx[-1]] = terminal_soc
    if np.any(fixed):
        lower[q_idx[fixed]] = upper[q_idx[fixed]] = planned[fixed]
    if not allow_emergency:
        upper[emergency_idx.ravel()] = 0.0
    if binary_idx.size:
        upper[binary_idx] = 1.0

    eq_rows, eq_cols, eq_values, eq_rhs = [], [], [], []
    ub_rows, ub_cols, ub_values, ub_rhs = [], [], [], []

    def add_eq(cols, values, rhs):
        eq_rows.extend([len(eq_rhs)] * len(cols))
        eq_cols.extend(cols)
        eq_values.extend(values)
        eq_rhs.append(rhs)

    def add_ub(cols, values, rhs):
        ub_rows.extend([len(ub_rhs)] * len(cols))
        ub_cols.extend(cols)
        ub_values.extend(values)
        ub_rhs.append(rhs)

    for t in range(horizon):
        add_eq([soc_idx[t + 1], soc_idx[t], c_idx[t], d_idx[t]], [1.0, -1.0, -eta, 1.0 / eta], 0.0)
    for j, t in enumerate(adj_times):
        add_eq([q_idx[t], up_idx[j], down_idx[j]], [1.0, -1.0, 1.0], midnight[t])
    for s in range(k_count):
        for t in range(horizon):
            add_ub([c_idx[t], d_idx[t], q_idx[t], emergency_idx[s, t]], [1.0, -1.0, -1.0, -1.0], -net[s, t])
    if binary_idx.size:
        for t in range(horizon):
            add_ub([c_idx[t], binary_idx[t]], [1.0, -MAX_INTERVAL_ENERGY], 0.0)
            add_ub([d_idx[t], binary_idx[t]], [1.0, MAX_INTERVAL_ENERGY], MAX_INTERVAL_ENERGY)
    a_eq = coo_matrix((eq_values, (eq_rows, eq_cols)), shape=(len(eq_rhs), nvars)).tocsr()
    a_ub = coo_matrix((ub_values, (ub_rows, ub_cols)), shape=(len(ub_rhs), nvars)).tocsr()
    b_eq, b_ub = np.asarray(eq_rhs), np.asarray(ub_rhs)
    options = {"presolve": True}
    if time_limit is not None:
        options["time_limit"] = float(time_limit)
    if milp:
        integrality = np.zeros(nvars, dtype=int)
        integrality[binary_idx] = 1
        constraints = LinearConstraint(
            vstack([a_eq, a_ub], format="csc"),
            np.concatenate([b_eq, np.full(len(b_ub), -np.inf)]),
            np.concatenate([b_eq, b_ub]),
        )
        options["mip_rel_gap"] = 1e-8
        result = scipy_milp(objective, integrality=integrality, bounds=Bounds(lower, upper), constraints=constraints, options=options)
    else:
        options.update(primal_feasibility_tolerance=1e-8, dual_feasibility_tolerance=1e-8)
        result = linprog(objective, A_ub=a_ub, b_ub=b_ub, A_eq=a_eq, b_eq=b_eq, bounds=np.column_stack([lower, upper]), method="highs", options=options)
    diagnostics = {
        "solver": "HiGHS MILP" if milp else "HiGHS LP",
        "solver_status": int(result.status),
        "message": str(result.message),
        "elapsed_seconds": time.perf_counter() - start,
        "scenarios": k_count,
        "horizon": horizon,
    }
    if result.status != 0 or result.x is None:
        diagnostics["has_incumbent"] = result.x is not None
        raise OptimizationError("Reference optimization failed: {}".format(result.message), result.status, diagnostics)

    x = result.x
    q = np.maximum(x[q_idx], 0.0)
    c, d = recover_exclusive_dispatch(x[c_idx], x[d_idx], eta)
    soc = np.r_[float(initial_soc), initial_soc + np.cumsum(eta * c - d / eta)]
    emergency = np.maximum(net + c[None, :] - d[None, :] - q[None, :], 0.0)
    spill = q[None, :] + d[None, :] + emergency - net - c[None, :]
    costs = mean_price * q
    if np.any(adjusted):
        costs[adjusted] = adjustment_cost(q[adjusted], midnight[adjusted], mean_price[adjusted], settlement)
    actual_objective = float(costs.sum() + np.mean(np.sum(5.0 * prices * emergency, axis=1)))
    raw_objective = float(result.fun + constant_cost)
    tolerance = 2e-5
    violations = {
        "soc_lower": float(max(0.0, SOC_MIN - soc.min())),
        "soc_upper": float(max(0.0, soc.max() - SOC_MAX)),
        "charge_limit": float(max(0.0, c.max() - (MAX_INTERVAL_ENERGY if capacity else 0.0))),
        "discharge_limit": float(max(0.0, d.max() - (MAX_INTERVAL_ENERGY if capacity else 0.0))),
        "terminal": 0.0 if terminal_soc is None else float(abs(soc[-1] - terminal_soc)),
        "fixed_purchase": 0.0 if not np.any(fixed) else float(np.max(np.abs(q[fixed] - planned[fixed]))),
        "forbidden_emergency": 0.0 if allow_emergency else float(emergency.max()),
        "simultaneous": float(np.minimum(c, d).max()),
        "negative_spill": float(max(0.0, -spill.min())),
    }
    diagnostics.update(raw_objective=raw_objective, recovered_objective=actual_objective, max_violation=max(violations.values()), violations=violations)
    if max(violations.values()) > tolerance:
        raise OptimizationError("Recovered reference schedule violates physical constraints", result.status, diagnostics)
    if abs(actual_objective - raw_objective) > 1e-4 + 1e-7 * max(1.0, abs(raw_objective)):
        raise OptimizationError("Recovered objective does not match the solver certificate", result.status, diagnostics)
    gap = float(getattr(result, "mip_gap", 0.0) or 0.0)
    return {
        "q": q, "c": c, "d": d, "soc": soc,
        "emergency": emergency, "spill": spill,
        "objective": actual_objective, "gap": gap, "status": "optimal",
        "diagnostics": diagnostics,
    }
