"""Question 2 primary policy: causal rolling forecasts and quantile contracts.

The 48-hour linear program is used only for the day-ahead purchase contract.
During operation the current ten-minute observation is handled by a transparent
safety feedback: charge realized surplus, or discharge before emergency supply.
No unexecuted intraday MPC path is part of this implementation.
"""
from __future__ import annotations

from dataclasses import asdict
import gzip
import json
from pathlib import Path
import time

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import csr_matrix, eye, hstack, lil_matrix, vstack

from .data import load_data
from .simulate import (CAP_MAX, CAP_MIN, INITIAL, MAX_ENERGY, STEPS, Policy,
                       settlement_components, summarize, write_json)


class JointForecaster:
    """Forecast load and PV using only observations before the operating day."""

    def __init__(self, load, pv, prior_load, prior_pv,
                 validation_days=14, residual_days=30):
        self.load = np.asarray(load, dtype=float)
        self.pv = np.asarray(pv, dtype=float)
        self.prior_load = np.asarray(prior_load, dtype=float)
        self.prior_pv = np.asarray(prior_pv, dtype=float)
        self.validation_days = validation_days
        self.residual_days = residual_days
        self._weights = {}
        self._forecast = {}

    def bases(self, values, day):
        if day <= 0:
            prior = self.prior_load if values is self.load else self.prior_pv
            return np.vstack([prior, prior, prior])
        previous = values[day - 1]
        if day < 7:
            return np.vstack([previous, previous, np.median(values[:day], axis=0)])
        return np.vstack([previous, values[day - 7],
                          np.median(values[day - 7:day], axis=0)])

    def weights(self, day):
        if day in self._weights:
            return tuple(value.copy() for value in self._weights[day])
        if day < 8:
            answer = (np.array([1., 0., 0.]), np.array([1., 0., 0.]))
            self._weights[day] = answer
            return tuple(value.copy() for value in answer)
        start = max(0, day - self.validation_days)
        load_basis, pv_basis, target = [], [], []
        for sample in range(start, day):
            load_basis.append(self.bases(self.load, sample).T)
            pv_basis.append(self.bases(self.pv, sample).T)
            target.append(self.load[sample] - self.pv[sample])
        load_basis = np.vstack(load_basis)
        pv_basis = np.vstack(pv_basis)
        target = np.concatenate(target)
        observations = len(target)
        prediction = hstack([csr_matrix(load_basis), -csr_matrix(pv_basis)], format="csr")
        identity = eye(observations, format="csr")
        inequality = vstack([hstack([prediction, -identity]),
                              hstack([-prediction, -identity])], format="csr")
        equality = lil_matrix((2, 6 + observations))
        equality[0, :3] = 1
        equality[1, 3:6] = 1
        solution = linprog(np.r_[np.zeros(6), np.ones(observations)],
                           A_ub=inequality, b_ub=np.r_[target, -target],
                           A_eq=equality.tocsr(), b_eq=[1, 1],
                           bounds=[(0, 1)] * 6 + [(0, None)] * observations,
                           method="highs")
        if solution.success:
            load_weights, pv_weights = solution.x[:3], solution.x[3:6]
            load_weights /= load_weights.sum()
            pv_weights /= pv_weights.sum()
        else:
            load_weights = pv_weights = np.array([1., 0., 0.])
        self._weights[day] = (load_weights, pv_weights)
        return load_weights.copy(), pv_weights.copy()

    def pair(self, day):
        if day not in self._forecast:
            load_weights, pv_weights = self.weights(day)
            load = np.maximum(load_weights @ self.bases(self.load, day), 0)
            pv = np.maximum(pv_weights @ self.bases(self.pv, day), 0)
            self._forecast[day] = (load, pv, load_weights, pv_weights)
        return tuple(value.copy() for value in self._forecast[day])

    def tomorrow(self, day, today_load, today_pv, load_weights, pv_weights):
        if day + 1 >= len(self.load):
            return today_load.copy(), today_pv.copy()

        def future_bases(values, today_prediction):
            weekly = values[day - 6] if day >= 6 else values[max(day - 1, 0)]
            known = [values[j] for j in range(max(0, day - 6), day)] + [today_prediction]
            return np.vstack([today_prediction, weekly, np.median(known, axis=0)])

        return (np.maximum(load_weights @ future_bases(self.load, today_load), 0),
                np.maximum(pv_weights @ future_bases(self.pv, today_pv), 0))

    def residual_margin(self, day, quantile):
        start = max(0, day - self.residual_days)
        residuals = []
        for sample in range(start, day):
            load, pv, _, _ = self.pair(sample)
            residuals.append((self.load[sample] - self.pv[sample]) - (load - pv))
        if not residuals:
            return np.zeros(STEPS)
        residuals = np.asarray(residuals)
        slot = np.quantile(residuals, quantile, axis=0)
        margin = np.empty(STEPS)
        for hour in range(24):
            section = slice(6 * hour, 6 * (hour + 1))
            margin[section] = (slot[section] +
                               np.quantile(residuals[:, section], quantile)) / 2
        return margin


def solve_contract_48h(net, price, initial_soc, eta=.9, storage=True):
    """Minimize 48-hour contract cost and restore 6000 kWh at horizon end."""
    net = np.asarray(net, dtype=float)
    price = np.asarray(price, dtype=float)
    if not storage:
        return np.maximum(net, 0), np.full(len(net) + 1, initial_soc)
    count = len(net)
    grid, charge, discharge, waste, state = 0, count, 2 * count, 3 * count, 4 * count
    variables = 5 * count + 1
    objective = np.zeros(variables)
    objective[grid:grid + count] = price
    objective[charge:charge + count] = 1e-7
    objective[discharge:discharge + count] = 1e-7
    objective[waste:waste + count] = 1e-8
    equality = lil_matrix((2 * count + 2, variables))
    target = np.zeros(2 * count + 2)
    for step in range(count):
        equality[step, grid + step] = 1
        equality[step, discharge + step] = 1
        equality[step, charge + step] = -1
        equality[step, waste + step] = -1
        target[step] = net[step]
        row = count + step
        equality[row, state + step + 1] = 1
        equality[row, state + step] = -1
        equality[row, charge + step] = -eta
        equality[row, discharge + step] = 1 / eta
    equality[2 * count, state] = 1
    target[2 * count] = initial_soc
    equality[2 * count + 1, state + count] = 1
    target[2 * count + 1] = INITIAL
    bounds = ([(0, None)] * count + [(0, MAX_ENERGY)] * count * 2 +
              [(0, None)] * count + [(CAP_MIN, CAP_MAX)] * (count + 1))
    solution = linprog(objective, A_eq=equality.tocsr(), b_eq=target,
                       bounds=bounds, method="highs")
    if not solution.success:
        raise RuntimeError("quantile contract LP failed: " + solution.message)
    return solution.x[grid:grid + count], solution.x[state:state + count + 1]


def simulate_quantile_q2(quantile=.8, data=None, end_day=365, output_dir=None,
                         name="q2", use_margin=True, storage=True, verbose=True):
    """Run the unique official Q2 primary implementation and common ledger."""
    if not 0 < quantile < 1:
        raise ValueError("quantile must lie strictly between zero and one")
    data = data or load_data()
    count = end_day * STEPS
    actual_load = data.load[:end_day] / 6
    actual_pv = data.pv[:end_day] / 6
    fixed_price = np.asarray(data.prior_price, dtype=float)
    forecaster = JointForecaster(actual_load, actual_pv,
                                 data.prior_load / 6, data.prior_pv / 6)
    keys = ["plan", "effective", "charge", "discharge", "emergency", "waste",
            "soc_start", "soc_end", "price", "load", "pv", "base_fee", "up_fee",
            "down_fee", "emergency_fee", "cost", "target_charge", "target_discharge",
            "issue_index", "expected_net", "expected_price"]
    log = {key: np.zeros(count) for key in keys}
    state = INITIAL
    start_time = time.perf_counter()
    maximum_balance = 0.
    forecast_error = {"load": 0., "pv": 0., "price": 0., "count": 0}
    fits = []
    for day in range(end_day):
        load_forecast, pv_forecast, load_weights, pv_weights = forecaster.pair(day)
        margin = forecaster.residual_margin(day, quantile) if use_margin else np.zeros(STEPS)
        today_net = np.maximum(load_forecast - pv_forecast + margin, 0)
        next_load, next_pv = forecaster.tomorrow(day, load_forecast, pv_forecast,
                                                load_weights, pv_weights)
        tomorrow_net = np.maximum(next_load - next_pv + margin, 0)
        plan48, _ = solve_contract_48h(np.r_[today_net, tomorrow_net],
                                      np.tile(fixed_price, 2), state,
                                      storage=storage)
        plan = plan48[:STEPS]
        origin = day * STEPS
        fits.append({"day": day, "training_actual_end_exclusive": origin,
                     "validation_days": min(14, day),
                     "residual_days": min(30, day),
                     "load_weights": load_weights.tolist(),
                     "pv_weights": pv_weights.tolist()})
        for slot in range(STEPS):
            index = origin + slot
            load = actual_load[day, slot]
            pv = actual_pv[day, slot]
            available = plan[slot] + pv - load
            if storage and available >= 0:
                charge = min(available, MAX_ENERGY, max(0., (CAP_MAX - state) / .9))
                discharge = emergency = 0.
                waste = max(available - charge, 0.)
            elif storage:
                charge = waste = 0.
                discharge = min(-available, MAX_ENERGY, max(0., .9 * (state - CAP_MIN)))
                emergency = max(-available - discharge, 0.)
            else:
                charge = discharge = 0.
                emergency = max(-available, 0.)
                waste = max(available, 0.)
            next_state = state + .9 * charge - discharge / .9
            fees = settlement_components(plan[slot], plan[slot], fixed_price[slot], emergency)
            row = [plan[slot], plan[slot], charge, discharge, emergency, waste,
                   state, next_state, fixed_price[slot], load, pv, *fees, sum(fees),
                   charge, discharge, origin, load_forecast[slot] - pv_forecast[slot],
                   fixed_price[slot]]
            for key, value in zip(keys, row):
                log[key][index] = value
            residual = plan[slot] + pv + discharge + emergency - load - charge - waste
            maximum_balance = max(maximum_balance, abs(float(residual)))
            state = float(np.clip(next_state, CAP_MIN, CAP_MAX))
        if day >= 31:
            forecast_error["load"] += float(np.abs(data.load[day] - load_forecast * 6).sum())
            forecast_error["pv"] += float(np.abs(data.pv[day] - pv_forecast * 6).sum())
            forecast_error["price"] += 0.
            forecast_error["count"] += STEPS
        if verbose and (day < 2 or (day + 1) % 30 == 0 or day + 1 == end_day):
            print(f"{name} day {day + 1}/{end_day} q={quantile:.2f} SOC={state:.2f} ",
                  f"elapsed={time.perf_counter() - start_time:.1f}s", flush=True)
    policy = Policy(name=name, scenarios=1, control_hours=0, storage=storage,
                    family="joint_causal_quantile")
    summary = summarize(log, policy, data)
    configuration = asdict(policy)
    configuration.update(model="rolling_quantile", quantile=quantile,
                         quantile_margin=bool(use_margin), feedback="immediate_safe")
    summary.update(elapsed_seconds=time.perf_counter() - start_time,
                   solver_calls=end_day, maximum_balance_error_kwh=maximum_balance,
                   forecast_mae={key: value / max(1, forecast_error["count"])
                                 for key, value in forecast_error.items() if key != "count"},
                   configuration=configuration)
    if output_dir:
        destination = Path(output_dir) / name
        destination.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(destination / "ledger.npz", **log)
        write_json(destination / "summary.json", summary)
        with gzip.open(destination / "revisions.json.gz", "wt", encoding="utf-8") as stream:
            json.dump([], stream, separators=(",", ":"))
        write_json(destination / "forecast_diagnostics.json",
                   {"joint_causal_quantile": {"model": "joint_causal_quantile",
                                                "daily_fits": fits}})
    return log, summary
