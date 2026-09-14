"""Chronological simulation. Only the physical safety layer sees current demand."""
from dataclasses import dataclass, asdict
from pathlib import Path
import json
import time
import numpy as np
import pandas as pd

from .data import load_data
from .forecast import ForecastEngine
from .optimize import solve_schedule

STEPS = 144
CAP_MIN, CAP_MAX, INITIAL = 1200., 10800., 6000.
MAX_ENERGY = 5000. / 6.


@dataclass
class Policy:
    name: str = 'q2'
    pv_forecasts: bool = False
    variable_price: bool = False
    updates: tuple = ()
    scenarios: int = 12
    control_hours: int = 4
    eta: float = .9
    storage: bool = True
    settlement: str = 'adjusted'
    family: str = 'naive'
    warmup_naive: bool = True
    perfect_price: bool = False
    seed: int = 2026


def settlement_components(original, effective, price, emergency, settlement='adjusted'):
    """Settle the final commitment against the midnight plan once, at delivery prices."""
    if settlement not in ('adjusted', 'full_plan_plus_penalty'):
        raise ValueError('settlement must be adjusted or full_plan_plus_penalty')
    original, effective, price, emergency = map(np.asarray, (original, effective, price, emergency))
    base = price * original
    upward = 1.5 * price * np.maximum(effective - original, 0.)
    downward = (-.5 if settlement == 'adjusted' else .5) * price * np.maximum(original - effective, 0.)
    urgent = 5. * price * emergency
    return base, upward, downward, urgent


def execute_safe(q, load, pv, c_target, d_target, soc, eta=.9, storage=True):
    """Interval-internal load-priority tracking under piecewise-constant power.

    load and pv are realized interval energies. They never change the pre-interval
    optimizer or purchase commitment. Emergency supply cannot charge the battery.
    """
    available = q + pv - load
    if storage:
        c = min(max(c_target, 0.), max(available, 0.), MAX_ENERGY, max(0., (CAP_MAX-soc)/eta))
        d = min(max(d_target, 0.), max(-available, 0.), MAX_ENERGY, max(0., eta*(soc-CAP_MIN)))
    else:
        c = d = 0.
    urgent = max(-available-d, 0.)
    waste = max(available-c, 0.)
    end = soc + eta*c - d/eta
    return c, d, urgent, waste, end


def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, default=_json_default), encoding='utf-8')


def base_policies(families=None):
    families = families or {}
    return [
        Policy('q2', family=families.get('q2', 'naive')),
        Policy('q3', pv_forecasts=True, updates=(6,12,18), family=families.get('q3', 'naive')),
        Policy('q4_2', variable_price=True, family=families.get('q4_2', 'naive')),
        Policy('q4_3', pv_forecasts=True, variable_price=True, updates=(6,12,18), family=families.get('q4_3', 'naive')),
    ]


def simulate(policy, data=None, end_day=365, output_dir=None, verbose=True):
    data = data or load_data()
    n = end_day * STEPS
    runtime = time.perf_counter()
    engines = {family: ForecastEngine(data, family=family, seed=policy.seed)
               for family in set([policy.family, 'naive'])}
    allowed = tuple(sorted(set((0,) + tuple(policy.updates)))) if policy.pv_forecasts else None
    keys = ['plan', 'effective', 'charge', 'discharge', 'emergency', 'waste',
            'soc_start', 'soc_end', 'price', 'load', 'pv', 'base_fee', 'up_fee',
            'down_fee', 'emergency_fee', 'cost', 'target_charge', 'target_discharge',
            'issue_index', 'expected_net', 'expected_price']
    log = {k: np.zeros(n) for k in keys}
    revisions = []
    soc = INITIAL
    max_eq_residual = 0.
    solver_calls = 0
    forecast_mae = {'load': 0., 'pv': 0., 'price': 0., 'count': 0}

    def predict(engine, origin, horizon):
        pred = engine.forecast(origin=origin, horizon=horizon,
                               use_pv_forecasts=policy.pv_forecasts,
                               variable_price=policy.variable_price,
                               scenario_count=policy.scenarios,
                               allowed_issues=allowed)
        net_s = np.array(pred.net_s, copy=True)
        price_s = np.array(pred.price_s, copy=True)
        if policy.perfect_price:
            # Explicit information-ablation only; never used by a main policy.
            stop = min(origin+horizon, len(data.flat_price))
            price_s[:, :stop-origin] = data.flat_price[origin:stop]
        return pred, net_s, price_s

    for day in range(end_day):
        origin = day * STEPS
        family = 'naive' if policy.warmup_naive and day < 31 else policy.family
        engine = engines[family]
        pred, net_s, price_s = predict(engine, origin, 288)
        ref = solve_schedule(net_s, price_s, initial_soc=soc, terminal_soc=INITIAL,
                             eta=policy.eta, capacity=policy.storage)
        solver_calls += 1
        plan = ref['q'][:STEPS].copy()
        effective = plan.copy()
        reference_q = ref['q'].copy()
        reference_soc = ref['soc'].copy()
        last_issue = origin
        midnight_mean = (pred.mean_load[:STEPS], pred.mean_pv[:STEPS], pred.mean_price[:STEPS])
        for slot in range(STEPS):
            t = origin + slot
            if slot and slot % 36 == 0 and slot//6 in policy.updates:
                remaining = STEPS-slot
                pred_new, net_new, price_new = predict(engine, t, 288-slot)
                mode = np.array(['adjust']*remaining + ['free']*144)
                original = np.r_[plan[slot:], np.zeros(144)]
                current = np.r_[effective[slot:], reference_q[STEPS:]]
                revised = solve_schedule(net_new, price_new, initial_soc=soc,
                                         purchase=current, original=original,
                                         mode=mode, terminal_soc=INITIAL,
                                         eta=policy.eta, capacity=policy.storage,
                                         settlement=policy.settlement)
                solver_calls += 1
                effective[slot:] = revised['q'][:remaining]
                reference_q[slot:] = revised['q']
                reference_soc[slot:] = revised['soc']
                last_issue = t
                # Archive the whole committed remaining-day vector, not a re-settlement.
                revisions.append({'issue': int(t), 'day': int(day), 'slot': int(slot),
                                  'q': np.round(effective[slot:], 9).tolist()})
            horizon = policy.control_hours * 6
            pred_rt, net_rt, price_rt = predict(engine, t, horizon)
            q_fixed = reference_q[slot:slot+horizon].copy()
            reference_end = float(np.clip(reference_soc[slot+horizon],
                                         max(CAP_MIN,soc-horizon*MAX_ENERGY/policy.eta),
                                         min(CAP_MAX,soc+horizon*MAX_ENERGY*policy.eta)))
            if policy.storage:
                control = solve_schedule(net_rt, price_rt, initial_soc=soc,
                                         purchase=q_fixed, terminal_soc=reference_end,
                                         eta=policy.eta, capacity=True)
                solver_calls += 1
            else:
                control = {'c':np.zeros(horizon), 'd':np.zeros(horizon)}
            q = effective[slot]
            load, pv = data.load[day, slot]/6., data.pv[day, slot]/6.
            p = data.price[day, slot] if policy.variable_price else data.prior_price[slot]
            c, d, u, w, next_soc = execute_safe(q, load, pv, control['c'][0],
                                               control['d'][0], soc, policy.eta, policy.storage)
            fees = settlement_components(plan[slot], q, p, u, policy.settlement)
            row = [plan[slot], q, c, d, u, w, soc, next_soc, p, load, pv,
                   *fees, sum(fees), control['c'][0], control['d'][0], last_issue,
                   float(np.mean(net_rt[:, 0])), float(np.mean(price_rt[:, 0]))]
            for key, value in zip(keys, row):
                log[key][t] = value
            residual = q+pv+d+u-load-c-w
            max_eq_residual = max(max_eq_residual, abs(residual))
            if not (CAP_MIN-1e-6 <= next_soc <= CAP_MAX+1e-6):
                raise AssertionError(('SOC out of bounds', policy.name, t, next_soc))
            if max_eq_residual > 1e-6 or c*d > 1e-8 or c*u > 1e-8:
                raise AssertionError(('physical violation', policy.name, t, residual, c, d, u))
            soc = next_soc
        for metric, real, estimate in zip(('load', 'pv', 'price'),
                                         (data.load[day], data.pv[day], data.price[day] if policy.variable_price else data.prior_price),
                                         midnight_mean):
            if day >= 31:
                forecast_mae[metric] += float(np.abs(real-estimate).sum())
        if day >= 31:
            forecast_mae['count'] += STEPS
        if verbose and (day < 2 or (day+1) % 14 == 0 or day+1 == end_day):
            print(f'{policy.name} day {day+1}/{end_day} family={family} SOC={soc:.2f} '
                  f'elapsed={time.perf_counter()-runtime:.1f}s', flush=True)
    summary = summarize(log, policy, data)
    summary.update({'elapsed_seconds': time.perf_counter()-runtime, 'solver_calls': solver_calls,
                    'maximum_balance_error_kwh': max_eq_residual,
                    'forecast_mae': {k: v/max(1,forecast_mae['count']) for k,v in forecast_mae.items() if k != 'count'},
                    'configuration': asdict(policy)})
    if output_dir:
        dest = Path(output_dir)/policy.name
        dest.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(dest/'ledger.npz', **log)
        write_json(dest/'summary.json', summary)
        import gzip
        with gzip.open(dest/'revisions.json.gz', 'wt', encoding='utf-8') as stream:
            json.dump(revisions, stream, separators=(',', ':'))
        write_json(dest/'forecast_diagnostics.json', {k: v.diagnostics() for k,v in engines.items()})
    return log, summary


def summarize(log, policy, data):
    n = len(log['plan'])
    starts = {'whole_year': 0, 'evaluation': min(31*STEPS, n), 'january_validation': min(14*STEPS, n)}
    valuation = float(np.min(data.prior_price)/.9)
    result = {}
    for label, start in starts.items():
        stop = min(n,31*STEPS) if label == 'january_validation' else n
        if stop <= start:
            result[label] = {}
            continue
        totals = {k: float(log[k][start:stop].sum()) for k in
                  ('plan','effective','charge','discharge','emergency','waste','base_fee','up_fee','down_fee','emergency_fee','cost')}
        initial = float(log['soc_start'][start]); final = float(log['soc_end'][stop-1])
        urgent = log['emergency'][start:stop].reshape(-1, STEPS) > 1e-7
        events = urgent.copy(); events[:,1:] &= ~urgent[:,:-1]
        totals.update(initial_soc=initial, final_soc=final,
                      inventory_corrected_cost=totals['cost'] + valuation*(initial-final),
                      emergency_events=int(events.sum()), emergency_intervals=int(urgent.sum()),
                      soc_min=float(log['soc_end'][start:stop].min()),
                      soc_max=float(log['soc_end'][start:stop].max()), days=(stop-start)//STEPS)
        result[label] = totals
    return result


def solve_question_one(data, destination):
    path = Path(destination); path.mkdir(parents=True, exist_ok=True)
    net = ((data.prior_load-data.prior_pv)/6.)[None,:]
    price = data.prior_price[None,:]
    solved = solve_schedule(net,price,INITIAL,terminal_soc=INITIAL,allow_emergency=False)
    audited = solve_schedule(net,price,INITIAL,terminal_soc=INITIAL,allow_emergency=False,milp=True)
    np.savez_compressed(path/'ledger.npz', plan=solved['q'],charge=solved['c'],discharge=solved['d'],
                        soc=solved['soc'],price=data.prior_price,load=data.prior_load/6.,pv=data.prior_pv/6.)
    output = {'cost':float(data.prior_price@solved['q']), 'energy':float(solved['q'].sum()),
              'lp_objective':solved['objective'], 'milp_objective':audited['objective'],
              'absolute_lp_milp_gap':abs(solved['objective']-audited['objective']),
              'solver_diagnostics':solved.get('diagnostics',{}), 'milp_gap':audited['gap']}
    write_json(path/'summary.json',output)
    return output


def perfect_information_bound(data, destination):
    """Valid, deliberately optimistic full-information LP lower bound.

    A free final stock and known realized net demand/price relax the online problem.
    This model buys energy at base price without adjustment or emergency penalties.
    """
    outputs = {}
    for mode in ('fixed','variable'):
        net = ((data.flat_load-data.flat_pv)/6.)[None,:]
        prices = (data.flat_price if mode=='variable' else np.tile(data.prior_price,365))[None,:]
        result = solve_schedule(net,prices,INITIAL,terminal_soc=None,allow_emergency=False,time_limit=600.)
        outputs[mode] = {'cost':float(result['objective']), 'energy':float(result['q'].sum()),
                         'initial_soc':INITIAL,'final_soc':float(result['soc'][-1]),'gap':result['gap']}
    write_json(destination,outputs)
    return outputs
