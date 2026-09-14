"""Validate physical, settlement, and workbook constraints for primary runs."""
from pathlib import Path
import datetime as dt
import json

import numpy as np
from openpyxl import load_workbook

from .data import load_data

ROOT = Path(__file__).resolve().parents[2]
NAMES = {"q1": "result1.xlsx", "q2": "result2.xlsx", "q3": "result3.xlsx",
         "q4_2": "result4-2.xlsx", "q4_3": "result4-3.xlsx"}
TOL = 1e-6


class VerificationError(RuntimeError):
    pass


def _maxabs(values):
    a = np.asarray(values, dtype=float)
    return float(np.max(np.abs(a))) if a.size else 0.0


def _period(a, b):
    return "{:02d}:{:02d}-{:02d}:{:02d}".format(a // 6, a % 6 * 10, b // 6, b % 6 * 10)


def _date(day):
    return (dt.datetime(2025, 1, 1) + dt.timedelta(days=int(day))).date()


def _asdate(value):
    if isinstance(value, dt.datetime):
        return value.date()
    if isinstance(value, dt.date):
        return value
    return dt.datetime.fromisoformat(str(value)).date()


def _events(values):
    """Independent scan; adjacent events cannot merge across calendar days."""
    answer, start = [], None
    for i, value in enumerate(np.r_[values, 0.0]):
        if value > 1e-7 and start is None:
            start = i
        elif value <= 1e-7 and start is not None:
            answer.append((_period(start, i), float(np.sum(values[start:i]))))
            start = None
    return answer


def verify_ledger(log, summary, name, expected_steps=None):
    """Verify one ledger without calling the implementation's safety/fee helpers."""
    errors = []
    n = len(log["plan"])
    def check(condition, message):
        if not condition:
            errors.append(message)
    check(expected_steps is None or n == expected_steps, "wrong number of intervals")
    for key, value in log.items():
        check(np.asarray(value).ndim == 1 and len(value) == (n + 1 if key == "soc" else n), "wrong shape: " + key)
        check(np.isfinite(value).all(), "nonfinite values: " + key)
    eta = float(summary.get("configuration", {}).get("eta", .9))
    q = log.get("effective", log["plan"])
    c, d = log["charge"], log["discharge"]
    u = log.get("emergency", np.zeros(n))
    if name == "q1":
        begin, end = log["soc"][:-1], log["soc"][1:]
    else:
        begin, end = log["soc_start"], log["soc_end"]
    surplus = q + log["pv"] + d + u - log["load"] - c
    waste = log.get("waste", np.maximum(surplus, 0.0))
    metrics = {
        "max_energy_residual_kwh": _maxabs(surplus - waste),
        "max_soc_equation_residual_kwh": _maxabs(end - begin - eta * c + d / eta),
        "max_soc_continuity_error_kwh": _maxabs(begin[1:] - end[:-1]),
        "max_charge_discharge_overlap_kwh": float(np.minimum(c, d).max()),
        "max_emergency_charge_overlap_kwh": float(np.minimum(c, u).max()),
        "max_power_excess_kwh": float(max(0.0, c.max() - 5000 / 6, d.max() - 5000 / 6)),
        "soc_min_kwh": float(min(begin.min(), end.min())),
        "soc_max_kwh": float(max(begin.max(), end.max())),
        "initial_soc_kwh": float(begin[0]),
    }
    for metric in ("max_energy_residual_kwh", "max_soc_equation_residual_kwh", "max_soc_continuity_error_kwh", "max_charge_discharge_overlap_kwh", "max_emergency_charge_overlap_kwh", "max_power_excess_kwh"):
        check(metrics[metric] <= TOL, metric + " exceeds tolerance")
    check(abs(begin[0] - 6000) <= TOL, "initial state is not 6000 kWh")
    check(metrics["soc_min_kwh"] >= 1200 - TOL and metrics["soc_max_kwh"] <= 10800 + TOL, "SOC bound violated")
    for key in ("plan", "effective", "charge", "discharge", "emergency", "waste"):
        if key in log:
            check(np.min(log[key]) >= -TOL, "negative physical flow: " + key)
    if name == "q1":
        check(abs(end[-1] - begin[0]) <= TOL, "Q1 terminal SOC differs from initial")
        check(np.min(surplus) >= -TOL, "Q1 supply below demand")
        check(abs(log["price"] @ q - summary["cost"]) <= 1e-5, "Q1 cost mismatch")
        check(abs(q.sum() - summary["energy"]) <= 1e-5, "Q1 purchase total mismatch")
        gap = abs(summary["lp_objective"] - summary["milp_objective"])
        metrics["q1_lp_milp_absolute_gap_yuan"] = float(gap)
        check(gap <= 1e-4 and abs(gap - summary["absolute_lp_milp_gap"]) <= 1e-6, "Q1 LP/MILP consistency mismatch")
        check(summary.get("milp_gap", 1) <= 1e-7, "Q1 MILP gap is not certified")
    else:
        config = summary["configuration"]
        natural = config["settlement"] == "adjusted"
        fees = {
            "base_fee": log["price"] * log["plan"],
            "up_fee": 1.5 * log["price"] * np.maximum(q - log["plan"], 0),
            "down_fee": (-.5 if natural else .5) * log["price"] * np.maximum(log["plan"] - q, 0),
            "emergency_fee": 5 * log["price"] * u,
        }
        fees["cost"] = sum(fees.values())
        metrics["max_fee_error_yuan"] = max(_maxabs(log[key] - value) for key, value in fees.items())
        check(metrics["max_fee_error_yuan"] <= 1e-5, "fees are not a single settlement against the midnight plan")
        if config["settlement"] == "full_plan_plus_penalty":
            check(np.min(q - log["plan"]) >= -TOL, "full-plan-plus-penalty settlement contains a dominated downward commitment")
        if not config.get("updates"):
            check(_maxabs(q - log["plan"]) <= TOL, "purchase changed despite no allowed revisions")
        # Recompute safety clipping from reference targets, independently.
        if "target_charge" in log and "target_discharge" in log:
            available = q + log["pv"] - log["load"]
            ec = np.minimum.reduce([np.maximum(log["target_charge"], 0), np.maximum(available, 0), np.full(n, 5000 / 6), np.maximum((10800 - begin) / eta, 0)])
            ed = np.minimum.reduce([np.maximum(log["target_discharge"], 0), np.maximum(-available, 0), np.full(n, 5000 / 6), np.maximum(eta * (begin - 1200), 0)])
            if not config["storage"]:
                ec[:] = ed[:] = 0
            metrics["max_safe_tracking_error_kwh"] = max(_maxabs(ec - c), _maxabs(ed - d), _maxabs(u - np.maximum(-available - ed, 0)))
            check(metrics["max_safe_tracking_error_kwh"] <= TOL, "actual controls do not follow the specified safety policy")
        if "issue_index" in log:
            check(np.all(log["issue_index"] <= np.arange(n)), "a procurement revision occurs before its release")
        for section, a, b in (("whole_year", 0, n), ("evaluation", min(31 * 144, n), n), ("january_validation", min(14 * 144, n), min(31 * 144, n))):
            if b <= a:
                continue
            saved = summary[section]
            check(saved["days"] == (b - a) // 144, "wrong number of days in " + section)
            for key in ("plan", "effective", "charge", "discharge", "emergency", "waste", "base_fee", "up_fee", "down_fee", "emergency_fee", "cost"):
                check(abs(log[key][a:b].sum() - saved[key]) <= 1e-4, "summary total mismatch: " + section + "/" + key)
            check(abs(saved["initial_soc"] - begin[a]) <= TOL and abs(saved["final_soc"] - end[b - 1]) <= TOL, "summary inventory mismatch: " + section)
            count = sum(len(_events(u[t:t + 144])) for t in range(a, b, 144))
            check(saved["emergency_events"] == count, "summary event count mismatch: " + section)
    return {"passed": not errors, "intervals": n, "errors": errors, **metrics}


def verify_workbook(path, name, log):
    errors, comparisons = [], 0
    def same(actual, expected, label):
        nonlocal comparisons
        comparisons += 1
        try:
            match = np.isfinite(float(actual)) and abs(float(actual) - float(expected)) <= 1e-5
        except (TypeError, ValueError):
            match = False
        if not match:
            errors.append(label)
    def check(condition, label):
        if not condition:
            errors.append(label)
    book = load_workbook(path, read_only=True, data_only=True)
    expected_sheets = ["计划购电量", "充放电量"] if name == "q1" else ["计划购电量", "充放电量", "紧急购电量"]
    if name in ("q3", "q4_3"):
        expected_sheets.append("调整购电量")
    check(set(book.sheetnames) == set(expected_sheets), "unexpected or missing sheets")
    try:
        if name == "q1":
            rows = list(book["计划购电量"].values)
            check(len(rows) == 145, "Q1 purchase sheet must contain 144 intervals")
            for slot in range(144):
                check(rows[slot + 1][0] == _period(slot, slot + 1), "Q1 interval label")
                same(rows[slot + 1][1], log["plan"][slot], "Q1 purchase value")
            rows = list(book["充放电量"].values)
            for block in range(6):
                check(rows[block + 1][0] == _period(24 * block, 24 * (block + 1)), "Q1 storage period")
                same(rows[block + 1][1], log["charge"][block * 24:(block + 1) * 24].sum(), "Q1 charge bin")
                same(rows[block + 1][2], log["discharge"][block * 24:(block + 1) * 24].sum(), "Q1 discharge bin")
            same(rows[1][4], log["soc"][0], "Q1 initial inventory")
            same(rows[2][4], log["soc"][-1], "Q1 final inventory")
        else:
            for title, key in (("计划购电量", "plan"), ("调整购电量", "effective")):
                if title not in book.sheetnames:
                    continue
                rows = list(book[title].values)
                check(len(rows) == 335 and len(rows[0]) == 147, title + " must have 334 dates and 144 interval columns plus two totals")
                check(list(rows[0][1:145]) == [_period(s, s + 1) for s in range(144)], title + " interval labels")
                for j, day in enumerate(range(31, 365), 1):
                    row = rows[j]
                    check(_asdate(row[0]) == _date(day), title + " date order")
                    a, b = day * 144, (day + 1) * 144
                    for slot in range(144):
                        same(row[slot + 1], log[key][a + slot], title + " interval value")
                    same(row[145], log[key][a:b].sum(), title + " daily quantity")
                    cost = log["base_fee"][a:b].sum()
                    if title == "调整购电量":
                        cost += log["up_fee"][a:b].sum() + log["down_fee"][a:b].sum()
                    same(row[146], cost, title + " daily commitment fee")
            rows = list(book["充放电量"].values)[1:]
            check(len(rows) == 334 * 6, "storage sheet must contain six rows per date")
            for j, day in enumerate(range(31, 365)):
                check(_asdate(rows[6 * j][0]) == _date(day), "storage date")
                for block in range(6):
                    row = rows[6 * j + block]
                    a, b = day * 144 + block * 24, day * 144 + (block + 1) * 24
                    check(row[1] == _period(block * 24, (block + 1) * 24), "storage interval")
                    same(row[2], log["charge"][a:b].sum(), "charge bin")
                    same(row[3], log["discharge"][a:b].sum(), "discharge bin")
                same(rows[6 * j][5], log["soc_start"][day * 144], "daily initial inventory")
                same(rows[6 * j + 1][5], log["soc_end"][(day + 1) * 144 - 1], "daily final inventory")
            rows = list(book["紧急购电量"].values)[1:]
            expected = []
            for day in range(31, 365):
                day_events = _events(log["emergency"][day * 144:(day + 1) * 144]) or [("无", 0.0)]
                expected.extend([(_date(day), label, value) for label, value in day_events])
            check(len(rows) == len(expected), "emergency sheet event count")
            carried_date = None
            for row, (date, label, value) in zip(rows, expected):
                if row[0] is not None:
                    carried_date = _asdate(row[0])
                check(carried_date == date and row[1] == label, "emergency date or interval")
                same(row[2], value, "emergency event energy")
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        errors.append("workbook structure: " + str(exc))
    finally:
        book.close()
    check(Path(path).stat().st_size < 20_000_000, "workbook exceeds conservative 20 MB limit")
    return {"passed": not errors, "errors": errors[:30], "error_count": len(errors), "numeric_cells_compared": comparisons}


def verify_all(root=None, require_workbooks=False, strict=True):
    """Validate completed primary runs and return a compact in-memory summary."""
    root = Path(root) if root is not None else ROOT
    data = load_data(root / "data/official-input")
    results, workbooks, errors = {}, {}, []
    for name, filename in NAMES.items():
        base = root / "artifacts/runs" / name
        if not (base / "ledger.npz").exists() or not (base / "summary.json").exists():
            errors.append("missing run: " + name)
            continue
        with np.load(base / "ledger.npz") as archive:
            log = {key: archive[key].copy() for key in archive.files}
        summary = json.loads((base / "summary.json").read_text())
        result = verify_ledger(log, summary, name, 144 if name == "q1" else 365 * 144)
        results[name] = result
        errors.extend([name + ": " + message for message in result["errors"]])
        if name == "q1":
            source_load, source_pv, source_price = data.prior_load / 6, data.prior_pv / 6, data.prior_price
        else:
            source_load, source_pv = data.flat_load / 6, data.flat_pv / 6
            source_price = data.flat_price if summary["configuration"]["variable_price"] else np.tile(data.prior_price, 365)
        for key, expected in (("load", source_load), ("pv", source_pv), ("price", source_price)):
            if log[key].shape != expected.shape or _maxabs(log[key] - expected) > TOL:
                errors.append(name + ": source data / energy conversion mismatch: " + key)
        path = root / "artifacts/exports" / filename
        if path.exists():
            workbooks[name] = verify_workbook(path, name, log)
            errors.extend([filename + ": " + message for message in workbooks[name]["errors"]])
        elif require_workbooks:
            errors.append("missing workbook: " + filename)
    report = {
        "passed": not errors, "errors": errors, "runs_checked": len(results),
        "workbooks_checked": len(workbooks), "workbooks_required": require_workbooks,
        "max_energy_residual_kwh": max([value["max_energy_residual_kwh"] for value in results.values()] or [0]),
        "max_soc_equation_residual_kwh": max([value["max_soc_equation_residual_kwh"] for value in results.values()] or [0]),
        "max_soc_continuity_error_kwh": max([value["max_soc_continuity_error_kwh"] for value in results.values()] or [0]),
        "tolerance_kwh": TOL, "runs": results, "workbooks": workbooks,
    }
    if errors and strict:
        raise VerificationError("{} validation failures".format(len(errors)))
    return report
