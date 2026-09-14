"""Audit completed experiment ledgers without running any optimization.

Run ``python -m microgrid.verify_experiments`` with ``PYTHONPATH=src``. Missing
experiments are reported as pending; --require-complete fails until all 33 exist.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np

from .data import load_data
from .verify import TOL, verify_ledger

ROOT = Path(__file__).resolve().parents[2]
EXPECTED = ["q2_" + e for e in
            ("no_storage", "no_margin", "q70", "q90", "scenario_reference")]
EXPECTED += [p + "_" + e for p in ("q3", "q4_2", "q4_3")
             for e in ("point", "no_storage")]
EXPECTED += [p + "_" + e for p in ("q3", "q4_3")
             for e in ("issue0", "issue06", "issue012", "issue018", "k6", "k24", "h2", "h6", "roundtrip90", "full_plan_plus_penalty")]
EXPECTED += [p + "_known_price" for p in ("q4_2", "q4_3")]


def verify_experiments(root: str | Path = ROOT, require_complete: bool = False) -> dict:
    root = Path(root)
    data = load_data(root / "data/official-input")
    results, pending, failures = {}, [], []
    maximum = lambda values: float(np.max(np.abs(values))) if np.size(values) else 0.
    for name in EXPECTED:
        folder = root / "artifacts/experiments" / name
        paths = {key: folder / value for key, value in (
            ("ledger", "ledger.npz"), ("summary", "summary.json"),
            ("revisions", "revisions.json.gz"), ("diagnostics", "forecast_diagnostics.json"))}
        if not all(path.exists() for path in paths.values()):
            pending.append(name)
            continue
        try:
            summary = json.loads(paths["summary"].read_text())
            with np.load(paths["ledger"]) as archive:
                log = {key: archive[key].copy() for key in archive.files}
            with gzip.open(paths["revisions"], "rt", encoding="utf-8") as stream:
                revisions = json.load(stream)
            diagnostics = json.loads(paths["diagnostics"].read_text())
        except (EOFError, json.JSONDecodeError):
            # The simulator writes the final diagnostics after the ledger and
            # summary. An experiment that is still closing files is not ready.
            pending.append(name)
            continue
        report = verify_ledger(log, summary, name, 365 * 144)
        errors = report["errors"]
        config = summary["configuration"]
        sources = {"load": data.flat_load / 6, "pv": data.flat_pv / 6,
                   "price": data.flat_price if config["variable_price"] else np.tile(data.prior_price, 365)}
        source_errors = {key: maximum(log[key] - value) for key, value in sources.items()}
        if max(source_errors.values()) > TOL:
            errors.append("ledger values do not match the supplied actual inputs")
        reconstructed = log["plan"].copy()
        allowed = {int(hour) * 6 for hour in config["updates"]}
        previous = -1
        for item in revisions:
            issue, day, slot = item["issue"], item["day"], item["slot"]
            if not (issue == day * 144 + slot and issue > previous and slot in allowed
                    and len(item["q"]) == 144 - slot):
                errors.append("invalid revision timestamp, order, or vector length")
                continue
            reconstructed[issue:(day + 1) * 144] = item["q"]
            previous = issue
        revision_error = maximum(reconstructed - log["effective"])
        if revision_error > TOL:
            errors.append("revision archive does not reconstruct effective commitments")
        if len(revisions) != 365 * len(allowed):
            errors.append("revision archive count does not match all permitted releases")
        inventory_error = 0.
        for label, start, stop in (("whole_year", 0, 365 * 144),
                                   ("evaluation", 31 * 144, 365 * 144),
                                   ("january_validation", 14 * 144, 31 * 144)):
            z = summary[label]
            corrected = log["cost"][start:stop].sum() + float(data.prior_price.min() / .9) * (
                log["soc_start"][start] - log["soc_end"][stop - 1])
            inventory_error = max(inventory_error, abs(corrected - z["inventory_corrected_cost"]))
            if int(np.count_nonzero(log["emergency"][start:stop] > 1e-7)) != z["emergency_intervals"]:
                errors.append(label + " emergency interval count mismatch")
            if abs(log["soc_end"][start:stop].min() - z["soc_min"]) > TOL or abs(log["soc_end"][start:stop].max() - z["soc_max"]) > TOL:
                errors.append(label + " inventory extrema mismatch")
        if inventory_error > 1e-5:
            errors.append("inventory-adjusted cost does not reconcile")
        for engine in diagnostics.values():
            for fit in engine.get("daily_fits", []):
                if fit["training_actual_end_exclusive"] > fit["day"] * 144:
                    errors.append("daily training audit exceeds its information boundary")
        report.update(passed=not errors, input_value_errors=source_errors,
                      max_revision_archive_error_kwh=revision_error,
                      max_inventory_cost_error_yuan=float(inventory_error),
                      revision_count=len(revisions),
                      configuration=config,
                      file_sha256={key: hashlib.sha256(path.read_bytes()).hexdigest()
                                   for key, path in paths.items()})
        results[name] = report
        failures.extend(name + ": " + error for error in errors)
    report = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "expected_experiments": len(EXPECTED), "verified_experiments": len(results),
        "complete": not pending, "passed": not failures and (not require_complete or not pending),
        "pending": pending, "errors": failures,
        "max_energy_residual_kwh": max((z["max_energy_residual_kwh"] for z in results.values()), default=0.),
        "max_soc_equation_residual_kwh": max((z["max_soc_equation_residual_kwh"] for z in results.values()), default=0.),
        "max_soc_continuity_error_kwh": max((z["max_soc_continuity_error_kwh"] for z in results.values()), default=0.),
        "max_fee_error_yuan": max((z["max_fee_error_yuan"] for z in results.values()), default=0.),
        "max_inventory_cost_error_yuan": max((z["max_inventory_cost_error_yuan"] for z in results.values()), default=0.),
        "max_revision_archive_error_kwh": max((z["max_revision_archive_error_kwh"] for z in results.values()), default=0.),
        "tolerance_kwh": TOL, "optimization_calls": 0, "results": results,
    }
    output = root / "artifacts/experiment_verification.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    report = verify_experiments(require_complete=args.require_complete)
    print(json.dumps({k: v for k, v in report.items() if k != "results"}, ensure_ascii=False, indent=2))
    if not report["passed"]:
        raise SystemExit(1)
