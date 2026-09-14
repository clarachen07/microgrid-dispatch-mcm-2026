#!/usr/bin/env python3
"""Run the four microgrid-dispatch modelling questions."""
import argparse
import json
import os
import sys
from dataclasses import replace
from pathlib import Path

for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(key, "1")

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from microgrid.data import load_data
from microgrid.quantile import simulate_quantile_q2
from microgrid.simulate import base_policies, simulate, solve_question_one, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["q1", "calibrate", "select", "main", "export", "verify", "all"])
    parser.add_argument("--policy", default="all", choices=["all", "q2", "q3", "q4_2", "q4_3"])
    parser.add_argument("--family", default="all", choices=["all", "naive", "ridge", "boost"])
    parser.add_argument("--days", type=int, default=365)
    parser.add_argument("--quantile", type=float, choices=[0.7, 0.8, 0.9], default=0.8)
    parser.add_argument("--out", default="artifacts/runs")
    parser.add_argument("--require-workbooks", action="store_true")
    args = parser.parse_args()

    if args.stage == "all":
        if args.out != "artifacts/runs" or args.days != 365 or args.policy != "all":
            parser.error("all requires the default output directory, all policies, and 365 days")
        import subprocess
        subprocess.run([sys.executable, str(ROOT / "reproduce.py")], check=True)
        return

    data = load_data()
    selection_file = ROOT / "artifacts/selection.json"
    selected = json.loads(selection_file.read_text())["selected"] if selection_file.exists() else {}

    if args.stage == "q1":
        print(solve_question_one(data, Path(args.out) / "q1"), flush=True)
    elif args.stage == "calibrate":
        families = ["naive", "ridge", "boost"] if args.family == "all" else [args.family]
        for family in families:
            for policy in base_policies():
                if args.policy != "all" and policy.name != args.policy:
                    continue
                candidate = replace(policy, name=f"{policy.name}_{family}", family=family, warmup_naive=False)
                simulate(candidate, data, end_day=31, output_dir="artifacts/calibration")
    elif args.stage == "select":
        scores, chosen = {}, {}
        for policy in base_policies():
            values = {
                family: json.loads((ROOT / "artifacts/calibration" / f"{policy.name}_{family}" / "summary.json").read_text())["january_validation"]["inventory_corrected_cost"]
                for family in ("naive", "ridge", "boost")
            }
            minimum = min(values.values())
            scores[policy.name] = values
            chosen[policy.name] = next(family for family in ("naive", "ridge", "boost") if values[family] <= minimum * 1.005)
        write_json(selection_file, {"selected": chosen, "validation_inventory_cost": scores,
                                    "rule": "Jan15-31 chronological cost; choose the simplest method within 0.5%."})
        print(chosen, flush=True)
    elif args.stage == "main":
        if not selected:
            raise RuntimeError("Run calibration and selection first.")
        for policy in base_policies(selected):
            if args.policy != "all" and policy.name != args.policy:
                continue
            if policy.name == "q2":
                simulate_quantile_q2(args.quantile, data, end_day=args.days, output_dir=args.out, name="q2")
            else:
                simulate(policy, data, end_day=args.days, output_dir=args.out)
    elif args.stage == "export":
        from microgrid.export import export_all
        export_all()
    elif args.stage == "verify":
        from microgrid.verify import verify_all
        print(verify_all(require_workbooks=args.require_workbooks), flush=True)


if __name__ == "__main__":
    main()
