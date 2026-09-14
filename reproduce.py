#!/usr/bin/env python3
"""Reproduce the four primary modelling results and their result workbooks."""
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def call(args: list[str]) -> None:
    environment = os.environ.copy()
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        environment[key] = "1"
    subprocess.run([sys.executable, "run.py", *args], cwd=ROOT, env=environment, check=True)


def main() -> None:
    call(["q1"])
    with ThreadPoolExecutor(max_workers=3) as pool:
        list(pool.map(call, [["calibrate", "--family", family] for family in ("naive", "ridge", "boost")]))
    call(["select"])
    with ThreadPoolExecutor(max_workers=3) as pool:
        list(pool.map(call, [["main", "--policy", policy] for policy in ("q2", "q3", "q4_2", "q4_3")]))
    call(["export"])
    call(["verify", "--require-workbooks"])


if __name__ == "__main__":
    main()
