#!/usr/bin/env python3
"""Reproduce calculations, experiments, checks, and both final deliverables."""
from pathlib import Path
import os
import subprocess
import sys
import argparse
from concurrent.futures import ThreadPoolExecutor

ROOT=Path(__file__).resolve().parent
EXPERIMENTS=[('q2',e) for e in ('no_storage','no_margin','q70','q90','scenario_reference')]
EXPERIMENTS += [(p,e) for p in ('q3','q4_2','q4_3') for e in ('point','no_storage')]
EXPERIMENTS += [(p,e) for p in ('q3','q4_3') for e in ('issue0','issue06','issue012','issue018','k6','k24','h2','h6','roundtrip90','full_plan_plus_penalty')]
EXPERIMENTS += [(p,'known_price') for p in ('q4_2','q4_3')]


def call(args):
    env=os.environ.copy()
    for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
        env[key]='1'
    subprocess.run([sys.executable,'run.py']+args,cwd=ROOT,env=env,check=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=['all','experiments','package'],default='all')
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--policy',action='append',choices=['q2','q3','q4_2','q4_3'],
                        help='limit the experiments stage; may be supplied more than once')
    args=parser.parse_args()
    if args.stage=='package':
        subprocess.run([sys.executable,'package.py'],cwd=ROOT,check=True)
        return
    if args.stage=='experiments':
        jobs=[]
        for p,e in EXPERIMENTS:
            if args.policy and p not in args.policy:continue
            ready=ROOT/'artifacts/experiments'/(p+'_'+e)/'summary.json'
            if args.resume and ready.exists():continue
            jobs.append(['experiment','--policy',p,'--experiment',e])
        with ThreadPoolExecutor(max_workers=3) as pool:
            list(pool.map(call,jobs))
        return
    call(['q1'])
    with ThreadPoolExecutor(max_workers=3) as pool:
        list(pool.map(call,[['calibrate','--family',f] for f in ('naive','ridge','boost')]))
    call(['select'])
    with ThreadPoolExecutor(max_workers=3) as pool:
        list(pool.map(call,[['main','--policy',p] for p in ('q2','q3','q4_2','q4_3')]))
    with ThreadPoolExecutor(max_workers=3) as pool:
        list(pool.map(call,[['experiment','--policy',p,'--experiment',e] for p,e in EXPERIMENTS]))
    call(['bound']);call(['verify'])
    environment=os.environ.copy();environment['PYTHONPATH']=str(ROOT/'src')
    subprocess.run([sys.executable,'-m','microgrid.verify_experiments','--require-complete'],
                   cwd=ROOT,env=environment,check=True)
    call(['export']);call(['verify','--require-workbooks'])
    subprocess.run([sys.executable,'package.py'],cwd=ROOT,check=True)


if __name__=='__main__':
    main()
