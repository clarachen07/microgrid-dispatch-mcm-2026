#!/usr/bin/env python3
"""Reproduce calculations: .venv/bin/python run.py --help."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(key,'1')
from pathlib import Path
import argparse
import json
import sys
from dataclasses import replace
sys.path.insert(0,str(Path(__file__).resolve().parent/'src'))
from microgrid.data import load_data
from microgrid.simulate import Policy, base_policies, simulate, solve_question_one, perfect_information_bound, write_json
from microgrid.quantile import simulate_quantile_q2


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('stage',choices=['q1','calibrate','select','main','experiment','bound','export','verify','all'])
    ap.add_argument('--policy',default='all',choices=['all','q2','q3','q4_2','q4_3'])
    ap.add_argument('--family',default='all',choices=['all','naive','ridge','boost'])
    ap.add_argument('--days',type=int,default=365)
    ap.add_argument('--experiment',default='point',choices=['point','no_storage','no_margin','q70','q90','scenario_reference','issue0','issue06','issue012','issue018','k6','k24','h2','h6','roundtrip90','full_plan_plus_penalty','known_price'])
    ap.add_argument('--quantile',type=float,choices=[.7,.8,.9],default=.8,
                    help='Q2 primary rolling residual quantile (default: 0.80)')
    ap.add_argument('--out',default='artifacts/runs')
    ap.add_argument('--require-workbooks',action='store_true')
    ap.add_argument('--paper-only',action='store_true',help='refresh paper tables and figures while retaining result workbooks and ledgers')
    ap.add_argument('--results-only',action='store_true',help='export only the five required result workbooks')
    args=ap.parse_args()
    if args.paper_only and args.stage!='export':ap.error('--paper-only is only available with export')
    if args.results_only and args.stage!='export':ap.error('--results-only is only available with export')
    if args.paper_only and args.results_only:ap.error('--paper-only and --results-only cannot be combined')
    if args.stage=='all':
        if args.out!='artifacts/runs' or args.days!=365 or args.policy!='all':
            ap.error('all requires the default output directory, all policies, and 365 days')
        import subprocess
        subprocess.run([sys.executable,str(Path(__file__).resolve().parent/'reproduce.py')],check=True)
        return
    data=load_data()
    selection_file=Path('artifacts/selection.json')
    selected=json.loads(selection_file.read_text())['selected'] if selection_file.exists() else {}
    if args.stage in ('q1','all'):
        print(solve_question_one(data,Path(args.out)/'q1'),flush=True)
    if args.stage=='calibrate':
        families=['naive','ridge','boost'] if args.family=='all' else [args.family]
        for family in families:
            for policy in base_policies():
                if args.policy!='all' and policy.name!=args.policy: continue
                p=replace(policy,name=policy.name+'_'+family,family=family,warmup_naive=False)
                simulate(p,data,end_day=31,output_dir='artifacts/calibration')
    if args.stage=='select':
        scores={}; chosen={}
        for policy in base_policies():
            vals={}
            for family in ('naive','ridge','boost'):
                file=Path('artifacts/calibration')/(policy.name+'_'+family)/'summary.json'
                vals[family]=json.loads(file.read_text())['january_validation']['inventory_corrected_cost']
            minimum=min(vals.values())
            choice=next(f for f in ('naive','ridge','boost') if vals[f]<=minimum*1.005)
            scores[policy.name]=vals; chosen[policy.name]=choice
        write_json(selection_file,{'selected':chosen,'validation_inventory_cost':scores,
                                  'rule':'Jan15-31 chronological cost, choose simplest within 0.5%; operational January remains naive.'})
        print(chosen)
    if args.stage in ('main','all'):
        if not selected: raise RuntimeError('Run calibration and selection first.')
        for policy in base_policies(selected):
            if args.policy!='all' and policy.name!=args.policy: continue
            if policy.name=='q2':
                simulate_quantile_q2(args.quantile,data,end_day=args.days,
                                     output_dir=args.out,name='q2')
            else:
                simulate(policy,data,end_day=args.days,output_dir=args.out)
    if args.stage=='experiment':
        names=[args.policy] if args.policy!='all' else ['q2','q3','q4_2','q4_3']
        for policy in base_policies(selected):
            if policy.name not in names: continue
            name=policy.name+'_'+args.experiment
            if policy.name=='q2':
                if args.experiment in ('q70','q90','no_margin','no_storage'):
                    q={'q70':.7,'q90':.9}.get(args.experiment,.8)
                    simulate_quantile_q2(q,data,end_day=args.days,
                                         output_dir='artifacts/experiments',name=name,
                                         use_margin=args.experiment!='no_margin',
                                         storage=args.experiment!='no_storage')
                    continue
                if args.experiment=='point':
                    simulate_quantile_q2(.8,data,end_day=args.days,
                                         output_dir='artifacts/experiments',name=name,
                                         use_margin=False)
                    continue
                if args.experiment=='scenario_reference':
                    simulate(replace(policy,name=name),data,end_day=args.days,
                             output_dir='artifacts/experiments')
                    continue
                raise ValueError('Q2 experiment is not applicable to the primary quantile model')
            changes={'point':{'scenarios':1}, 'no_storage':{'storage':False},
                     'issue0':{'updates':()}, 'issue06':{'updates':(6,)},
                     'issue012':{'updates':(12,)}, 'issue018':{'updates':(18,)},
                     'k6':{'scenarios':6},'k24':{'scenarios':24},
                     'h2':{'control_hours':2},'h6':{'control_hours':6},
                     'roundtrip90':{'eta':.9**.5},
                     'full_plan_plus_penalty':{'settlement':'full_plan_plus_penalty'},
                     'known_price':{'perfect_price':True}}
            p=replace(policy,name=name,**changes[args.experiment])
            simulate(p,data,end_day=args.days,output_dir='artifacts/experiments')
    if args.stage=='bound':
        print(perfect_information_bound(data,Path('artifacts/perfect_bound.json')),flush=True)
    if args.stage in ('export','all'):
        from microgrid.export import export_all
        export_all(paper_only=args.paper_only,results_only=args.results_only)
    if args.stage in ('verify','all'):
        from microgrid.verify import verify_all
        print(verify_all(require_workbooks=args.require_workbooks),flush=True)


if __name__=='__main__':
    main()
