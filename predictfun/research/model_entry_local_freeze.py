"""Commit-bound design and admissible input identities for the local candidate."""
import ast
import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[2]; HERE=ROOT/'predictfun/research'
DATA=Path('/home/yuqing/ctaNew/data/predict_fun/research')
DAYS=('20261025','20261026','20261027','20261031','20261101','20261102','20261103')


def digest(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--name',default='freeze.json');parser.add_argument('--supersedes');args=parser.parse_args()
    if Path(args.name).name!=args.name: raise ValueError('invalid freeze name')
    pending=['pf_entry_local','pf_entry_local_data','test_entry_local','model_entry_local_runner','model_entry_local_freeze']; seen=set()
    while pending:
        name=pending.pop()
        if name in seen: continue
        seen.add(name)
        for node in ast.walk(ast.parse((HERE/(name+'.py')).read_text())):
            names=[a.name for a in node.names] if isinstance(node,ast.Import) else [node.module] if isinstance(node,ast.ImportFrom) else []
            pending += [n for n in names if n and n.startswith(('pf_','model_','test_entry_local')) and '.' not in n and n not in seen and (HERE/(n+'.py')).exists()]
    paths=[Path('predictfun/research')/(n+'.py') for n in sorted(seen)]
    paths += [Path('predictfun/research/run_light.sh'),Path('predictfun/handoffs/20261115_entry_local_plan.md')]
    files={str(p):digest(ROOT/p) for p in paths}; commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    for p,h in files.items():
        b=subprocess.check_output(['git','show',commit+':'+p],cwd=ROOT)
        if hashlib.sha256(b).hexdigest()!=h: raise ValueError('uncommitted experiment source '+p)
    old=json.loads((ROOT/'predictfun/reviews/entry_repair_20261115/freeze_v2.json').read_text())
    raw={d:dict(old['raw_inputs'][d]) for d in DAYS}; inputs={}
    for d in DAYS:
        for p in list(raw[d]):
            if Path(p).parent.name=='books':
                stamp=Path(p).stem.removeprefix('part_'); date=stamp[:8]
                if date in ('20261028','20261029','20261030') or date>'20261103': raise ValueError('inadmissible inherited file')
        # Add only declared, admissible raw Binance/relay hours; never glob days.
        start=dt.datetime.strptime(d,'%Y%m%d')
        for i in range(-1,24):
            hour=(start+dt.timedelta(hours=i)).strftime('%Y%m%d_%H')
            if hour[:8] in ('20261028','20261029','20261030') or hour[:8]>'20261103': continue
            for p in (Path('/home/yuqing/ctaNew/data/mm_hf/raw/bookTicker/BTCUSDT')/(hour+'.csv.gz'),
                      Path('/home/yuqing/ctaNew/data/predict_reference/runs/20261022T132901Z_8c043d17b6/reference_prices')/(hour+'.jsonl.gz')):
                if p.exists(): raw[d][str(p)]=digest(p)
        for name in ('receipt.json','rows.parquet'):
            p=DATA/'entry_repair_20261115_v2'/('rows_'+d)/name; inputs[str(p)]=digest(p)
        if d in DAYS[2:]:
            for name in (f'fit_{d}_v6.json',f'pred_fold{d}_{d}_v6.parquet'):
                p=DATA/'stop_model_20261114'/name; inputs[str(p)]=digest(p)
    for d,items in raw.items():
        for p,h in items.items():
            if digest(p)!=h: raise ValueError('changed admissible raw input '+p)
    review=ROOT/'predictfun/reviews/entry_local_20261115'; review.mkdir(exist_ok=True)
    r=dict(asof=dt.datetime.now(dt.timezone.utc).isoformat(),source_commit=commit,files=files,inputs=inputs,raw_inputs=raw,
        days=DAYS,learned_candidates=2,forward_candidates=0,primary=dict(fill_horizon_ms=5000,markout_ms=1000,size=30,cap=1500),
        model=dict(leaves=7,rounds=100,lambda_l2=10,explanation='Predeclared compact-model change, same in DIRECT and FACTOR; no per-fold tuning.'),
        null=dict(draws=500,match=['day','side','hour','rung','initial_later','action_count','sampling_mass'],bonferroni=4),
        keeper=dict(version=6,horizon_ms=1000,threshold=.20,entry_gate=False),
        replay_scenarios=[dict(shift=50,queue='zero'),dict(shift=250,queue='zero'),dict(shift=50,queue='block')],
        claim_limit='Mechanistic single-order surrogate. Neutral-KEEP labels; separate frozen keeper only in full replay. No promotion from this consumed-data experiment.')
    if args.supersedes:
        prior=review/args.supersedes
        r['supersedes']=dict(path=args.supersedes,sha256=digest(prior),reason='Initial smoke refused pandas float64 correction into float32 columns before any rows/model; corrected assignment, added keeper/data controls and queue-availability metrics before fitting.')
    with (review/args.name).open('x') as f: json.dump(r,f,indent=2); f.write('\n')
    print('Frozen',len(files),'sources;',len(inputs),'derived inputs;',sum(map(len,raw.values())),'admissible raw files')


if __name__=='__main__': main()
