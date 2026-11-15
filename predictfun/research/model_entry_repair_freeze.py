"""Freeze only admissible inputs before historical entry-repair execution."""
import ast
import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
import re
import subprocess

ROOT=Path(__file__).resolve().parents[2]
HERE=ROOT/'predictfun/research'
REVIEW=ROOT/'predictfun/reviews/entry_repair_20261115'
DATA=Path('/home/yuqing/ctaNew/data/predict_fun/research')
DAYS=('20261025','20261026','20261027','20261031','20261101','20261102','20261103')


def digest(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def allowed(p,day):
    if day not in DAYS:raise ValueError('undeclared day')
    if Path(p).parent.name!='books':return True
    match=re.fullmatch(r'part_(\d{8})_(\d{2})\.parquet',Path(p).name)
    if not match:raise ValueError('malformed book part')
    t=dt.datetime.strptime(''.join(match.groups()),'%Y%m%d%H');d=dt.datetime.strptime(day,'%Y%m%d')
    return d-dt.timedelta(hours=1)<=t<d+dt.timedelta(days=1) and not dt.datetime(2026,10,28)<=t<dt.datetime(2026,10,31) and t<dt.datetime(2026,11,4)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--name',default='freeze.json');parser.add_argument('--supersedes')
    args=parser.parse_args()
    if Path(args.name).name!=args.name or not args.name.endswith('.json'):raise ValueError('invalid receipt name')
    old=json.loads((ROOT/'predictfun/reviews/entry_representation_20261115/freeze_v2.json').read_text())
    pending=['pf_entry_repair','test_entry_repair','model_entry_repair_runner','model_entry_repair_freeze'];found=set()
    while pending:
        mod=pending.pop()
        if mod in found:continue
        found.add(mod);path=HERE/(mod+'.py')
        for node in ast.walk(ast.parse(path.read_text())):
            ns=[a.name for a in node.names] if isinstance(node,ast.Import) else [node.module] if isinstance(node,ast.ImportFrom) else []
            pending.extend(n for n in ns if n and n.startswith(('pf_','model_','test_entry_repair')) and '.' not in n and n not in found and (HERE/(n+'.py')).exists())
    files={f'predictfun/research/{m}.py':digest(HERE/(m+'.py')) for m in sorted(found)}
    files['predictfun/research/run_light.sh']=digest(HERE/'run_light.sh')
    plan='predictfun/handoffs/20261115_entry_repair_plan.md';files[plan]=digest(ROOT/plan)
    # Relevant old label lineage. Already committed raw path inventory is filtered
    # BEFORE hashing; rejected paths are neither read nor printed into a receipt.
    inputs=dict(old['inputs'])
    base=DATA/'wait_post_v2_20261114_v1/full'
    for n in ('receipt.json','0_post.txt','1_post.txt'):
        p=base/'bootstrap'/n;inputs[str(p)]=digest(p)
    for d in DAYS:
        for sub in (base/('rows_learned_'+d),DATA/'entry_representation_20261115_v2'/('rows_'+d)):
            for n in ('receipt.json','rows.parquet'):
                p=sub/n;inputs[str(p)]=digest(p)
        if d in DAYS[2:]:
            sub=DATA/'entry_representation_20261115_v2'/('fit_'+d)
            for n in ('receipt.json','predictions.parquet',*[f'CONDITION_{s}_{h}.txt' for s in (0,1) for h in ('delta','qpost')]):
                p=sub/n;inputs[str(p)]=digest(p)
    raw={};skipped={}
    for d in DAYS:
        raw[d]={};skipped[d]=0
        for path,h in old['raw_inputs'][d].items():
            if not allowed(path,d):skipped[d]+=1;continue
            if digest(path)!=h:raise ValueError('admissible raw input changed')
            raw[d][path]=h
    for path,h in inputs.items():
        if digest(path)!=h:raise ValueError('label input changed')
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    for path,h in files.items():
        committed=subprocess.check_output(['git','show',f'{commit}:{path}'],cwd=ROOT)
        if hashlib.sha256(committed).hexdigest()!=h:raise ValueError('uncommitted experiment source')
    receipt=dict(asof=dt.datetime.now(dt.timezone.utc).isoformat(),source_commit=commit,files=files,inputs=inputs,raw_inputs=raw,
        rejected_part_counts=skipped,days=DAYS,size_shares=[30,50,100],fixed_cap_shares=1500,new_timing_candidates=2,control_candidates=1,
        size_queue_scenarios=6,model_changes=0,forward_candidates=0,
        null={'draws':500,'match':['day','side','hour','rung','initial_later','action_count','sampling_weight_mass'],'bonferroni':6},
        training='Denser 2/4 weights; sparse control and fixed old evaluation 2/32; same 351 CONDITION features and pinned learner',
        qualifiers=['Inherited retrospective per-day clock','Neutral KEEP and frozen bootstrap continuation','Unknown queue sensitivity, not ground truth','No size-specific entry model fitted'])
    if args.supersedes:
        prior=REVIEW/args.supersedes
        receipt['supersedes']={'receipt':args.supersedes,'sha256':digest(prior),'reason':'Preserve rung as both action key and model feature in dense-row identity check; first smoke failed before writing dense rows or fitting models.'}
    with (REVIEW/args.name).open('x') as f:json.dump(receipt,f,indent=2);f.write('\n')
    print('Frozen',len(files),'source files',len(inputs),'label/model inputs',sum(map(len,raw.values())),'admissible raw inputs; rejected parts',sum(skipped.values()))


if __name__=='__main__':main()
