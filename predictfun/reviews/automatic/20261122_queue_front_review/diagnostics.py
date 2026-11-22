"""Saved-output and resource cross-checks for the pinned review; no replay."""
from pathlib import Path
import gzip, hashlib, json, re, subprocess
import pandas as pd

ROOT=Path('/home/yuqing/ctaNew'); PIN=Path('/tmp/pf-qf-review-c7404f21')
OUT=Path(__file__).resolve().parent; R=PIN/'predictfun/reviews/ladder_queue_front_20261122'
sha=lambda b:hashlib.sha256(b).hexdigest()
read=lambda n:pd.read_csv(R/'outputs'/f'{n}.csv.gz',float_precision='round_trip',dtype={'day':str})
K=read('classes'); A=read('arms'); gm=json.loads((R/'manifest.json').read_text())
rows=[]
for (arm,cls),g in K.groupby(['arm','cls']):
    v=g.select_dtypes('number').sum()
    rows.append(dict(arm=arm,cls=cls,all_fills=int(v.fills),all_q=v.q,end=v.end,
                     eligible_1s_q=v.el_q,mo1s=v.mo1s,mo5s=v.mo5s,eligible_5s_q=v.el5_q,censored_5s=int(v.cens5),
                     mo30s=v.mo30s,eligible_30s_q=v.el30_q,censored_30s=int(v.cens30)))
pd.DataFrame(rows).to_csv(OUT/'class_horizon_denominators.csv',index=False)

# The claims omitted by the frozen checker were verified independently in verify.py.
# Exercise the actual comparison's positive control and three bad-value refusals.
def eq(a,b):
    assert abs(a-b)<=1e-6*max(1,abs(a),abs(b))
eq(K.iloc[0].mo30s,K.iloc[0].mo30s)
refusals={}
for name,a,b in [('class_30s',K.iloc[0].mo30s,K.iloc[0].mo30s+99999),
                 ('added_end',read('lost_added').iloc[0].added_end,read('lost_added').iloc[0].added_end+99999),
                 ('day_sign',1.,0.)]:
    try: eq(a,b); refusals[name]=False
    except AssertionError:refusals[name]=True
assert all(refusals.values())

RL=re.compile(r'^(\S+) tag=(\S+) rc=(\d+) peak_rss_mib=(\d+) wall_s=([\d.]+) cpu_user_s=([\d.]+)')
logs=[p for n in gm['line_receipts'] for p in (PIN/'predictfun/reviews'/n/'logs').rglob('*.log')]
logs += [p for n in gm['line_reviews'] for p in (PIN/'predictfun/reviews/automatic'/n).glob('*.log')]
logs += list((R/'logs').rglob('*.log'))
seen={};duplicates=[]
for p in logs:
    for l in p.read_text(errors='replace').splitlines():
        m=RL.match(l)
        if m:
            key=(m[1],m[2]); val=float(m[6])
            if key in seen:duplicates.append([*key,str(p)])
            else:seen[key]=val
frozen_log_seconds=sum(seen.values())
# Review and prior tiny planning verification jobs not included by the frozen lists.
extras=[]
for l in (ROOT/'predictfun/research/run_light.log').read_text().splitlines():
    m=RL.match(l)
    if not m:continue
    key=(m[1],m[2])
    if key in seen:continue
    if m[2].startswith(('codex_qf_review_', 'codex_async_execution_', 'codex_queue_latency_', 'codex_queue_sweep_')):
        seen[key]=float(m[6]);extras.append(dict(time=m[1],tag=m[2],cpu_s=float(m[6])))
used=sum(seen.values())/3600
assert used<2.25
projection=json.loads((R/'logs/projection.json').read_text())
own=projection['batch_spent']; remainder=(projection['projected_rest_cpu_s']+projection['tail_s'])/3600
eq(own+remainder,projection['batch_projected'])
eq(projection['line_now']+remainder+.03,projection['line_projected'])
assert projection['run_rest'] and projection['batch_projected']<=.25 and projection['line_projected']<=2.25
for d in gm['days']:
    assert gm['parts']['pilot'][d]==gm['complete'][d][:2]
    assert gm['parts']['rest'][d]==gm['complete'][d][2:]

# Prove the supersession changes only instrument plumbing, not any family.
import ast
old=ast.parse((R/'fq_checks.py').read_text());new=ast.parse((R/'fq_checks_v2.py').read_text())
defs=lambda tree:{n.name:ast.dump(n,include_attributes=False) for n in tree.body if isinstance(n,ast.FunctionDef)}
oa,na=defs(old),defs(new);changed=[k for k in oa if oa[k]!=na[k]]
assert set(changed)=={'_edit','main'},changed
assert all((PIN/'predictfun/research'/n).read_bytes()==subprocess.check_output(['git','-C',str(PIN),'show','b4ecbc34:predictfun/research/'+n]) for n in gm['sources'])

report=dict(target='c7404f212700ebf1c6977e1806a28c23b5961f02',class_horizons=rows,
            extra_check_refusals=refusals,checker_v2_changed_functions=changed,
            frozen_source_closure_unchanged=True,projection_recomputed=True,
            frozen_logged_core_hours=frozen_log_seconds/3600,duplicate_log_receipts=duplicates,
            additional_jobs=extras,line_core_hours_before_this_check=used,
            note='This job and final commit verification are added from their run_light log lines in the report.')
(OUT/'diagnostics.json').write_text(json.dumps(report,indent=2,default=str)+'\n')
print(json.dumps(report,indent=2,default=str))
