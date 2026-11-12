"""Declared first-stage fill/value diagnostic; saved neutral actions only."""
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile

WORK = Path(__file__).resolve().parents[2]
ROOT = Path('/home/yuqing/ctaNew')
PRIOR = ROOT / 'data/predict_fun/research/default_pull_20261111_v4'
OUT = ROOT / 'data/predict_fun/research/fill_value_20261112'
REL = Path('predictfun/reviews/fill_value_20261112')
BRANCH = 'research/predict-fill-value-20261112'
DAYS = ['20261023','20261024','20261025','20261026','20261027','20261031','20261101','20261102','20261103']
TEST = DAYS[3:]
HEADS = ['fill', 'fraction', 'fraction2', 'gain', 'loss']
SOURCES = ['pf_fill_value_spec.py','pf_fill_value.py','pf_fill_value_report.py']
KEYS = ['day','side','hour','age_bin','inv_sign']
SEED = 20261112
EPS = 1e-9


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024), b''): h.update(b)
    return h.hexdigest()


def native(v):
    if isinstance(v, dict): return {str(k):native(x) for k,x in v.items()}
    if isinstance(v, (list,tuple)): return [native(x) for x in v]
    if hasattr(v, 'item'): return native(v.item())
    return v


def write(name, obj, copies=True):
    text = json.dumps(native(obj),indent=2,allow_nan=False)+'\n'
    for base in ([OUT,WORK/REL,ROOT/REL] if copies else [OUT]):
        p=base/name; p.parent.mkdir(parents=True,exist_ok=True)
        with p.open('x') as f: f.write(text)


def atomic(p,text):
    p.parent.mkdir(parents=True,exist_ok=True)
    fd,name=tempfile.mkstemp(dir=p.parent,prefix='.'+p.name+'.')
    with os.fdopen(fd,'w') as f: f.write(text);f.flush();os.fsync(f.fileno())
    os.replace(name,p)


def progress(stage,state,detail):
    now=dt.datetime.now(dt.timezone.utc).isoformat()
    event=dict(as_of=now,stage=stage,state=state,detail=detail,branch=BRANCH)
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/'progress.jsonl').open('a') as f: f.write(json.dumps(event)+'\n');f.flush()
    atomic(OUT/'progress.json',json.dumps(event,indent=2)+'\n')
    for base in (WORK,ROOT):
        p=base/'predictfun/HANDOFF.md';s=p.read_text();a='<!-- FILL_VALUE_20261112 -->';b='<!-- /FILL_VALUE_20261112 -->'
        block=f'{a}\n\n## {now} — Fill opportunity / value diagnostic: {state}\n\nStage `{stage}`. {detail}\n\nDefault quoting and useful queue retention remain the objective. Same saved short-horizon neutral labels; no entry-gate restoration or live changes. Source/declaration and receipts: `reviews/fill_value_20261112/`; branch `{BRANCH}`.\n\n{b}'
        if a in s:
            x=s.index(a);y=s.index(b,x)+len(b);s=s[:x]+block+s[y:]
        else:
            x=s.index('\n')+1;s=s[:x]+'\n'+block+'\n'+s[x:]
        atomic(p,s)
        p=base/'predictfun/STATUS.yml';s=p.read_text();a='# FILL_VALUE_20261112';b='# /FILL_VALUE_20261112'
        block=f'{a}\nfill_value_20261112:\n  as_of: {json.dumps(now)}\n  state: {json.dumps(state)}\n  stage: {json.dumps(stage)}\n  detail: {json.dumps(detail)}\n  branch: {BRANCH}\n  receipts: reviews/fill_value_20261112/\n  validation: false\n{b}'
        if a in s:
            x=s.index(a);y=s.index(b,x)+len(b);s=s[:x]+block+s[y:]
        else:s=s.rstrip()+'\n\n'+block+'\n'
        atomic(p,s)
    print(json.dumps(event),flush=True)


def declare():
    if (OUT/'declaration.json').exists(): raise ValueError('declaration exists; supersede')
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=WORK,text=True).strip()
    old=json.loads((PRIOR/'declaration.json').read_text());src=dict(old['source_sha'])
    for name in SOURCES:
        p=Path('predictfun/research')/name
        if subprocess.check_output(['git','show',commit+':'+str(p)],cwd=WORK)!=(WORK/p).read_bytes():
            raise ValueError('source must be committed before declaration')
        src[str(p)]=sha(WORK/p)
    for p,h in src.items():
        if sha(WORK/p)!=h: raise ValueError('source identity: '+p)
    inputs={}
    for name in ['declaration.json','fit.json','predictions.parquet','comparison_final.json','completion_audit.json','completion_runtime_v4.json']:
        p=PRIOR/name;inputs[str(p)]=sha(p)
    for day in DAYS:
        p=PRIOR/'runs/build'/day;check=json.loads((p/'checks.json').read_text())
        if check['status']!='PASS' or not check['reference_exact']:raise ValueError('neutral path not verified')
        h=next(v['generation_sha'] for v in check['variants'] if v['mode']=='default')
        if sha(p/'generations/default.parquet')!=h:raise ValueError('generation identity')
        for name in ['checks.json','feature_check.json','generations/default.parquet']:
            inputs[str(p/name)]=sha(p/name)
    fit=json.loads((PRIOR/'fit.json').read_text())
    for fold in fit['folds']:
        for model,h in fold['models_sha'].items():
            p=PRIOR/'models'/f"{fold['day']}_{model}.txt"
            if sha(p)!=h:raise ValueError('old model identity')
            inputs[str(p)]=h
    d=dict(as_of=dt.datetime.now(dt.timezone.utc).isoformat(),source_commit=commit,source_sha=src,input_sha=inputs,
        days=DAYS,test_days=TEST,validation=False,
        scope='First-stage diagnostic of fill probability/quantity and component value on the existing neutral short-horizon labels. No raw data, new portfolio replay, entry gate or live change. Does not measure later queue option value, arrival times, or calibrate fills against a live capture.',
        population='All existing eligible finite-label neutral actions, one assigned probe per generation. All original statuses retained. Same chronology and exact test population as original direct model. No filled-only training or class rebalancing.',
        targets='For each KEEP and RESET branch, train fill indicator, filled fraction of client-known remaining, squared filled fraction, positive part and negative part of aggregate average 1s/5s markout. Every head trains on every action including zeros. Positive/negative branch net markout is not a per-fill gross gain/loss decomposition.',
        features='Original MARKET and original FEATURES (MARKET plus four observed queue proxies), unchanged. No hidden queue, future label or future-horizon endpoint features.',
        fitting='Two feature sets x two branches x five heads x six expanding folds = 120 models, 300 rounds each. Original pinned LGBM parameters; only fill indicator uses binary objective because its target is a probability. No tuning or early stopping. Binary bounds [1e-6,1-1e-6]; regression moments constrained by probability/quantity support, stated below.',
        constraints='p clipped to [1e-6,1-1e-6]; fraction clipped to [0,p]; fraction2 clipped to [fraction^2/p,fraction]. Gain/loss predictions clipped nonnegative. Save all raw predictions, constraint counts, and constrained outputs.',
        value='Branch expected markout = predicted gain minus predicted loss. Expected inventory penalty = .01 * (2*inventory*side_sign*remaining*E[fraction] + remaining^2*E[fraction2]). KEEP-minus-RESET is the difference of branch values. Fill probability is a diagnostic and moment support constraint; markout heads learn unconditional joint value directly. No independent probability-times-unconditional-edge multiplication, no extra queue bonus, no rebate or cost setting change.',
        baseline='Training-only shrunk means in side x fixed age bucket x remaining bucket x displayed queue-depth bucket, shrink strength 50 toward whole training mean. Same branch targets and output support rules. Original direct-value market/queue predictions and guard margin remain controls; model feature comparison isolates added queue proxies.',
        evaluation='Per branch Brier/log loss/AUC, quantity MAE/RMSE/bias, positive/negative markout and net-value errors, fixed probability/value calibration bins, exact zero-fill and censored/excluded counts. Differences versus training-only baseline and market feature control are paired within day; 1000 day-unit bootstrap draws. Component value decisions evaluated at fixed zero threshold, matched old queue/guard rankings, and matched random pulls. These are local overlapping utility sums, not portfolio P&L.',
        null=dict(draws=500,seed=20261111,keys=KEYS,design='Shuffle component-model pull indicators within day/side/hour/age/inventory sign, preserving each stratum count, evaluate RESET-minus-KEEP label.'),
        metrics_seed=SEED,age_bins=[0,100,250,500,1000,2000,5000,None],remaining_bins=[0,1,3,5,None],depth_bins=[-1,0,10,50,200,1000,None],
        multiplicity=dict(feature_sets=2,branches=2,heads=5,folds=6,models=120,diagnostic_policies=2,thresholds_per_policy=1,new_sequential_strategy_candidates=0),
        controls='Perfect and reversed probability/quantity/value controls; out-of-range probabilities and duplicate/invalid/overfilled actions refused; chronology and feature-leakage refusals; branch/risk identities; reloaded saved models reconstruct every test prediction including old direct models; one-unit corrupt score refusal; fixed population identities and old score parity.',
        limits='Simulation-conditional, consumed dates; inherited guard values use LODO. Existing ~2s fill horizon, same-price 1s RESET and primary latency/queue assumptions remain. Fill timing/raw incoming volume are not predicted from these saved labels. Component forecasts may improve interpretability without improving decisions; promotion needs a consistent sequential replay and later untouched validation.')
    write('declaration.json',d)
    progress('declaration','complete','Frozen explicit fill and value diagnostic; 120 models across six chronological folds, no policy tuning.')


def check():
    d=json.loads((OUT/'declaration.json').read_text())
    for p,h in d['source_sha'].items():
        if sha(WORK/p)!=h:raise ValueError('source changed: '+p)
    for p,h in d['input_sha'].items():
        if sha(p)!=h:raise ValueError('input changed: '+p)
    return d


if __name__=='__main__':declare()
