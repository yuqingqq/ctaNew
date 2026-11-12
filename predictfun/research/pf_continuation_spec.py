"""Declared full-engine continuation study; all historical dates are consumed."""
import datetime as dt
import hashlib
import json
import subprocess
from pathlib import Path
from . import pf_joint_fill_spec as OLD

WORK=OLD.WORK; ROOT=OLD.ROOT; PRIOR=OLD.PRIOR
OUT=ROOT/'data/predict_fun/research/continuation_20261112'
REL=Path('predictfun/reviews/continuation_20261112')
DAYS=OLD.DAYS; TEST=OLD.TEST
HORIZONS=(2000,10000,30000)
BRANCH='research/predict-continuation-20261112'
sha=OLD.sha; native=OLD.native; atomic=OLD.atomic

def selected(gid):
    return int.from_bytes(hashlib.sha256(('continuation-v1:'+gid).encode()).digest()[:4], 'big') % 4 == 0

def write(name,obj):
    text=json.dumps(native(obj),indent=2,allow_nan=False)+'\n'
    for base in (OUT,WORK/REL,ROOT/REL):
        p=base/name;p.parent.mkdir(parents=True,exist_ok=True)
        with p.open('x') as f:f.write(text)

def progress(stage,state,detail):
    now=dt.datetime.now(dt.timezone.utc).isoformat();r=dict(as_of=now,stage=stage,state=state,detail=detail)
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/'progress.jsonl').open('a') as f:f.write(json.dumps(r)+'\n')
    atomic(OUT/'progress.json',json.dumps(r,indent=2)+'\n')
    for base in (WORK,ROOT):
        for rel,begin,end,body in [
            ('predictfun/HANDOFF.md','<!-- CONTINUATION_20261112 -->','<!-- /CONTINUATION_20261112 -->',
             f'## {now} — Full-engine continuation: {state}\n\nStage `{stage}`. {detail}\n\nBranch `{BRANCH}`; receipts `reviews/continuation_20261112/`. Default resting and useful queue retention remain the objective. Live calibration awaits a new capture.'),
            ('predictfun/STATUS.yml','# CONTINUATION_20261112','# /CONTINUATION_20261112',
             f'continuation_20261112:\n  as_of: {json.dumps(now)}\n  state: {json.dumps(state)}\n  stage: {json.dumps(stage)}\n  detail: {json.dumps(detail)}\n  validation: false\n  receipts: reviews/continuation_20261112/')]:
            p=base/rel;s=p.read_text();block=begin+'\n'+body+'\n'+end
            if begin in s:
                i=s.index(begin);j=s.index(end,i)+len(end);s=s[:i]+block+s[j:]
            elif rel.endswith('.md'):
                i=s.index('\n')+1;s=s[:i]+'\n'+block+'\n'+s[i:]
            else:s=s.rstrip()+'\n\n'+block+'\n'
            atomic(p,s)
    print(json.dumps(r),flush=True)

def check():
    d=json.loads((OUT/'declaration.json').read_text())
    for name,h in d['source_sha'].items():
        if sha(WORK/name)!=h:raise ValueError('changed source: '+name)
    return d

def declare():
    prior=json.loads((OLD.OUT/'declaration.json').read_text())
    src=dict(prior['source_sha']);inputs=dict(prior['input_sha'])
    replay=json.loads((PRIOR/'declaration.json').read_text())
    inputs.update(replay['input_sha'])
    for day in DAYS:
        for kind in ('win','fills','requests','generations'):
            for p in (PRIOR/'runs/build'/day/kind).glob('*.parquet'):inputs[str(p)]=sha(p)
    files=list((WORK/'predictfun/research').glob('pf_continuation_*.py'))
    files += [WORK/'predictfun/research'/p for p in ('continuation_driver.py','tests/test_continuation.py','tests/test_simulation_causality.py')]
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=WORK,text=True).strip()
    for p in files:
        rel=str(p.relative_to(WORK))
        if subprocess.check_output(['git','show',commit+':'+rel],cwd=WORK)!=p.read_bytes():raise ValueError('commit source before declaration')
        src[rel]=sha(p)
    for p,h in src.items():
        if sha(WORK/p)!=h:raise ValueError('inherited source changed')
    for p,h in inputs.items():
        if sha(p)!=h:raise ValueError('inherited input changed')
    d=dict(as_of=dt.datetime.now(dt.timezone.utc).isoformat(),source_commit=commit,source_sha=src,input_sha=inputs,days=DAYS,test_days=TEST,validation=False,
        population='All sent neutral default generations retained as statuses. One original causal probe per generation. Deterministic SHA256 sampling selects one quarter of identities independently of fills/outcomes for full-engine paired forks; no resampling after results. All original neutral policy fills/requests/windows must match v4 exactly.',
        target='Fork the full current execution state at the neutral decision, including alias-preserved pending requests/cancellations/fill notices, actual and known inventory/cash, token buckets, both sides and order history. KEEP follows neutral default; PULL cancels the selected side now and waits 1s, then follows the same default current-touch re-entry. Same engine matcher, reservations, hard stops and order sizing in both. Keep-fork fills must exactly equal the neutral realized future for every action.',
        horizons_ms=list(HORIZONS),primary_horizon_ms=10000,
        valuation='At 2/10/30s, sum half-1s/half-5s per-fill markouts over ALL incremental portfolio fills, minus 0.01c change in actual net inventory squared. KEEP-minus-PULL. Common unavoidable prefix is included equally and cancels in the difference. Horizons extending beyond S240 are explicit censored statuses. Missing mark endpoints are explicit statuses, never zero. Also report quantity profiles, first-fill cumulative events and incoming same-side executable tape volume at the decision price. This is local counterfactual utility, not additive portfolio PnL.',
        estimators='One coherent empirical paired-outcome distribution per tree leaf, averaged with positive equal weights and 1% training prior. Pinned LightGBM partition parameters/300 rounds, target 10s value; MARKET and QUEUE unchanged causal features. Reserve the last preceding day for joint 10s-fill-event calibration, with the same earlier-day 50-prior-count procedure. All quantities, value components, time-profile probabilities and flow forecasts share weights. Baseline is the core-training unconditional joint mean. No filled-only fit.',
        evaluation='Six chronological outer dates. Same-core raw/calibrated and market/queue comparisons; quantity and executable-flow squared errors, first-fill Brier, coherent support, action-value errors, horizon increments and local matched pull value. Fixed zero threshold, matched day/side/hour/age/inventory-sign rankings and 500 random shuffles, 1000 day bootstrap draws. No threshold tuning. Full sequential candidate promotion requires a separate frozen replay after this target repair passes.',
        controls='Engine source anchors; cloned alias identity and rekeyed cancellation reservations; full neutral reference parity; all KEEP fork fills reconcile to neutral future; synthetic price-change re-entry, pending cancel/late notice, prefix and longer-horizon fill cases plus intentionally broken forks; all source/input/model hashes and missing/censored status checks.',
        multiplicity=dict(horizons=3,primary_horizons=1,feature_sets=2,calibration_variants=2,local_thresholds=1,sampling_denominator=4,sequential_candidates=0),
        limits='Consumed dates and inherited LODO guard features. Finite 30s horizon does not establish terminal queue value beyond 30s. Incoming flow uses inferred execution tape, not new precise live timestamps. User confirms no new wallet-event capture; live queue/fill calibration remains blocked on that capture. No live code changes, rebate/budget changes or protected-day reads.')
    write('declaration.json',d);progress('declaration','complete','Full-engine paired continuation, timing/flow profiles and coherent estimator declared before numerical builds.')

if __name__=='__main__':declare()
