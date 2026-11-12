"""Frozen coherent paired-outcome model and chronological calibration study."""
import datetime as dt
import json
from pathlib import Path
import subprocess
from . import pf_fill_value_spec as OLD

WORK=OLD.WORK;ROOT=OLD.ROOT;PRIOR=OLD.PRIOR
PREVIOUS=OLD.OUT
OUT=ROOT/'data/predict_fun/research/joint_fill_20261112'
REL=Path('predictfun/reviews/joint_fill_20261112')
DAYS=OLD.DAYS;TEST=OLD.TEST;KEYS=OLD.KEYS;SEED=20261112;EPS=1e-9
BRANCH='research/predict-fill-value-20261112'
SOURCES=['pf_joint_fill_spec.py','pf_joint_fill.py','pf_joint_fill_report.py']
sha=OLD.sha;native=OLD.native;atomic=OLD.atomic


def write(name,obj):
    text=json.dumps(native(obj),indent=2,allow_nan=False)+'\n'
    for base in (OUT,WORK/REL,ROOT/REL):
        p=base/name;p.parent.mkdir(parents=True,exist_ok=True)
        with p.open('x') as f:f.write(text)


def progress(stage,state,detail):
    now=dt.datetime.now(dt.timezone.utc).isoformat();e=dict(as_of=now,stage=stage,state=state,detail=detail)
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/'progress.jsonl').open('a') as f:f.write(json.dumps(e)+'\n');f.flush()
    atomic(OUT/'progress.json',json.dumps(e,indent=2)+'\n')
    for base in (WORK,ROOT):
        for rel,a,b,block in [
            ('predictfun/HANDOFF.md','<!-- JOINT_FILL_20261112 -->','<!-- /JOINT_FILL_20261112 -->',
             f'## {now} — Coherent fill/value model: {state}\n\nStage `{stage}`. {detail}\n\nDefault resting / useful queue retention remains the objective. Receipts: `reviews/joint_fill_20261112/`; branch `{BRANCH}`. No live changes.'),
            ('predictfun/STATUS.yml','# JOINT_FILL_20261112','# /JOINT_FILL_20261112',
             f'joint_fill_20261112:\n  as_of: {json.dumps(now)}\n  state: {json.dumps(state)}\n  stage: {json.dumps(stage)}\n  detail: {json.dumps(detail)}\n  receipts: reviews/joint_fill_20261112/\n  validation: false')]:
            p=base/rel;s=p.read_text();text=a+'\n'+block+'\n'+b
            if a in s:
                x=s.index(a);y=s.index(b,x)+len(b);s=s[:x]+text+s[y:]
            elif rel.endswith('.md'):
                x=s.index('\n')+1;s=s[:x]+'\n'+text+'\n'+s[x:]
            else:s=s.rstrip()+'\n\n'+text+'\n'
            atomic(p,s)
    print(json.dumps(e),flush=True)


def declare():
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=WORK,text=True).strip()
    prior=json.loads((PREVIOUS/'declaration.json').read_text());src=dict(prior['source_sha']);inputs=dict(prior['input_sha'])
    for name in SOURCES:
        p='predictfun/research/'+name
        if subprocess.check_output(['git','show',commit+':'+p],cwd=WORK)!=(WORK/p).read_bytes():raise ValueError('commit sources first')
        src[p]=sha(WORK/p)
    for name in ('declaration.json','fit.json','predictions.parquet','result.json','audit.json'):
        p=PREVIOUS/name;inputs[str(p)]=sha(p)
    for p,h in src.items():
        if sha(WORK/p)!=h:raise ValueError('changed source')
    for p,h in inputs.items():
        if sha(p)!=h:raise ValueError('changed input')
    d=dict(as_of=dt.datetime.now(dt.timezone.utc).isoformat(),source_commit=commit,source_sha=src,input_sha=inputs,
        days=DAYS,test_days=TEST,validation=False,
        design='One shared empirical distribution of paired KEEP/RESET outcomes per action. Train a scalar LGBM partition model on original KEEP-minus-RESET label using pinned parameters/300 rounds; average the empirical paired-outcome distributions in its terminal leaves with equal positive tree weights. Use the same weights for every quantity, probability and value component of BOTH actions, then mix 1% core-training prior. Boosted raw regression scores are not used as moment forecasts.',
        targets='Joint event class = 1[KEEP fills] + 2[RESET fills]. For each branch retain filled fraction, squared fraction, positive aggregate markout normalized by remaining*maximum favorable per-share move, and negative markout normalized by remaining*maximum adverse move. All eligible neutral actions including zero fills train the partition and empirical joint moments. No outcome-selected training; no arbitrary queue bonus.',
        chronology='For each original outer test day, reserve the last preceding consumed complete day for calibration; partition model and leaf distributions use only earlier core days. Do not refit on calibration day. Raw and calibrated predictions therefore share exactly the same core model. Original direct/component controls used one additional prior training day; architecture comparison with them is not equal-data attribution.',
        calibration='Fit positive joint-event class multipliers on the held-out earlier calibration day to match class frequencies shrunk by 50 core-prior pseudocounts. Newton solution of convex intercept-only multinomial log loss, fixed algorithm and tolerance. Reweight the complete joint outcome distribution by event class, including ALL moments and BOTH branches. No independently calibrated quantity or value. If calibration contains an event absent from core support, refuse calibration and retain raw weights with an explicit status; no dropped actions.',
        features='Original MARKET versus original FEATURES, unchanged. Hidden queue states, labels and future endpoints never used as inputs.',
        evaluation='Exact original 40059 test action identities. Branch probability Brier/log loss/AUC, quantity errors and local KEEP-minus-RESET value errors/calibration. Compare coherent raw/calibrated variants, previous independent components and original direct model. All probabilities/moments must satisfy support before results are accepted. Every saved partition model, leaf distribution and calibrator reconstructs predictions.',
        selection='Four fixed zero-threshold local policies (market/queue x raw/calibrated). Same-count rankings from original queue/market and guard margin; 500 matched-count shuffles per policy. Match day/side/hour/age/inventory sign. Day-unit 1000-draw intervals only; all outcomes and statuses preserved. No portfolio-PnL interpretation of overlapping local values.',
        null=dict(draws=500,seed=20261111,keys=KEYS),
        multiplicity=dict(partition_models=12,calibrators=12,feature_sets=2,local_policies=4,thresholds=1,new_sequential_strategies=0),
        controls='Exact reconstruction of valid paired outcomes; convex-mixture and joint-calibration support; planted zero-quantity positive-value, impossible moments and invalid weights refused. Chronology and future-feature refusals. Calibration positive target matching and unseen-event refusal. Model/leaf/calibrator save-reload identity, corrupt forecast refusal and all source/input hashes.',
        limits='Same short-horizon simulator labels and same-price reset assumption retained. This repairs internal support and tests calibration, not fill-time, incoming-volume, later queue continuation, live execution calibration, or sequential strategy profitability. Consumed days and inherited LODO guard caveat remain.')
    write('declaration.json',d);progress('declaration','complete','Frozen coherent paired-outcome model and earlier-day calibration; no threshold tuning.')


def check():
    d=json.loads((OUT/'declaration.json').read_text())
    for p,h in d['source_sha'].items():
        if sha(WORK/p)!=h:raise ValueError('source changed: '+p)
    for p,h in d['input_sha'].items():
        if sha(p)!=h:raise ValueError('input changed: '+p)
    return d


if __name__=='__main__':declare()
