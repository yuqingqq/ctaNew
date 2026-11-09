"""Declared follow-up: split only when growing an existing small order.

The six-arm screen showed extra request cost for unconditional splitting. This
one additional mechanism is an engineering follow-up on the same consumed trace,
not independent strategy validation. Exact old-arm artifact parity is required.
"""
import copy
import datetime as dt
import json
import subprocess
import sys
import pandas as pd
from . import pf_quote_capture_study as s
from .pf_quote_capture_verify import verify_sums

OLD=s.DEST
DEST=s.a.ROOT/'data/predict_fun/research/quote_incremental_20261109'
REVIEW=s.a.HERE.parent/'reviews/quote_incremental_20261109'
ARMS={'single':(False,0.,False),'layered':(True,0.,False),'incremental':(True,0.,True)}
SOURCES=s.SOURCE+['pf_quote_incremental_study.py','pf_quote_capture_verify.py']


def declare():
    if DEST.exists():raise ValueError('follow-up already exists')
    (DEST/'logs').mkdir(parents=True)
    previous=json.loads((OLD/'declaration_v2.json').read_text())
    d=dict(previous,version=3,as_of=dt.datetime.now(dt.timezone.utc).isoformat(),
        primary_arm='incremental',arms=ARMS,multiplicity=7,
        primary='one additional post-screen mechanism: start with a whole target-sized order; only split an existing same-price small base on size growth',
        followup_to_receipt_sha256=s.a.sha(OLD/'receipt_v2.json'),
        motivation_known_before_followup='always-layered requests increased10.78% in the engineering workload; entry buffers lost substantial desired quote time; no PnL claim',
        validation='prior single and always-layered per-window artifacts must reproduce exactly, then evaluate incremental; unchanged source decisions and parameters',
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=s.a.HERE,text=True).strip(),
        sources={p:s.a.sha(s.a.HERE/p) for p in SOURCES},
        inherited_decisions_sha256=s.a.sha(OLD/'decisions.parquet'),
        inherited_reference_sha256=s.a.sha(OLD/'window_metrics_v2.parquet'))
    for k in ['correction','supersedes_declaration_sha256','prior_receipt_sha256']:d.pop(k,None)
    (DEST/'decisions.parquet').symlink_to(OLD/'decisions.parquet')
    s.a.write(DEST/'extraction.json',json.loads((OLD/'extraction.json').read_text()))
    s.a.write(DEST/'declaration.json',d);s.a.write(REVIEW/'declaration.json',d)
    print('DECLARED incremental follow-up; no new dates/thresholds/fill model')


def run():
    s.DEST=DEST;s.REVIEW=REVIEW;s.ARMS=ARMS
    s.run()


def verify():
    if (DEST/'verification.json').exists():raise ValueError('verification exists')
    r=json.loads((DEST/'receipt.json').read_text());d=r['declaration']
    for p,h in d['sources'].items():
        if s.a.sha(s.a.HERE/p)!=h:raise ValueError('source changed')
    for p,h in r['artifacts'].items():
        if s.a.sha(DEST/p)!=h:raise ValueError('artifact changed')
    if s.a.sha(OLD/'window_metrics_v2.parquet')!=d['inherited_reference_sha256']:raise ValueError('reference changed')
    if s.a.sha(DEST/'receipt.json')!=s.a.sha(REVIEW/'receipt.json'):raise ValueError('review copy differs')
    m=pd.read_parquet(DEST/'window_metrics.parquet');prior=pd.read_parquet(OLD/'window_metrics_v2.parquet')
    for arm in ['single','layered']:
        cols=list(prior.columns)
        x=prior[prior.arm==arm].sort_values(['run','mid']).reset_index(drop=True)
        y=m[m.arm==arm].sort_values(['run','mid']).reset_index(drop=True)[cols]
        pd.testing.assert_frame_equal(x,y,check_exact=True)
    verify_sums(m,r['summary'])
    bad=copy.deepcopy(r['summary']);bad['incremental']['requests']+=1;refused=False
    try:verify_sums(m,bad)
    except ValueError:refused=True
    if not refused:raise ValueError('corrupted count accepted')
    if len(m)!=660 or m.duplicated(['run','mid','arm']).any():raise ValueError('wrong paired population')
    cov=m.pivot(index=['run','mid'],columns='arm',values='desired_lane_ms')
    if not (cov.incremental==cov.single).all():raise ValueError('incremental changes desired coverage')
    if (m.peak_reserved_side>150+1e-9).any():raise ValueError('side budget exceeded')
    out=dict(as_of=dt.datetime.now(dt.timezone.utc).isoformat(),receipt_sha256=s.a.sha(DEST/'receipt.json'),
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=s.a.HERE,text=True).strip(),
        baseline_and_layered_window_rows_exact=440,incremental_windows=220,reference_columns_verified=len(prior.columns),
        controls=dict(old_arm_artifact_parity=True,corrupted_count_refused=refused),
        input_decisions_sha256=s.a.sha(DEST/'decisions.parquet'),limitation='no fill, queue-rank, profit or variance validation')
    s.a.write(DEST/'verification.json',out);s.a.write(REVIEW/'verification.json',out)
    print(json.dumps(out,indent=2))


if __name__=='__main__':{'declare':declare,'run':run,'verify':verify}[sys.argv[1]]()
