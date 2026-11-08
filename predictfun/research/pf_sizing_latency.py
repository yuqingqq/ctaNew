"""Measure actual cancel acknowledgements; stress reservations, not physical fills."""
import datetime as dt
import gzip
import json
import os
import subprocess
import sys
import numpy as np
from .pf_sizing_study import DATA,DEST,HERE,write,sha
from .pf_sizing_run import COMMON,specs

CAP='/home/yuqing/pmmm-data/review-exports/tokyo5-import-20261108/tokyo5-samewindow-20261108/live/run-1794071211506/'
SCOPE='/home/yuqing/pmmm-data/review-exports/tokyo5-local-comparison-20261108/scope.json'

def delay(sent,seen):
    v=seen-sent
    if not np.isfinite(v) or v<-.001:raise ValueError('cancel acknowledged before request')
    return max(0.,v)

def measure():
    assert delay(1,23)==22
    try:delay(23,1)
    except ValueError:pass
    else:raise AssertionError('negative-delay control failed')
    cohort={x['market'] for x in json.load(open(SCOPE))['windows']};cancel={};done={}
    with gzip.open(CAP+'outputs.jsonl.gz','rt') as f:
        for line in f:
            if '"Cancel"' not in line:continue
            r=json.loads(line)['Request']['Cancel']
            if r['market'] in cohort:
                key=(r['market'],r['id'])
                if key in cancel:raise ValueError('duplicate cancel identity')
                cancel[key]=r['t']
    with gzip.open(CAP+'inputs.jsonl.gz','rt') as f:
        for line in f:
            if '"CancelDone"' not in line:continue
            r=json.loads(line)['CancelDone'];key=(r['market'],r['id'])
            if key in cancel:done.setdefault(key,r['known_ms'])
    values=np.array([delay(t,done[k]) for k,t in cancel.items() if k in done])
    if len(cancel)!=13775:raise ValueError('wrong cancel population')
    report=dict(as_of=dt.datetime.now(dt.timezone.utc).isoformat(),windows=105,cancels=len(cancel),
      matched_ack=len(values),missing_ack=len(cancel)-len(values),
      quantiles_ms=dict(zip(['p50','p90','p99','max'],np.quantile(values,[.5,.9,.99,1]).tolist())),
      controls=dict(known_delay_detected=True,negative_delay_refused=True),
      source_sha={p:sha(p) for p in [CAP+'inputs.jsonl.gz',CAP+'outputs.jsonl.gz',SCOPE]},
      interpretation='request to strategy-visible cancellation confirmation; completed cancels only, not physical venue removal',
      stress='all cancellations reserve until the measured p90; physical removal remains original 22ms/pending95ms; diagnostic not calibrated simulation')
    write(DEST/'cancel_ack_latency.json',report);print(report)

def stress(tm):
    if tm not in ['J','X']:raise ValueError('timing')
    ack=json.load(open(DEST/'cancel_ack_latency.json'))['quantiles_ms']['p90']
    day='20261103';tag='sst_ack90_'+tm
    out=DATA/'step4'/f'ladder_tokyoreal_custom_e2_preds_basis_study_{tag}'/day
    if out.exists():raise ValueError('refuse overwrite')
    env=dict(os.environ,**COMMON,PF_PREDS='preds_basis_study',PF_TAG=tag,PF_MAX_WINDOWS='0',
      PF_ARMS=';'.join(specs().values()),PF_STUDY_EXEC='1',PF_CAP_CACHE='1',PF_STUDY_CANCEL_ACK_MS=str(ack),
      PF_TXJIT='frame' if tm=='J' else '',PF_TXSHIFT='0' if tm=='J' else '-50',PF_QACT='' if tm=='J' else 'max')
    env.pop('PF_LIVE_CAPSTATE_IN',None)
    subprocess.run([sys.executable,'-m','predictfun.research.pf_sizing_engine',day,'tokyoreal','custom'],env=env,cwd=HERE.parents[1],check=True)

if __name__=='__main__':
    if sys.argv[1:]==['measure']:measure()
    elif len(sys.argv)==3 and sys.argv[1]=='stress':stress(sys.argv[2])
    else:raise ValueError('mode')
