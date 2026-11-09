"""Artifact checks for inventory-aware grace and duration/latency rounds."""
import copy
import datetime as dt
import json
import math
import subprocess
import sys
import pandas as pd
from . import pf_inventory_grace_study as s
from .pf_inventory_grace import inventory_reason
from .pf_soft_pull_verify import compare_summary


def verify_checks(checks):
    for r in checks.to_dict('records'):
        mode=s.ARMS[r['arm']][1]
        reason=inventory_reason(mode,r['net'],r['lane'],r['remaining'],r['reserved'])
        if r['status']!=('allow' if reason is None else reason):raise ValueError('inventory check differs')


def verify(round_name):
    a=s.a;d=s.check(round_name);dest=s.DEST/round_name;spec=s.ROUNDS[round_name]
    if (dest/'verification.json').exists():raise ValueError('verification exists')
    receipt=json.loads((dest/'receipt.json').read_text())
    for p,h in receipt['artifacts'].items():
        if a.sha(dest/p)!=h:raise ValueError('artifact changed')
    m=pd.read_parquet(dest/'window_metrics.parquet');act=pd.read_parquet(dest/'actions.parquet');checks=pd.read_parquet(dest/'checks.parquet');stat=pd.read_parquet(dest/'window_status.parquet')
    if len(stat)!=220 or int(stat.decisions.sum())!=279059 or not (stat.status=='present').all():raise ValueError('window population differs')
    scope=json.loads((a.RP/'scope.json').read_text());keys={(w['run'],w['market']) for w in scope['windows']}
    if set(zip(stat.run,stat.mid))!=keys:raise ValueError('window identities differ')
    expected=220*len(spec['arms'])*len(spec['latencies'])
    if len(m)!=expected or m.duplicated(['run','mid','latency','arm']).any():raise ValueError('window-arm population differs')
    if act.duplicated(['run','mid','latency','arm','action']).any():raise ValueError('duplicate action')
    s.baseline_parity(m,act)
    ref=act[(act.arm=='reference')&(act.latency=='base')].sort_values(['run','mid','action']).reset_index(drop=True)
    identity=['run','mid','action','lane','start','old_tick','v','midpoint','input_index','spread','start_net']
    for latency in spec['latencies']:
        mm=m[m.latency==latency];compare_summary(mm,receipt['summary'][latency])
        for arm in spec['arms']:
            aa=act[(act.arm==arm)&(act.latency==latency)].sort_values(['run','mid','action']).reset_index(drop=True)
            pd.testing.assert_frame_equal(aa[identity],ref[identity])
            mg=mm[mm.arm==arm];ret=aa[aa.status=='retained']
            if len(aa)!=int(mg.actions.sum()) or len(ret)!=int(mg.retained_actions.sum()):raise ValueError('action total differs')
            if not ((ret.retained_ms>=0)&(ret.retained_ms<=s.ARMS[arm][2])&((ret.end-ret.start)==ret.retained_ms)).all():raise ValueError('hold duration differs')
            deadlines=ret[ret.end_reason=='deadline']
            if not (deadlines.retained_ms==s.ARMS[arm][2]).all():raise ValueError('deadline differs')
            if int((aa.end_reason=='same_price_reopen').sum())!=int(mg.same_price_reopens.sum()):raise ValueError('reopen count differs')
        base=mm[mm.arm=='reference'].set_index(['run','mid'])
        for r in mm.to_dict('records'):
            b=base.loc[(r['run'],r['mid'])]
            if not math.isclose(r['aligned_lane_ms']-b.aligned_lane_ms,r['reference_unquoted_gained_ms']-r['reference_quoted_lost_ms'],abs_tol=1e-4,rel_tol=1e-11):raise ValueError('coverage identity differs')
            if r['inventory_known_ms']+r['inventory_unobserved_ms']!=60000:raise ValueError('inventory-clock coverage differs')
            if any(r[k]<-1e-8 for k in s.RISK) or r['off_reducing_share_ms']+r['off_other_share_ms']>r['share_ms']+1e-4:raise ValueError('exposure totals invalid')
    if s.summarize(m)!=receipt['summary']:raise ValueError('risk summary differs')
    for run,rm in m.groupby('run'):
        if s.summarize(rm)!=receipt['by_run'][run]:raise ValueError('run summary differs')
    verify_checks(checks)
    # Verify the native net at the exact observer/decision identity named by each
    # inventory check, not a nearby timestamp or another decision on that clock.
    for run in s.old.n.RUNS:
        sources=[]
        for kind,path in [(0,s.old.n.DEST/f'states_{run}.parquet'),(1,s.old.grid.DEST/f'probes_{run}.parquet'),(2,s.native_inventory.DEST/f'inventory_{run}.parquet')]:
            z=pd.read_parquet(path,columns=['mid','t','ordinal','input_index','net']);z['source_kind']=kind;z=z.rename(columns={'ordinal':'source_ordinal','net':'native_net'});sources.append(z)
        src=pd.concat(sources,ignore_index=True)
        cc=checks[checks.run==run].merge(src,on=['mid','t','source_kind','source_ordinal','input_index'],how='left',validate='many_to_one')
        if cc.native_net.isna().any() or not (cc.net==cc.native_net).all():raise ValueError('inventory source identity differs')
    # A positive reducing example and deliberately wrong-side allowance prove
    # the validator can distinguish the invariant, rather than only report zero.
    control=pd.DataFrame([dict(arm='reserved100',net=-5.,lane=0,remaining=5.,reserved=5.,status='allow')]);verify_checks(control)
    bad=control.copy();bad.loc[0,'net']=5.
    try:verify_checks(bad)
    except ValueError:refused=True
    else:raise ValueError('wrong-side known-bad accepted')
    badsummary=copy.deepcopy(receipt['summary']['base']);badsummary['reference']['requests']+=1
    try:compare_summary(m[m.latency=='base'],badsummary)
    except ValueError:count_refused=True
    else:raise ValueError('corrupt count accepted')
    repeated=None
    if round_name=='duration_latency':
        parent=s.DEST/'inventory';pr=json.loads((parent/'receipt.json').read_text())
        for name in ['window_metrics','actions','checks']:
            before=pd.read_parquet(parent/f'{name}.parquet')
            if a.sha(parent/f'{name}.parquet')!=pr['artifacts'][f'{name}.parquet']:raise ValueError('inventory round changed')
            after={'window_metrics':m,'actions':act,'checks':checks}[name]
            after=after[(after.latency=='base')&after.arm.isin(s.ROUNDS['inventory']['arms'])]
            # Probe integration splits may vary when shorter-grace arms are also
            # present; order counts and action/check identities remain exact.
            sort=['run','mid','arm']+(['action'] if name=='actions' else ['t','order_id','opening','source_kind','source_ordinal'] if name=='checks' else [])
            pd.testing.assert_frame_equal(before.sort_values(sort,kind='stable').reset_index(drop=True),after.sort_values(sort,kind='stable').reset_index(drop=True),check_exact=False,rtol=1e-11,atol=1e-4)
        repeated=dict(windows=220,arms=4,metrics_actions_checks_reconciled=True)
    population=dict(windows=220,decisions=279059,neutral_actions=len(ref),excluded_windows=0)
    if receipt['population']!=population:raise ValueError('receipt population differs')
    out=dict(as_of=dt.datetime.now(dt.timezone.utc).isoformat(),round=round_name,
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=a.HERE,text=True).strip(),verifier_sha256=a.sha(a.HERE/'pf_inventory_grace_verify.py'),receipt_sha256=a.sha(dest/'receipt.json'),
        rows=dict(window_arm=len(m),action_arm=len(act),inventory_checks=len(checks)),
        old_baseline_and_unrestricted_arm_reconciled=True,native_inventory_identity_reconciled=True,repeated_inventory_round=repeated,
        controls=dict(reducing_positive=True,wrong_side_refused=refused,corrupt_count_refused=count_refused),limitation='conditional mechanics only; no economic or counterfactual risk validation')
    a.write(dest/'verification.json',out);a.write(s.REVIEW/round_name/'verification.json',out);print(json.dumps(out,indent=2))


if __name__=='__main__':verify(sys.argv[1])
