"""Independent artifact checks of the identity/clock repair, including real CLI refusal."""
import datetime as dt
import json
import os
import subprocess
import sys
import numpy as np
import pandas as pd
from . import pf_trade_tape_build as b


def overlap_counts(c):
    cols=['lifecycle_lo_ns','lifecycle_hi_ns','match_lo_ns','match_hi_ns']
    if not len(c) or c[cols].isna().any().any():raise ValueError('missing interval population')
    if (c.lifecycle_lo_ns>c.lifecycle_hi_ns).any() or (c.match_lo_ns>=c.match_hi_ns).any():
        raise ValueError('invalid intervals')
    yes=(c.match_lo_ns<=c.lifecycle_hi_ns)&(c.match_hi_ns>c.lifecycle_lo_ns)
    return dict(own_interval_overlaps=int(yes.sum()),own_interval_nonoverlap=int((~yes).sum()))


def check_counts(actual,expected):
    if any(expected[k]!=v for k,v in actual.items()):raise ValueError('receipt differs from actual artifact')


def main():
    updated=sys.argv[1:]==['updated']
    name='verification_v3.json' if updated else 'verification.json'
    if (b.DEST/name).exists():raise ValueError('verification already exists')
    control=pd.DataFrame(dict(lifecycle_lo_ns=[100,100],lifecycle_hi_ns=[200,200],match_lo_ns=[150,210],match_hi_ns=[160,220]))
    positive=overlap_counts(control)==dict(own_interval_overlaps=1,own_interval_nonoverlap=1)
    bad=False
    try:check_counts(overlap_counts(control),dict(own_interval_overlaps=2,own_interval_nonoverlap=0))
    except ValueError:bad=True
    if not positive or not bad:raise ValueError('verification controls failed')
    r=json.loads((b.DEST/'receipt.json').read_text())
    changed=[p for p,h in r['declaration']['sources'].items() if b.audit.sha(b.audit.HERE/p)!=h]
    if changed!=(['pf_step4_ladder_identity.py'] if updated else []):raise ValueError('unexpected repair source change: '+str(changed))
    if updated:
        checks=(b.DEST/'logs/controls_v3.log').read_text()
        if '\nOK\n' not in checks or 'rc=0 ' not in checks:raise ValueError('missing final controls')
    for p,h in r['artifacts'].items():
        if b.audit.sha(b.DEST/p)!=h:raise ValueError('repair artifact changed')
    if b.audit.sha(b.DEST/'manifest.json')!=r['manifest_sha256']:raise ValueError('manifest changed')
    if b.audit.sha(b.DEST/'receipt.json')!=b.audit.sha(b.REVIEW/'receipt.json'):raise ValueError('receipt copy differs')
    L=pd.read_parquet(b.DEST/'legs.parquet',columns=['leg_id','maker_order_hash','transaction_hash','taker_order_hash','settlement_id',
        'mid','shares','clock_status','match_lo_ns','match_hi_ns','price_grid_status'])
    E=pd.read_parquet(b.DEST/'status_events.parquet');O=pd.read_parquet(b.DEST/'action_reconciliation.parquet')
    C=pd.read_parquet(b.DEST/'own_clock_cases.parquet')
    check_counts(overlap_counts(C),r['reconciliation'])
    if L.leg_id.duplicated().any() or not (L.leg_id==L.transaction_hash+'|'+L.taker_order_hash+'|'+L.maker_order_hash).all():
        raise ValueError('full identity disagrees')
    if len(O)!=42064 or len(L)!=r['population']['rest_legs'] or set(O.market)!=set(L.mid):raise ValueError('population changed')
    if not np.isclose(O.core_filled,O.tape_filled,rtol=0,atol=1e-6).all():raise ValueError('quantity mismatch')
    m=L[L.clock_status=='matched_status_interval'];unknown=L[L.clock_status=='missing_matched_event']
    if unknown[['match_lo_ns','match_hi_ns']].notna().any().any():raise ValueError('unknown trade acquired a clock')
    groups={k:g for k,g in E.groupby(['maker_order_hash','trade_id'])}
    for x in m.itertuples():
        g=groups[(x.maker_order_hash,x.settlement_id or x.transaction_hash)]
        matched=g[g.status=='Matched']
        spans=set(zip(matched.source_lo_ns,matched.source_hi_ns))
        if spans!={(x.match_lo_ns,x.match_hi_ns)}:raise ValueError('matched clock replaced or clipped')
    own=L[L.maker_order_hash.isin(set(O.order_hash.dropna()))]
    old=pd.read_parquet(b.audit.DEST/'own_rest_legs.parquet')
    if set(zip(own.transaction_hash,own.maker_order_hash,own.settlement_id))!=set(zip(old.tx,old.order_hash,old.trade_id)):
        raise ValueError('exact own fill population differs')
    for col,report_key in [('clock_status','clock_status'),('price_grid_status','price_grid_status')]:
        expected=r['population'][report_key] if col=='clock_status' else r[report_key]
        if L[col].value_counts().to_dict()!=expected:raise ValueError('status population differs')
    env={k:v for k,v in os.environ.items() if not k.startswith('PF_')};env['PF_TRADE_TAPE']=str(b.DEST)
    run=subprocess.run([sys.executable,'-m','predictfun.research.pf_step4_ladder_identity',b.audit.DAY,'tokyoreal','custom'],
        cwd=b.audit.HERE.parents[1],env=env,text=True,capture_output=True)
    refused=run.returncode!=0 and 'REFUSED: point replay requires match-event timestamps' in run.stderr and 'Traceback' not in run.stderr
    if not refused:raise ValueError('actual engine CLI accepted unsupported clocks: '+run.stderr)
    out=dict(as_of=dt.datetime.now(dt.timezone.utc).isoformat(),verifier_sha256=b.audit.sha(b.audit.HERE/'pf_trade_tape_verify.py'),
        receipt_sha256=b.audit.sha(b.DEST/'receipt.json'),artifact_hashes=len(r['artifacts']),
        exact_trade_rows_verified=len(L),order_actions_verified=len(O),own_fills_verified=len(own),
        status_intervals_without_clipping=len(m),unknown_rows_without_invented_time=len(unknown),
        engine_cli=dict(returncode=run.returncode,refused=refused,message=run.stderr.strip()),
        controls=dict(overlap_and_nonoverlap=positive,corrupted_receipt_refused=bad))
    if updated:
        out.update(supersedes_verification_sha256=b.audit.sha(b.DEST/'verification.json'),
            correction='new replay entrypoint partitions actual match-event times into pre/live/post market windows and records exclusion counts; data producer and its receipt are unchanged',
            runtime_source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=b.audit.HERE,text=True).strip(),
            runtime_source_sha256={p:b.audit.sha(b.audit.HERE/p) for p in ['pf_step4_ladder_identity.py','pf_trade_windows.py','tests/test_trade_windows.py']},
            final_controls=checks)
    b.audit.write(b.DEST/name,out);b.audit.write(b.REVIEW/name,out)
    print(json.dumps(out,indent=2))


if __name__=='__main__':main()
