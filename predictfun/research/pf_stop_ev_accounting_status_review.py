"""Read-only audit of saved STOP-EV actions/receipts; no raw or reserved days.

Checks quantities, components, normalization, exclusions, unique actions,
timing-population consistency and source identities. Refuses planted duplicate,
quantity/value, invalid-status and source corruptions. No P&L/model selection.
"""
import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import resource

import numpy as np
import pandas as pd

DAYS = ('20261023', '20261024', '20261025', '20261026', '20261027',
        '20261031', '20261101', '20261102', '20261103')
TIMINGS = ('ws_strict', 'all_anchors_sensitivity')
KEY = ['day', 'm', 'side', 'rung', 'generation']
LABEL = ['value_cents', 'value_per_known_remaining_share', 'filled_quantity',
         'past_quantity', 'common_prefix_quantity', 'actual_remaining_after_prefix',
         'at_price_value_cents', 'through_price_value_cents']


def require(ok, message):
    if not bool(ok): raise ValueError(message)


def close(a, b):
    return np.allclose(a, b, rtol=1e-10, atol=1e-7)


def source_check(receipt, source_dir):
    require(bool(receipt.get('sources')), 'no source identities')
    for name, digest in receipt['sources'].items():
        require(hashlib.sha256((source_dir/name).read_bytes()).hexdigest()[:16] == digest,
                'source differs: '+name)


def table_check(df, size):
    require(not df.duplicated(KEY).any(), 'duplicate generation')
    valid = df.status.isin(('eligible', 'eligible_estimated_timing'))
    a, bad = df[valid], df[~valid]
    require(not a.empty, 'empty eligible population')
    require(np.isfinite(a[LABEL].values).all(), 'nonfinite eligible outcome')
    require(bad[LABEL].isna().all().all(), 'invalid action has a value')
    require((bad.n_tranches == -1).all(), 'invalid action has tranches')
    for col in ('filled_quantity', 'past_quantity', 'common_prefix_quantity', 'actual_remaining_after_prefix'):
        require((a[col] >= -1e-7).all(), 'negative '+col)
    require((a.filled_quantity <= a.actual_remaining_after_prefix + 1e-7).all(), 'quantity overused')
    require(close(a.actual_remaining_after_prefix, size-a.past_quantity-a.common_prefix_quantity), 'remaining mismatch')
    require(close(a.x_known_remaining, size-a.x_known_filled), 'client remaining mismatch')
    require((a.x_known_remaining > 0).all() and (a.x_known_filled <= a.past_quantity+1e-7).all(), 'noncausal client fill quantity')
    require(close(a.value_cents, a.at_price_value_cents+a.through_price_value_cents), 'value component mismatch')
    require(close(a.value_cents, a.value_per_known_remaining_share*a.x_known_remaining), 'normalization mismatch')
    z = a.filled_quantity <= 1e-9
    require(close(a.loc[z, 'value_cents'], 0.) and (a.loc[z, 'n_tranches'] == 0).all(), 'unfilled action not zero')
    require((a.loc[~z, 'n_tranches'] >= 1).all(), 'filled action has no tranche')
    return dict(n=len(df), eligible=len(a), statuses={str(k): int(v) for k,v in df.status.value_counts().items()})


def controls(df, rep, source_dir):
    table_check(df, rep['size']); source_check(rep, source_dir)
    tests = {}
    variants = {'duplicate': pd.concat([df, df.iloc[:1]], ignore_index=True)}
    eligible = df.index[df.status.isin(('eligible', 'eligible_estimated_timing'))][0]
    invalid = df.index[~df.status.isin(('eligible', 'eligible_estimated_timing'))][0]
    for name, col, val, idx in [('quantity', 'filled_quantity', 10000., eligible),
                              ('value', 'value_cents', 10000., eligible),
                              ('invalid_status', 'value_cents', 0., invalid)]:
        bad=df.copy(); bad.loc[idx,col]=val; variants[name]=bad
    for name,bad in variants.items():
        try: table_check(bad,rep['size'])
        except ValueError: tests[name]=True
        else: tests[name]=False
    broken=copy.deepcopy(rep); broken['sources']['pf_stop_ev_accounting.py']='0'*16
    try: source_check(broken,source_dir)
    except ValueError: tests['source']=True
    else: tests['source']=False
    require(all(tests.values()), 'a planted corruption escaped')
    return tests


def run(data, out):
    source_dir=Path(__file__).resolve().parent; records={}; hashes={}; src=None; ctl=None
    for day in DAYS:
        p=data/f'day_{day}.json'; rep=json.loads(p.read_text()); source_check(rep,source_dir)
        hashes[p.name]=hashlib.sha256(p.read_bytes()).hexdigest()
        require(rep['day']==day, 'wrong receipt day')
        require(all(x['pass'] for x in rep['checks'].values()), 'failed declared check')
        require(rep['checks']['E9']['mismatch']==0 and rep['checks']['E9']['ledger_failures']==0, 'primary reconciliation failed')
        require(src is None or rep['sources']==src, 'mixed sources'); src=rep['sources']
        pairs={}; item={'asof':rep['asof'], 'E9':rep['checks']['E9'], 'timings':{}}
        for timing in TIMINGS:
            p=data/f'actions_{day}_{timing}.parquet'; df=pd.read_parquet(p)
            hashes[p.name]=hashlib.sha256(p.read_bytes()).hexdigest()
            require((df.day.astype(str)==day).all(), 'wrong action day')
            x=table_check(df,rep['size']); item['timings'][timing]=x
            for status,n in x['statuses'].items():
                require(rep['stat'].get(f'actions_{timing}_{status}',0)==n, 'status count mismatch')
            if ctl is None: ctl=controls(df,rep,source_dir)
            pairs[timing]=df
        strict,all_=pairs['ws_strict'],pairs['all_anchors_sensitivity']
        require(strict[KEY+['decision_ms']].equals(all_[KEY+['decision_ms']]), 'timing variants changed action population')
        features=[c for c in strict if c.startswith('x_')]
        require(strict[features].equals(all_[features]), 'timing variant changed causal features')
        valid=strict.status=='eligible'
        require((all_.loc[valid,'status']=='eligible').all() and close(strict.loc[valid,LABEL],all_.loc[valid,LABEL]), 'strict label differs in sensitivity')
        require(item['timings']['all_anchors_sensitivity']['eligible']==item['E9']['compared'], 'E9 population mismatch')
        records[day]=item
        require(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss < 2048*1024, 'RSS cap')
    totals={t:dict(n=sum(x['timings'][t]['n'] for x in records.values()),eligible=sum(x['timings'][t]['eligible'] for x in records.values())) for t in TIMINGS}
    dev={t:dict(n=sum(records[d]['timings'][t]['n'] for d in DAYS[:4]),eligible=sum(records[d]['timings'][t]['eligible'] for d in DAYS[:4])) for t in TIMINGS}
    result=dict(asof=datetime.now(timezone.utc).isoformat(),status='saved_primary_accounting_checks_pass',
                checker_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                source_asof=max(x['asof'] for x in records.values()),days=records,totals=totals,dev=dev,
                planted_corruptions_refused=ctl,artifact_sha256=hashes,source_ids=src,
                limits=['No raw-feed fill truth validation.','Diagnostic single-order tables are not primary labels.',
                        'Timing exclusions reduce and can select the training population.','No model or performance claim.'])
    out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('x') as f: json.dump(result,f,indent=2); f.write('\n')
    print(json.dumps({k:result[k] for k in ('asof','status','source_asof','totals','dev','planted_corruptions_refused')}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();run(a.data,a.out)
