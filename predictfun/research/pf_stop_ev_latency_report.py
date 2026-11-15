"""Arithmetic-only aggregation of all frozen latency cells; no selection."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path

DAYS=('20261027','20261031','20261101','20261102','20261103')
CFGS=('current','order_zero','modeled_zero','modeled_zero_trade0')
ARMS=('KA','EV','B2','QR','MD1000')


def run(src,out):
    days={d:json.loads((src/d/'receipt.json').read_text()) for d in DAYS}
    assert all(not r['smoke_only'] and r['day']==d and r['stats']['runs']==20*r['stats']['windows'] for d,r in days.items())
    assert len({r['source_commit'] for r in days.values()})==1
    assert all(r['sources']==days[DAYS[0]]['sources'] for r in days.values())
    allsum={};delta={}
    for cfg in CFGS:
        allsum[cfg]={}
        for arm in ARMS:
            rows=[r['totals'][cfg][arm] for r in days.values()]
            z={c:sum(r[c] for r in rows) for c in ['windows','pnl','pnl_old','shares','shares_young','pulls','orders','fill_events','good_value','gross_loss','markout1_cents','markout5_cents','imputed_pnl','imputed_shares','pend_shares','qunk_shares']}
            z.update(pnl_per_day=z['pnl']/5,cents_per_share=100*z['pnl']/z['shares'] if z['shares'] else None,
                live_share=sum(r['live_share']*r['windows'] for r in rows)/z['windows'],
                positive_days=sum(r['pnl']>0 for r in rows),positive_pooled=z['pnl']>0,
                max_abs_position=max(r['max_abs_position'] for r in rows))
            allsum[cfg][arm]=z
    for arm in ARMS:
        delta[arm]={}
        for cfg in CFGS[1:]:
            ds={d:days[d]['totals'][cfg][arm]['pnl']-days[d]['totals']['current'][arm]['pnl'] for d in DAYS}
            delta[arm][cfg]=dict(paired_pnl_by_day=ds,pooled_delta=sum(ds.values()),positive_delta_days=sum(v>0 for v in ds.values()))
    comp={}
    for cfg in CFGS:
        comp[cfg]={}
        for other in ['B2','QR','MD1000','KA']:
            ds={d:days[d]['totals'][cfg]['EV']['pnl']-days[d]['totals'][cfg][other]['pnl'] for d in DAYS}
            comp[cfg][other]=dict(ev_minus_other_by_day=ds,pooled_delta=sum(ds.values()),positive_delta_days=sum(v>0 for v in ds.values()),
                interpretation='Same execution environment, different endogenous orders/volumes; no matched-retention or causal information-skill claim.')
    result=dict(asof=datetime.now(timezone.utc).isoformat(),source_commit=days[DAYS[0]]['source_commit'],
        receipt_hashes={d:hashlib.sha256((src/d/'receipt.json').read_bytes()).hexdigest() for d in DAYS},
        windows=sum(r['stats']['windows'] for r in days.values()),days=days,totals=allsum,clock_deltas=delta,model_comparisons=comp,
        no_significance_claim=True,no_policy_promoted=True)
    out.mkdir(parents=True,exist_ok=True);(out/'report.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    lines=[f"As of {result['asof']}; {result['windows']} first-minute windows on 5 consumed dates. All fills included.",
        'cfg | rule | $/day | c/share | shares | fills | uptime | positive days']
    for cfg in CFGS:
        for arm,z in allsum[cfg].items():
            cps='NA' if z['cents_per_share'] is None else f"{z['cents_per_share']:+.4f}"
            lines.append(f"{cfg} | {arm} | {z['pnl_per_day']:+.2f} | {cps} | {z['shares']:.1f} | {z['fill_events']:.0f} | {z['live_share']:.3%} | {z['positive_days']}/5")
    (out/'summary.txt').write_text('\n'.join(lines)+'\n');print('\n'.join(lines))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--src',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();run(a.src,a.out)
