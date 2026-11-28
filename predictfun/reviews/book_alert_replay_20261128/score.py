"""Reconcile the new market rows to the original receipts and render the three-day result."""
import json
import math
import sys

import numpy as np
import pandas as pd
from book_signal import DAYS, HERE, OUT, digest
sys.path.insert(0,str(HERE.parent/'pressure_market_sharpe_20261128'))
from diagnose import load_day, metrics


def validate(f,ref):
    assert f.mid.is_unique and set(f.mid)==set(ref.mid)
    source=ref.set_index('mid').loc[f.mid]
    assert list(source.tape_id)==list(f.tape_id)
    for key in ('settle','turnover','residual','shares','posts','requests','abs_end','closed','reposts'):
        np.testing.assert_allclose(f['current_'+key],source['event_'+key],rtol=0,atol=1e-6)
    for a,b in [('current','baseline'),('alert','candidate')]:
        np.testing.assert_allclose(f[a+'_settle'],f[a+'_turnover']+f[a+'_residual'],rtol=0,atol=1e-6)
        np.testing.assert_allclose(f[a+'_settle'],f[b+'_payout_win']+f[b+'_payout_loss'],rtol=0,atol=1e-6)
        np.testing.assert_allclose(f[a+'_shares'],f[b+'_shares'],rtol=0,atol=1e-6)
    for arm in ('lost','added','baseline','candidate','timely_baseline'):
        np.testing.assert_allclose(f[arm+'_shares'],sum(f[arm+'_'+k] for k in ('censored','win_sh','loss_sh','flat_sh')),rtol=0,atol=1e-6)
    assert (f.timely_baseline_shares<=f.baseline_shares+1e-6).all()
    for k in ('shares','payout_win','payout_loss','adverse','favorable','censored'):
        np.testing.assert_allclose(f['candidate_'+k]-f['baseline_'+k],f['added_'+k]-f['lost_'+k],rtol=0,atol=1e-6)


def score(g):
    r={}
    for a,b in [('current','baseline'),('alert','candidate')]:
        v=metrics(g[a+'_settle'])
        v.update({k:float(g[a+'_'+k].sum()) for k in ('turnover','residual','closed','shares','posts','requests','reposts')})
        v['mean_abs_end']=float(g[a+'_abs_end'].mean())
        v['mark_1s']=100*(g[b+'_favorable']+g[b+'_adverse']).sum()/(g[b+'_shares']-g[b+'_censored']).sum()
        r[a]=v
    for b in ('baseline','candidate','lost','added','timely_baseline'):
        r[b+'_fills']={k:float(g[b+'_'+k].sum()) for k in ('favorable','adverse','censored','shares','payout_win','payout_loss')}
    for k in ('grid_pulls','offgrid_pulls','waits','reentries','ahead_before','ahead_after','reentry_ms'):
        r[k]=float(g[k].sum())
    diff=g.alert_settle-g.current_settle
    r.update(net=float(diff.sum()),better=int((diff>1e-7).sum()),worse=int((diff< -1e-7).sum()))
    return r


def main():
    panels=[]
    for day in DAYS:
        f=pd.read_csv(OUT/day/'markets.csv',float_precision='round_trip')
        ref=load_day(day)
        validate(f,ref)
        corrupted=f.copy()
        corrupted.loc[0,'lost_adverse']-=1
        try:
            validate(corrupted,ref)
        except AssertionError:
            pass
        else:
            raise AssertionError('attribution corruption survived')
        panels.append(f)
    f=pd.concat(panels)
    by={str(d):score(g) for d,g in f.groupby('day')}
    p=score(f)
    deltas=[v['net'] for v in by.values()]
    ex=p['net']-max(deltas)
    gates={'settlement':p['net']>0,'market_sharpe':p['alert']['sharpe']>p['current']['sharpe'],
           'at_least_two_days':sum(x>0 for x in deltas)>=2,'without_best_day':ex>0}
    tagged=p['timely_baseline_fills']
    all_fills=p['baseline_fills']
    def mark(x):
        qty=x['shares']-x['censored']
        return (x['favorable']+x['adverse'])*100/qty if qty>0 else None
    untagged={k:all_fills[k]-tagged[k] for k in all_fills}
    timing=dict(eligible_alert_mark=mark(tagged),remaining_mark=mark(untagged),
                gross_adverse_share=tagged['adverse']/all_fills['adverse'],
                gross_favorable_share=tagged['favorable']/all_fills['favorable'])
    result=dict(timing=timing,days=by,pooled=p,gates=gates,passed=all(gates.values()),without_best=ex,
                sources={d:digest(OUT/d/'markets.csv') for d in DAYS},model_sha256=digest(HERE/'model.json'))
    (HERE/'results.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    f.to_csv(HERE/'paired_markets.csv',index=False)
    m=json.loads((HERE/'model.json').read_text())
    text=['# Binance best-level alert replay','',
          f'The candidate {"passed" if result["passed"] else "failed"} the frozen development screen. November 28, 2026 UTC. The replay includes {len(f)} eligible five-minute markets of 864 scheduled, on three previously studied dates. The original census excludes 465 missing endpoints and 181 stale books. Excluded markets have unknown P&L. The two arms use the same tape, zero maker fees and rebates, and independent simulated fills and inventory.',
          '',f'The ten-input logistic score was fit on {m["train_rows"]:,} random and nearby points on October 23, October 31, November 1 and November 2. The cutoff of {m["probability"]:.6f} was fixed from the earlier out-of-fold random scores. Training excluded the top taker’s order moments. The replay uses the Binance exchange clock plus 4 ms, 37 ms to remove a live order, and the original 50 ms placement grid. There is no cooldown. Directional book alerts use the pre-move population of the original study.',
          '', '| UTC day | Markets | Current USD | With alert USD | Difference USD | Current / alert market Sharpe |',
          '|---|---:|---:|---:|---:|---:|']
    for d,r in [*by.items(),('Pooled',p)]:
        a,b=r['current'],r['alert']
        text.append(f'| {d} | {a["n"]} | {a["total"]:+,.2f} | {b["total"]:+,.2f} | {r["net"]:+,.2f} | {a["sharpe"]:.4f} / {b["sharpe"]:.4f} |')
    text+=['',f'The paired policy improved {p["better"]} markets, worsened {p["worse"]}, and tied on {len(f)-p["better"]-p["worse"]}. The improvement excluding its best day was {ex:+,.2f} USD. Sharpe is mean settlement divided by sample SD at fixed capital, per market, unannualized. No interval on three days.',
           '', '| Measure | Current | With alert |','|---|---:|---:|']
    for k,name in [('worst_decile','Worst-decile mean market P&L, USD'),('turnover','Completed FIFO P&L, USD'),('residual','Residual inventory P&L, USD'),('shares','Filled shares'),('requests','Cancel requests'),('reposts','Reposts'),('mark_1s','Net one-second markout, cents per valued share'),('mean_abs_end','Mean absolute inventory at expiry, shares')]:
        text.append(f'| {name} | {p["current"][k]:,.3f} | {p["alert"][k]:,.3f} |')
    text+=['', '| Fill cohort | Shares | Shares lacking +1s mark | Adverse +1s USD | Favorable +1s USD | Settlement USD |','|---|---:|---:|---:|---:|---:|']
    for b in ('baseline','candidate','lost','added','timely_baseline'):
        x=p[b+'_fills']
        text.append(f'| {b} | {x["shares"]:,.1f} | {x["censored"]:,.1f} | {x["adverse"]:+,.2f} | {x["favorable"]:+,.2f} | {x["payout_win"]+x["payout_loss"]:+,.2f} |')
    text+=['','Lost and added shares compare entire feedback paths on the same public trade, side and own quote price. The timely baseline cohort asks whether an extra live-order pull could have reached the venue before an actual baseline fill. It is an exposure diagnostic. Keeping later orders and inventory fixed would give a different counterfactual.',
           '',f'The overlay issued {p["offgrid_pulls"]:,.0f} incremental off-grid pulls and {p["grid_pulls"]:,.0f} incremental grid pulls on the candidate path, and withheld {p["waits"]:,.0f} otherwise eligible grid proposals. Repeated proposals can refer to one slot. Of the pulled generations, {p["reentries"]:,.0f} had a subsequent accepted order on the same side at the same price.']
    if p['reentries']:
        text.append(f'On those observations, the mean tracked external shares ahead at pull and at the next accepted activation were {p["ahead_before"]/p["reentries"]:,.1f} and {p["ahead_after"]/p["reentries"]:,.1f}. Mean time from pull request to that activation was {p["reentry_ms"]/p["reentries"]/1000:,.2f} seconds. A depth change between the two observations can change that difference.')
    if timing['eligible_alert_mark'] is not None:
        text+=['',f'Baseline fills with enough cancel lead had a net one-second markout of {timing["eligible_alert_mark"]:+.3f} cents per valued share. Remaining baseline fills averaged {timing["remaining_mark"]:+.3f}. The timely cohort contained {100*timing["gross_adverse_share"]:.1f}% of baseline gross adverse dollars and {100*timing["gross_favorable_share"]:.1f}% of gross favorable dollars.']
    text+=['','The original grid and event controls reproduced their archived market panels. Earlier raw feature vectors matched the source vectors in both directions, and truncating future updates preserved earlier features. Removing one dollar from a lost-fill subtotal failed the reconciliation. Live execution, market impact and unobserved markets remain outside this replay.',
           '', '[Protocol](protocol.md) · [Paired markets](paired_markets.csv) · [Computed metrics](results.json)']
    (HERE/'report.md').write_text('\n'.join(text)+'\n')
    print('\n'.join(text))
if __name__=='__main__':
    main()
