"""Reconciled capture/risk decomposition and day-clustered exploratory comparisons."""
import datetime as dt
import json
import numpy as np
import pandas as pd
from .pf_inventory_cost_study import HERE,DEST,DATA,DAYS,ARMS,PAIRS,sha,write,check
from .pf_inventory_cost_run import folder,stem,verify
from .pf_entry_report import full_path
from .pf_mm_decomp import mid_maps,decompose,recon
from .pf_settlement_study import paired_inference

SUMS=['settle','plus1','spread','adverse_1s','mark_stop','carry','shares','paired_pnl','paired_shares','residual_pnl','concession','committed_concession','fills','places','cancels']
MEANS=['inventory_square','mean_abs_inventory','mean_abs_net_end','seconds_above10']


def summarize(day,tm,mode,books,registry):
    out=folder(day,tm,mode);checks=json.loads((out/'checks.json').read_text())
    for rel,h in checks['artifacts'].items():
        if sha(out/rel)!=h:raise ValueError('changed artifact')
    meta=pd.read_parquet(out/'late_fills.parquet');budgets=pd.read_parquet(out/'cost_budgets.parquet');rows=[]
    for label,spec in ARMS.items():
        w,f,q=(pd.read_parquet(out/k/stem(spec)) for k in ('win','fills','requests'))
        D=decompose(f,books,registry,240)
        bad=recon(D,f,w)
        if bad:raise ValueError(str(bad))
        if D[['spread','plus1','mark']].isna().any().any():raise ValueError('missing valuation')
        if (f.t>240.1).any():raise ValueError('fill beyond common mark boundary')
        groups={m:g for m,g in f.groupby('mid',sort=False)}
        paths=pd.DataFrame([dict(mid=r.mid,**full_path(groups.get(r.mid,f.iloc[:0]))) for r in w.itertuples()])
        if not np.allclose(paths.net_end,w.net_end,atol=1e-7,rtol=0):raise ValueError('inventory reconciliation')
        matched=float(paths.paired_pnl.sum())
        late=meta[meta.variant==label]
        row=dict(day=day,timing=tm,mode=mode,variant=label,markets=len(w),settle=float(w.pnl.sum()),
            plus1=float(f.mo1.sum()),spread=float(D.spread.sum()),adverse_1s=float(D.informed.sum()),
            mark_stop=float(w.mtm_stop.sum()),carry=float(w.carry_stop.sum()),shares=float(f.q.sum()),fills=len(f),
            paired_pnl=matched,paired_shares=float(paths.paired_shares.sum()),residual_pnl=float(w.pnl.sum()-matched),
            concession=float(late.concession.sum()),committed_concession=float(budgets[budgets.variant==label].committed_concession.sum()),
            inventory_square=float(paths.inventory_square.mean()),mean_abs_inventory=float(paths.mean_abs_inventory.mean()),
            mean_abs_net_end=float(paths.net_end.abs().mean()),seconds_above10=float(paths.seconds_above10.mean()),
            peak_inventory=float(paths.peak_abs_inventory.max()),places=int((q.kind=='place').sum()),cancels=int((q.kind=='cancel').sum()),
            statuses=dict(excluded_markets=0,missing_valuations=0))
        rows.append(row)
    a=pd.read_parquet(out/'exit_episodes.parquet');a['seconds']=a.end-a.start
    # Joint predicates can overlap; first reasons form a partition. Flat/stop retained in all tables.
    active=a[(a.net.abs()>1e-9)&(a.start<240)]
    byreason=active.groupby('first_reason').agg(episodes=('mid','size'),decisions=('decisions','sum'),seconds=('seconds','sum')).to_dict('index')
    allflags={}
    for r in active.itertuples():
        for flag in r.flags:
            v=allflags.setdefault(flag,dict(episodes=0,seconds=0.));v['episodes']+=1;v['seconds']+=r.seconds
    blockers=set(('invalid_price','off_touch','book_gap','binance','missing_score','minimum_order','rounded_minimum','room_below_minimum','no_book','invalid_book'))
    score_only=active[active.flags.map(lambda flags:'score_line' in flags and not blockers.intersection(flags))]
    audit=dict(day=day,timing=tm,mode=mode,markets=int(a.mid.nunique()),episodes=len(a),decisions=int(a.decisions.sum()),
        total_seconds=float(a.seconds.sum()),inventory_episode_count=len(active),inventory_seconds=float(active.seconds.sum()),
        first_reason=byreason,overlapping_flags=allflags,score_only_executable_seconds=float(score_only.seconds.sum()),
        note='decision-sampled reference states, no fill selection; intervals carry observed state to next decision, not true continuous queue observations')
    return rows,audit


def main():
    declaration=check()
    for p,h in declaration['input_sha'].items():
        if sha(p)!=h:raise ValueError('input changed')
    from . import pf_io
    pf_io.OUT=str(DATA)
    daily=[];audits=[];artifact_hashes={}
    for day in DAYS:
        reg=pd.read_parquet(DATA/'step1'/day/'registry.parquet').set_index('mid')
        summary=json.loads((folder(day,'J')/'summary.json').read_text());bounds=summary['input_bounds']
        books=mid_maps(day,bounds['t_lo_ms'],bounds['t_hi_ms'])
        cases=[('J','main'),('X','main')]+([('X','stress')] if day=='20261103' else [])
        for tm,mode in cases:
            verify(day,tm,mode)
            rr,aa=summarize(day,tm,mode,books,reg);daily.extend(rr);audits.append(aa)
            artifact_hashes[str(folder(day,tm,mode)/'checks.json')]=sha(folder(day,tm,mode)/'checks.json')
        del books
    result=[]
    for tm in ('J','X'):
        for label in ARMS:
            ds=[r for r in daily if (r['timing'],r['variant'],r['mode'])==(tm,label,'main')]
            if len(ds)!=9 or sum(r['markets'] for r in ds)!=2592:raise ValueError('aggregate population')
            result.append(dict(timing=tm,variant=label,markets=2592,days=9,
                **{k:sum(r[k] for r in ds) for k in SUMS},**{k:float(np.mean([r[k] for r in ds])) for k in MEANS},
                peak_inventory=max(r['peak_inventory'] for r in ds),daily_pnl_sd=float(np.std([r['settle'] for r in ds],ddof=1)),
                per_day=ds))
    idx={(r['timing'],r['variant']):r for r in result}
    family=[(tm,a,b,m) for tm in ('J','X') for a,b in PAIRS for m in ('settle','plus1','inventory_square')]
    delta=np.array([[idx[t,a]['per_day'][i][m]-idx[t,b]['per_day'][i][m] for t,a,b,m in family] for i in range(9)])
    inference={f'{t}/{a}-{b}/{m}':v for (t,a,b,m),v in zip(family,paired_inference(delta))}
    comparisons=[]
    for tm in ('J','X'):
        for a,b in PAIRS:
            x,y=idx[tm,a],idx[tm,b]
            comparisons.append(dict(timing=tm,candidate=a,reference=b,delta={k:x[k]-y[k] for k in SUMS+MEANS},
                retained_plus1=x['plus1']/y['plus1'] if y['plus1'] else None,
                terminal_inventory_ratio=x['mean_abs_net_end']/y['mean_abs_net_end'] if y['mean_abs_net_end'] else None,
                value_and_inventory_objective=(x['plus1']>=y['plus1'] and x['mark_stop']>=y['mark_stop'] and x['mean_abs_net_end']<y['mean_abs_net_end'])))
    report=dict(as_of=dt.datetime.now(dt.timezone.utc).isoformat(),declaration=declaration,results=result,
        comparisons=comparisons,inference=inference,stress=[r for r in daily if r['mode']=='stress'],audits=audits,
        artifacts=artifact_hashes,controls=(DEST/'logs/controls.log').read_text(),
        limitations=['Consumed-data experiment; no calibrated or independently validated live performance.',
        'Guard relief has no dollar guarantee. Budget bounds quoted concession only, not adverse selection.',
        'FIFO includes price drift; mid-at240 is not an executable exit; inventory-square is an exposure metric.',
        'Current native incremental execution is not modelled. J/X/stress are assumptions, not observed clock calibration.'],
        live_changes=False,protected_data_read=False)
    if 'rc=0 ' not in report['controls'] or 'passed' not in report['controls']:raise ValueError('missing controls')
    if (DEST/'report.json').exists():raise ValueError('refuse report overwrite')
    write(DEST/'report.json',report);write(HERE.parent/'reviews/inventory_cost_20261110/receipt_v2.json',report)
    pd.DataFrame(result).drop(columns=['per_day']).to_csv(DEST/'comparison.csv',index=False)
    for r in result:print({k:r[k] for k in ('timing','variant','settle','spread','adverse_1s','plus1','mean_abs_net_end','inventory_square','concession')})


if __name__=='__main__':main()
