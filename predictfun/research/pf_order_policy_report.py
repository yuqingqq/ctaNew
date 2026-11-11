"""Predeclared portfolio comparisons and matched-action diagnostics, consumed dates only."""
import datetime as dt
import json
import numpy as np
import pandas as pd
from . import pf_order_policy_spec as S


def bootstrap(day_values):
    a=np.asarray(day_values,float)
    if len(a)<5 or not np.isfinite(a).all():raise ValueError('day interval population')
    draws=np.random.default_rng(20261111).choice(a,(1000,len(a)),replace=True).sum(axis=1)
    return np.quantile(draws,[.025,.975]).tolist()


def null(a,keep,values,keys):
    if not np.isfinite(values).all():raise ValueError('null missing values')
    keep=np.asarray(keep,bool);values=np.asarray(values,float)
    groups=list(a.groupby(keys,dropna=False,sort=True).indices.values())
    observed=float(values[keep].sum());rng=np.random.default_rng(20261111);draws=[]
    for _ in range(500):
        z=keep.copy()
        for ix in groups:z[ix]=rng.permutation(keep[ix])
        if z.sum()!=keep.sum():raise ValueError('action count mismatch')
        draws.append(float(values[z].sum()))
    return dict(n=len(a),kept=int(keep.sum()),groups=len(groups),shuffleable=sum(0<keep[ix].sum()<len(ix) for ix in groups),
                observed_c=observed,null_mean_c=float(np.mean(draws)),null_p95_c=float(np.quantile(draws,.95)),
                p_ge=(1+int((np.asarray(draws)>=observed).sum()))/501,draws=500)


def comparison(a,b):
    z=a.merge(b,on=['day','mid'],suffixes=('_a','_b'),validate='one_to_one')
    common=z.marked_strict_a.notna()&z.marked_strict_b.notna();c=z[common]
    daily=(c.marked_strict_a-c.marked_strict_b).groupby(c.day).sum().reindex(S.TEST,fill_value=0)
    settle=(z.pnl_a-z.pnl_b).groupby(z.day).sum().reindex(S.TEST,fill_value=0)
    exposure=(z.inventory_square_s_a-z.inventory_square_s_b).sum()
    return dict(windows=len(z),common_mark_windows=len(c),missing_either_mark=int((~common).sum()),
        marked_delta=float(daily.sum()),marked_day_interval=bootstrap(daily),marked_daily=daily.to_dict(),
        settlement_delta=float(settle.sum()),settlement_day_interval=bootstrap(settle),settlement_daily=settle.to_dict(),
        exposure_delta=float(exposure),exposure_pct=float(100*exposure/z.inventory_square_s_b.sum()),
        terminal_abs_delta=float(z.terminal_abs_a.sum()-z.terminal_abs_b.sum()))


def run():
    declaration=S.check();asof=dt.datetime.now(dt.timezone.utc).isoformat();allrows={};table=[];status=[];receipts=[]
    for scenario in ('X','stress'):
        modes=list(S.MODES) if scenario=='X' else ['reference','continuation','calm_seed','calm_seed_continuation']
        for mode in modes:
            wins=[];gens=[];fills=[];requests=[]
            for day in S.TEST:
                group='stress' if scenario=='stress' else 'build' if mode in ('reference','calm_seed') else 'eval'
                p=S.OUT/'runs'/group/day;c=json.loads((p/'checks.json').read_text())
                if c['status']!='PASS':raise ValueError('unverified run')
                r=next(x for x in c['variants'] if x['mode']==mode)
                for kind,h in r['hashes'].items():
                    if S.sha(p/kind/S.stem(mode))!=h:raise ValueError('artifact changed')
                if S.sha(p/'generations'/f'{mode}.parquet')!=r['generation_sha']:raise ValueError('generation changed')
                ind=pd.read_parquet(p/'independent_windows.parquet');ind=ind[ind['mode']==mode]
                from .pf_order_policy_verify import reconcile
                w=pd.read_parquet(p/'win'/S.stem(mode));f=pd.read_parquet(p/'fills'/S.stem(mode));q=pd.read_parquet(p/'requests'/S.stem(mode))
                g=pd.read_parquet(p/'generations'/f'{mode}.parquet');reconcile(w,f,q,ind,g,288)
                wins.append(ind);gens.append(g);fills.append(f);requests.append(q)
                receipts.append(dict(scenario=scenario,mode=mode,day=day,checks_sha=S.sha(p/'checks.json')))
            w=pd.concat(wins,ignore_index=True);g=pd.concat(gens,ignore_index=True);f=pd.concat(fills);q=pd.concat(requests)
            allrows[(scenario,mode)]=w
            valid=w.marked_strict.notna();vg=g.entry_value1.notna()&g.entry_value5.notna()
            # Each generation contributes once, including zero-fill generations.
            daily=w.groupby('day').pnl.sum()
            table.append(dict(scenario=scenario,mode=mode,windows=len(w),marked_windows=int(valid.sum()),
                settlement=float(w.pnl.sum()),marked=float(w.loc[valid,'marked_strict'].sum()),
                carry_common=float((w.loc[valid,'pnl']-w.loc[valid,'marked_strict']).sum()),
                settlement_common=float(w.loc[valid,'pnl'].sum()),
                markout1_engine=float(f.mo1.sum()),markout60_engine=float(f.mo60.sum()),
                strict_generation_value1=float(g.loc[vg,'entry_value1'].sum()/100),strict_generation_value5=float(g.loc[vg,'entry_value5'].sum()/100),
                generation_mark_known=int(vg.sum()),generation_mark_missing=int((~vg).sum()),generations=len(g),
                filled_generations=int(g.entry_q.gt(0).sum()),fills=len(f),shares=float(f.q.sum()),
                inventory_rms=float(np.sqrt(w.inventory_square_s.sum()/(240.1*len(w)))),inventory_square_s=float(w.inventory_square_s.sum()),
                terminal_abs_mean=float(w.terminal_abs.mean()),terminal_abs_p95=float(w.terminal_abs.quantile(.95)),
                day_settlement_sd=float(daily.std(ddof=1)),places=int(q.kind.eq('place').sum()),cancels=int(q.kind.eq('cancel').sum()),
                eligible=int(g.status.eq('eligible').sum()),kept=int(g.kept.sum()),seeded=int(g.seed.sum()),
                fill_age_median_ms=float(f.age_ms.median()) if len(f) else None))
            for (day,s),n in g.groupby(['day','status']).size().items():status.append(dict(scenario=scenario,mode=mode,day=day,status=s,n=int(n)))
    comparisons={}
    for scenario in ('X','stress'):
        for mode in S.MODES:
            if (scenario,mode) in allrows and mode!='reference':
                comparisons[scenario+':'+mode+'-reference']=comparison(allrows[(scenario,mode)],allrows[(scenario,'reference')])
        comparisons[scenario+':calm_seed_continuation-calm_seed']=comparison(allrows[(scenario,'calm_seed_continuation')],allrows[(scenario,'calm_seed')])
    fit=json.loads((S.OUT/'fit.json').read_text());p=S.OUT/'predictions.parquet'
    if S.sha(p)!=fit['predictions_sha']:raise ValueError('prediction identity')
    p=pd.read_parquet(p);diagnostics={}
    for mode in ('reference','calm_seed'):
        a=p[p['mode']==mode].copy().reset_index(drop=True)
        a['hour']=(a.decision//3600000).astype(int);a['age_bin']=pd.cut(a.order_age_ms,[0,50,100,250,1000,np.inf],include_lowest=True).astype(str)
        a['inv_sign']=np.sign(a.inventory)
        for model in ('queue','market'):
            d=null(a,a[model+'_keep'],a.label,['day','side','hour','age_bin','inv_sign'])
            d['hold_all_value_c']=float(a.label.sum());d['no_incremental_fill_n']=int(a.label_q.eq(0).sum())
            diagnostics[mode+':'+model]=d
    gs=[]
    for day in S.TEST:gs.append(pd.read_parquet(S.OUT/'runs/build'/day/'generations/reference.parquet'))
    a=pd.concat(gs,ignore_index=True);a=a[(np.where(a.side==0,1,-1)*a.entry_inventory>=0)].copy()
    missing=a.entry_value1.isna()|a.entry_value5.isna();missing_n=int(missing.sum());a=a[~missing].reset_index(drop=True)
    a['hour']=(a.t_send//3600000).astype(int);a['size_bin']=pd.cut(a.initial_size,[0,5.00001,10.00001,25.00001,np.inf]).astype(str)
    d=null(a,a.entry_calm,.5*(a.entry_value1+a.entry_value5),['day','side','hour','size_bin'])
    d['missing_mark_n']=missing_n;d['unfilled_n']=int(a.entry_q.eq(0).sum());diagnostics['calm_entry_reference']=d
    predicates={}
    for mode in ('continuation','calm_seed','calm_seed_continuation'):
        x=comparisons['X:'+mode+'-reference'];st=comparisons['stress:'+mode+'-reference']
        predicates[mode]=dict(primary_improves_trading_reduces_exposure=x['marked_delta']>0 and x['exposure_delta']<=0,
            stress_improves_trading_reduces_exposure=st['marked_delta']>0 and st['exposure_delta']<=0,
            marked_interval_positive=x['marked_day_interval'][0]>0)
    result=dict(as_of=asof,days=S.TEST,days_consumed=True,validation=False,table=table,comparisons=comparisons,
        diagnostics=diagnostics,predicates=predicates,statuses=status,fit=fit,receipts=receipts,declaration_sha=S.sha(S.OUT/'declaration.json'))
    S.write(S.OUT/'comparison.json',result);S.write(S.REVIEW/'comparison.json',result)
    lines=['# Own-order continuation and calm entry',f'As of {asof}. Six consumed dates: {", ".join(S.TEST)}. Exploratory, not untouched validation.',
        '', 'The neutral engine uses the frozen v3 admissions. Every build reproduces the saved reference win/fill/request artifacts exactly. All sent generations are traced, including unfilled and rejected orders. Continuation is trained chronologically on first soft-pull decisions; cached admission-guard predictions are LODO, so this is conditional chronology, not a historically deployable full-strategy backtest.',
        '', '| Timing | Policy | Windows / valid marks | Settlement $ | Marked $ | Carry on marked windows $ | 1s markout $ | Inventory RMS sh | End abs mean sh | Places | Kept | Seeded |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in table:lines.append(f"| {r['scenario']} | {r['mode']} | {r['windows']} / {r['marked_windows']} | {r['settlement']:.2f} | {r['marked']:.2f} | {r['carry_common']:.2f} | {r['markout1_engine']:.2f} | {r['inventory_rms']:.3f} | {r['terminal_abs_mean']:.3f} | {r['places']} | {r['kept']} | {r['seeded']} |")
    lines += ['', 'Marked totals have different missing-mark populations; use the paired common-window deltas below. Carry + marked equals settlement only on that marked population. Engine 1s/60s markouts retain its historical last-valid-book convention; strict 1s/5s generation markouts and missing counts are in comparison.json. Markouts are not additive to settlement P&L. Rebates excluded.',
        '', '| Comparison | Common marks / windows | Marked delta $ [day interval] | Full settle delta $ | Inventory-square delta % |', '|---|---:|---:|---:|---:|']
    for key,r in comparisons.items():lines.append(f"| {key} | {r['common_mark_windows']} / {r['windows']} | {r['marked_delta']:+.2f} [{r['marked_day_interval'][0]:+.2f}, {r['marked_day_interval'][1]:+.2f}] | {r['settlement_delta']:+.2f} | {r['exposure_pct']:+.3f} |")
    lines+=['','Intervals resample six UTC days, 1,000 draws. They do not correct for historical strategy selection. Primary X: trade anchor -50ms, place40/cancel22/notice100ms, queue M. Stress: -250ms,80/100/250ms, queue L. No fill/queue calibration claim.','',
        '| Neutral-action diagnostic | n / keep | Observed value c | Matched null mean c | p >= observed |', '|---|---:|---:|---:|---:|']
    for k,r in diagnostics.items():lines.append(f"| {k} | {r['n']} / {r['kept']} | {r['observed_c']:.2f} | {r['null_mean_c']:.2f} | {r['p_ge']:.3f} |")
    lines+=['','500 matched-count shuffles per diagnostic. Continuation matches day/side/hour/age bucket/inventory sign; the market control includes age. Calm entry matches day/side/hour/size. All include unfilled zero outcomes. Shadow labels are overlapping action counterfactuals, not portfolio P&L; existing competitors/pending cancels are represented, future reference admissions are not forked. Full replays above test interactions.',
        '', 'Computed predicates: '+json.dumps(predicates,sort_keys=True), '', 'No promotion: these are consumed dates and the live incremental-size/soft-pull implementation is not the frozen v3 baseline. Source/declaration, per-day checks, complete statuses, model fold counts, artifact hashes and all comparisons are in reviews/own_orders_calm_20261111/ and data/predict_fun/research/own_orders_calm_20261111/.']
    report=S.HERE.parent/'handoffs/20261111_own_orders_calm_results.md'
    with report.open('x') as f:f.write('\n'.join(lines)+'\n')
    print('REPORT',report,flush=True)


if __name__=='__main__':run()
