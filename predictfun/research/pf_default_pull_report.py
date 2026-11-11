"""Complete portfolio comparisons, nulls and saved-model identity checks."""
import datetime as dt
import json
import numpy as np
import pandas as pd
from . import pf_default_pull_spec as S
from .pf_order_policy_report import comparison,null
from .pf_order_policy_inputs import FEATURES,MARKET
from .pf_order_policy_verify import reconcile
from .pf_default_pull_verify import actions

def collect(scenarios):
    populations={};table=[];statuses=[];receipts=[]
    for scenario in scenarios:
        for mode in S.MODES:
            ws=[];gs=[];fs=[];qs=[]
            for day in S.TEST:
                group='stress' if scenario=='stress' else 'build' if mode in ('reference','default') else 'eval'
                out=S.folder(day,group);c=json.loads((out/'checks.json').read_text())
                if c['status']!='PASS' or S.sha(out/'independent_windows.parquet')!=c['independent_sha']:raise ValueError('unverified independent windows')
                v=next(x for x in c['variants'] if x['mode']==mode)
                for k,h in v['hashes'].items():
                    if S.sha(out/k/S.stem(mode))!=h:raise ValueError('portfolio artifact changed')
                if S.sha(out/'generations'/f'{mode}.parquet')!=v['generation_sha']:raise ValueError('generation artifact changed')
                w,f,q=(pd.read_parquet(out/k/S.stem(mode)) for k in ('win','fills','requests'))
                g=pd.read_parquet(out/'generations'/f'{mode}.parquet');ind=pd.read_parquet(out/'independent_windows.parquet');ind=ind[ind['mode']==mode]
                reconcile(w,f,q,ind,g,288);actions(g)
                ws.append(ind);gs.append(g);fs.append(f);qs.append(q)
                receipts.append(dict(day=day,scenario=scenario,mode=mode,checks_sha=S.sha(out/'checks.json')))
            w=pd.concat(ws,ignore_index=True);g=pd.concat(gs,ignore_index=True);f=pd.concat(fs,ignore_index=True);q=pd.concat(qs,ignore_index=True)
            populations[scenario,mode]=w;valid=w.marked_strict.notna();vm=g.entry_value1.notna()&g.entry_value5.notna();spread=g.spread_value.notna()
            table.append(dict(scenario=scenario,mode=mode,windows=len(w),marked_windows=int(valid.sum()),marked_missing=int((~valid).sum()),
                marked=float(w.loc[valid,'marked_strict'].sum()),settlement=float(w.pnl.sum()),
                carry_common=float((w.loc[valid,'pnl']-w.loc[valid,'marked_strict']).sum()),settlement_common=float(w.loc[valid,'pnl'].sum()),
                value1=float(g.loc[vm,'entry_value1'].sum()/100),value5=float(g.loc[vm,'entry_value5'].sum()/100),value_known=int(vm.sum()),value_missing=int((~vm).sum()),
                spread=float(g.loc[spread,'spread_value'].sum()/100),spread_known=int(spread.sum()),spread_missing=int((~spread).sum()),
                inventory_square_s=float(w.inventory_square_s.sum()),inventory_rms=float(np.sqrt(w.inventory_square_s.sum()/(240.1*len(w)))),
                terminal_abs_mean=float(w.terminal_abs.mean()),terminal_abs_p95=float(w.terminal_abs.quantile(.95)),
                max_inventory=float(w.max_inventory.max()),settlement_day_sd=float(w.groupby('day').pnl.sum().std(ddof=1)),
                generations=len(g),filled_generations=int(g.entry_q.gt(0).sum()),fills=len(f),shares=float(f.q.sum()),
                places=int(q.kind.eq('place').sum()),cancels=int(q.kind.eq('cancel').sum()),
                model_pulls=int(g.pull.fillna(0).sum()),model_scores=int(g.score_count.fillna(0).sum()),
                order_life_p50=float(g.life_ms.median()),order_life_p90=float(g.life_ms.quantile(.9)),
                rest_side_fraction=float(g.life_ms.sum()/(len(w)*240100*2))))
            for (day,status),n in g.groupby(['day','status']).size().items():statuses.append(dict(scenario=scenario,mode=mode,day=day,status=status,n=int(n)))
    comparisons={}
    for scenario in scenarios:
        for mode in S.MODES:
            for baseline in ('reference','matched_guard','default'):
                if mode!=baseline:comparisons[f'{scenario}:{mode}-{baseline}']=comparison(populations[scenario,mode],populations[scenario,baseline])
    return table,comparisons,statuses,receipts

def diagnostics():
    import lightgbm as lgb
    fit=json.loads((S.OUT/'fit.json').read_text());path=S.OUT/'predictions.parquet'
    if S.sha(path)!=fit['predictions_sha']:raise ValueError('predictions changed')
    a=pd.read_parquet(path)
    for fold in fit['folds']:
        b=a[a.day==fold['day']]
        for model,cols in [('market',MARKET),('queue',FEATURES)]:
            p=S.OUT/'models'/f"{fold['day']}_{model}.txt"
            if S.sha(p)!=fold['models_sha'][model]:raise ValueError('model changed')
            m=lgb.Booster(model_file=str(p));score=m.predict(b[cols],num_threads=1)
            np.testing.assert_allclose(score,b[model+'_score'],atol=1e-12,rtol=0)
    a['hour']=(a.decision//3600000).astype(int)
    a['age_bin']=pd.cut(a.order_age_ms,[0,100,250,500,1000,2000,5000,np.inf],include_lowest=True).astype(str)
    a['inv_sign']=np.sign(a.inventory);out={}
    for mode in ('market','queue','queue_cost'):
        score=a['market_score' if mode=='market' else 'queue_score'];pull=score<(-.01*a.remaining if mode=='queue_cost' else 0.)
        r=null(a,pull,-a.label,['day','side','hour','age_bin','inv_sign'])
        r.update(zero_keep_and_reset=int((a.label_q.eq(0)&a.reset_q.eq(0)).sum()),
            keep_all_value_c=float(a.keep_value.sum()),reset_all_value_c=float(a.reset_value.sum()),
            selected_keep_c=float(a.loc[pull,'keep_value'].sum()),selected_reset_c=float(a.loc[pull,'reset_value'].sum()))
        out[mode]=r
    return fit,out

def run(stage='final'):
    S.check()
    if stage not in ('primary','final'):raise ValueError('report stage')
    scenarios=['X'] if stage=='primary' else ['X','stress']
    table,comparisons,statuses,receipts=collect(scenarios);fit,diag=diagnostics();predicates={}
    for mode in ('pressure','floor','market','queue','queue_cost'):
        predicates[mode]={}
        for baseline in ('reference','matched_guard'):
            rows=[comparisons[f'{s}:{mode}-{baseline}'] for s in scenarios]
            predicates[mode][baseline]=dict(all_scenarios_trading_up_exposure_down=all(r['marked_delta']>0 and r['exposure_delta']<=0 for r in rows),
                primary_day_interval_positive=rows[0]['marked_day_interval'][0]>0,stress_complete=stage=='final')
    asof=dt.datetime.now(dt.timezone.utc).isoformat()
    r=dict(as_of=asof,stage=stage,days=S.TEST,days_consumed=True,validation=False,table=table,comparisons=comparisons,
        statuses=statuses,receipts=receipts,fit=fit,diagnostics=diag,predicates=predicates,declaration_sha=S.sha(S.OUT/'declaration.json'))
    S.write(S.OUT/f'comparison_{stage}.json',r);S.write(S.REVIEW/f'comparison_{stage}.json',r)
    lines=['# Quote by default, learn when to pull',f'As of {asof}. Stage: {stage}. Six consumed test dates /1,728 windows per arm and scenario. Exploratory only.',
        '', 'Frozen ladder v3 is the operational benchmark. Matched_guard shares the new5-share JOIN/resting architecture, no ladder, and reserved +/-10 inventory range; it retains PM90/G1/G2. Comparing a model against matched_guard isolates the cancellation-policy change more closely than comparing only with the frozen strategy.',
        '', '| Timing | Policy | Windows / marked | Marked $ | Settlement $ | Carry on marked $ | 1s /5s markout $ | Spread $ | Inventory RMS | End abs mean | Orders |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for z in table:lines.append(f"| {z['scenario']} | {z['mode']} | {z['windows']} /{z['marked_windows']} | {z['marked']:.2f} | {z['settlement']:.2f} | {z['carry_common']:.2f} | {z['value1']:.2f} /{z['value5']:.2f} | {z['spread']:.2f} | {z['inventory_rms']:.3f} | {z['terminal_abs_mean']:.3f} | {z['places']} |")
    lines+=['','Marks/spread require fresh valid books. Missing populations and statuses are in the JSON; do not compare raw marked totals with different missing populations. Markout and spread are alternative valuations, not additive P&L. Rebates excluded.','',
        '| Comparison | Common marked / windows | Paired marked delta $ [day interval] | Settlement delta $ | Exposure delta % |', '|---|---:|---:|---:|---:|']
    for key,z in comparisons.items():lines.append(f"| {key} | {z['common_mark_windows']} /{z['windows']} | {z['marked_delta']:+.2f} [{z['marked_day_interval'][0]:+.2f}, {z['marked_day_interval'][1]:+.2f}] | {z['settlement_delta']:+.2f} | {z['exposure_pct']:+.3f} |")
    lines+=['','Primary: -50ms trade anchor,40/22/100ms place/cancel/notice, queueM. Stress: -250ms,80/100/250ms,queueL. Day intervals resample6UTC days1,000times; they do not correct for selection on consumed history.','',
        '| Policy | Neutral actions / pulls | Estimated local benefit c | Matched null mean c | p >= observed | Zero-fill actions |', '|---|---:|---:|---:|---:|---:|']
    for k,z in diag.items():lines.append(f"| {k} | {z['n']} /{z['kept']} | {z['observed_c']:.2f} | {z['null_mean_c']:.2f} | {z['p_ge']:.4f} | {z['zero_keep_and_reset']} |")
    lines+=['','One independent hash-selected probe per neutral order generation; unfilled outcomes included.500matched-count shuffles by day/side/hour/age/inventory sign. Local labels include effective cancellation delay and a1s wait followed by reset at the same price. Future portfolio responses are approximated locally; the sequential portfolios above evaluate the actual policy. Market control includes order age and guard prediction.','',
        'Computed criteria: '+json.dumps(S.native(predicates),sort_keys=True), '', 'No live promotion. Fill/cancel clocks and queue rules remain uncalibrated; protected dates untouched. Complete source/declaration, per-stage checks, statuses and model identities are retained.']
    p=S.HERE.parent/f'handoffs/20261111_default_pull_{stage}.md'
    with p.open('x') as f:f.write('\n'.join(lines)+'\n')
    print('REPORT',p,flush=True)

if __name__=='__main__':
    import sys
    run(*(sys.argv[1:] or ['final']))
