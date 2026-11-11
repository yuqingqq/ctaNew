"""Independent arithmetic checks against the saved generation and portfolio artifacts."""
import datetime as dt
import json
import numpy as np
import pandas as pd
from .pf_queue_model_spec import OUT, REVIEW, DAYS, CONFIGS, RISK_C, check, sha, write


def close(actual, expected, label):
    if not np.allclose(actual,expected,atol=1e-6,rtol=1e-9,equal_nan=True):
        raise ValueError('artifact reconciliation failed: '+label)


def verify_window(row,fills,stop_mid,outcome):
    """Reconstruct from selected trade records, without using portfolio/simulator code."""
    net=cash=square=last=maxnet=shares=0.
    for f in fills.sort_values(['t','side'],kind='stable').itertuples():
        if not last<=f.t<=240000 or f.q<=0:raise ValueError('invalid selected fill')
        square+=(f.t-last)/1000*net*net;last=f.t
        signed=f.q if f.side==0 else -f.q
        net+=signed;cash-=signed*f.price;shares+=f.q;maxnet=max(maxnet,abs(net))
    square+=(240000-last)/1000*net*net
    values={'net':net,'cash_c':cash,'stop_value_c':cash+net*stop_mid,
            'settle_value_c':cash+net*100*outcome,'inventory_square_s':square,
            'terminal_abs':abs(net),'max_inventory':maxnet,'shares':shares,'fills':len(fills)}
    for k,v in values.items():close(row[k],v,k)
    if maxnet>25+1e-7:raise ValueError('inventory cap')


def verify_selection(chosen,potential):
    cols=['id','action','fill_index','t','q','through','generation','anchor','price','side']
    if chosen.empty:return
    joined=chosen[cols].merge(potential[cols].drop_duplicates(),on=cols,how='left',indicator=True)
    if len(joined)!=len(chosen) or not joined._merge.eq('both').all():raise ValueError('selected fill not in action artifact')
    if chosen.duplicated(['id','action','fill_index']).any():raise ValueError('duplicated selected fill')


def run():
    declaration=check();report=json.loads((OUT/'report.json').read_text())
    for k,n in [('prediction_sha','predictions'),('portfolio_sha','portfolio_windows')]:
        if sha(OUT/f'{n}.parquet')!=report[k]:raise ValueError('reported artifact hash mismatch')
    for fold in report['folds']:
        if max(fold['train_days'])>=fold['test_day']:raise ValueError('future training day')
    results=pd.read_parquet(OUT/'portfolio_windows.parquet');pred=pd.read_parquet(OUT/'predictions.parquet')
    if results.duplicated(['day','mid','config','arm']).any():raise ValueError('duplicate portfolio')
    if pred.id.duplicated().any():raise ValueError('duplicate prediction generation')
    for name in ('market','queue'):
        scores=np.column_stack([pred[name+'_keep_score'],np.zeros(len(pred)),pred[name+'_rejoin_score']])
        close(pred[name+'_action'],np.argmax(scores,axis=1),'saved action vs score')
    checked=selected_n=0;selected_sha={}
    for day in DAYS[3:]:
        rows=pd.read_parquet(OUT/f'{day}.parquet');potential=pd.read_parquet(OUT/f'{day}_fills.parquet')
        identity=rows.drop_duplicates('mid').set_index('mid')
        for config in CONFIGS:
            for arm in results.arm.unique():
                path=OUT/f'{day}_{config}_{arm}_selected.parquet';chosen=pd.read_parquet(path)
                selected_sha[path.name]=sha(path);verify_selection(chosen,potential)
                bymid={mid:g for mid,g in chosen.groupby('mid',sort=False)}
                rr=results[(results.day==day)&(results.config==config)&(results.arm==arm)]
                for r in rr.to_dict('records'):
                    ident=identity.loc[r['mid']]
                    verify_window(r,bymid.get(r['mid'],chosen.iloc[:0]),ident.stop_mid,ident.outcome)
                    checked+=1
                selected_n+=len(chosen)
        # Each action reward is priced against the same neutral inventory and other-side fills.
        pairs=rows.merge(rows[['config','mid','episode','side','reference_admitted','keep_net']],
                         on=['config','mid','episode'],suffixes=('','_other'))
        pairs=pairs[pairs.side!=pairs.side_other]
        valid=pairs.keep_q.notna();g=pairs[valid]
        other=np.where(g.reference_admitted_other,g.keep_net_other,0.)
        origin=g.initial_net.to_numpy()+other
        for action in ('keep','cancel','rejoin'):
            expected=g[action+'_value']-RISK_C*((origin+g[action+'_net'])**2-origin**2)
            close(g[action+'_utility'],expected,'neutral utility')
    for s in report['summary']:
        g=results[(results.config==s['config'])&(results.arm==s['arm'])]
        for field,col,factor in [('stop_value_usd','stop_value_c',100),('settle_value_usd','settle_value_c',100),
                                 ('inventory_square_s','inventory_square_s',1),('shares','shares',1)]:
            close(s[field],g[col].sum()/factor,'reported '+field)
        if s['windows']!=len(g):raise ValueError('reported population mismatch')
    receipt=dict(as_of=dt.datetime.now(dt.timezone.utc).isoformat(),source_commit=declaration['source_commit'],
                 portfolio_windows_checked=checked,selected_fill_records_checked=selected_n,
                 prediction_generations_checked=len(pred),selected_sha=selected_sha,status='PASS',
                 scope='Saved action identity, score/action consistency, chronology, neutral utility, and independent selected-fill cash/inventory/mark arithmetic. Does not calibrate venue queues.')
    write(OUT/'verification.json',receipt);write(REVIEW/'verification.json',receipt)
    print('VERIFIED',checked,'windows',selected_n,'selected fills',flush=True)


if __name__=='__main__':run()
