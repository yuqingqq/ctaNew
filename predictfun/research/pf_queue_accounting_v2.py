"""Supersede flat-position floating-point marking; preserve every original policy/fill.

Independent accounting uses math.fsum and a 1e-9-share zero tolerance. This is
output accounting only: source models, labels, decisions and selected fills stay
unchanged. The original failed verifier and uncorrected report remain on disk.
"""
import datetime as dt
import json
import math
from pathlib import Path
import subprocess
import numpy as np
import pandas as pd
from . import pf_queue_model_spec as S
from .pf_queue_model_verify import close,verify_selection
from .pf_queue_model_fit import bootstrap_day

OUT=S.ROOT/'data/predict_fun/research/queue_accounting_v2_20261111'
REVIEW=Path(__file__).resolve().parents[1]/'reviews/queue_accounting_v2_20261111'


def ledger(fills,mid,outcome):
    signed=[];cashflows=[];net=square=last=maxnet=0.
    for f in fills.sort_values(['t','side'],kind='stable').itertuples():
        if not last<=f.t<=240000 or f.q<=0:raise ValueError('bad selected fill')
        square+=(f.t-last)/1000*net**2;last=f.t
        sg=1 if f.side==0 else -1;signed.append(sg*f.q);cashflows.append(-sg*f.q*f.price)
        net=math.fsum(signed)
        if abs(net)<1e-9:net=0.
        maxnet=max(maxnet,abs(net))
    square+=(240000-last)/1000*net**2;cash=math.fsum(cashflows)
    return dict(net=net,cash_c=cash,stop_value_c=cash+net*mid if net else cash,
                settle_value_c=cash+net*100*outcome if net else cash,inventory_square_s=square,
                terminal_abs=abs(net),max_inventory=maxnet,shares=math.fsum(fills.q),fills=len(fills))


def correct_windows(rows,chosen,identity):
    result=rows.copy();groups={m:g for m,g in chosen.groupby('mid',sort=False)}
    for i,r in rows.iterrows():
        ident=identity.loc[r.mid];g=groups.get(r.mid,chosen.iloc[:0]);values=ledger(g,ident.stop_mid,ident.outcome)
        for k,v in values.items():
            # Only a previously missing mark on an economically flat inventory may change materially.
            if not (k in ('stop_value_c','settle_value_c') and pd.isna(r[k]) and values['net']==0):close(r[k],v,'v2 '+k)
            result.loc[i,k]=v
    return result


def controls():
    f=pd.DataFrame(dict(t=[1.,1.,1.,2.],side=[0,0,0,1],q=[2.,1.3333333333333333,1.6666666666666667,5.],price=[49.,49.,49.,51.]))
    v=ledger(f,np.nan,1.)
    close(v['net'],0.,'flat control');close(v['stop_value_c'],10.,'flat cash control')
    try:ledger(f.assign(q=[-1.,1.,1.,1.]),50.,1.)
    except ValueError:pass
    else:raise ValueError('negative-fill falsifier failed')
    ident=pd.DataFrame([dict(mid=1,stop_mid=np.nan,outcome=1.)]).set_index('mid');f['mid']=1
    row=pd.DataFrame([{**v,'mid':1,'stop_value_c':np.nan}]);correct_windows(row,f,ident)
    try:correct_windows(row.assign(cash_c=999.),f,ident)
    except ValueError:pass
    else:raise ValueError('cash falsifier failed')


def run():
    controls();declaration=S.check()
    if OUT.exists():raise ValueError('refuse overwrite')
    old=json.loads((S.OUT/'report.json').read_text())
    if S.sha(S.OUT/'portfolio_windows.parquet')!=old['portfolio_sha'] or S.sha(S.OUT/'predictions.parquet')!=old['prediction_sha']:raise ValueError('original artifact changed')
    original=pd.read_parquet(S.OUT/'portfolio_windows.parquet');pred=pd.read_parquet(S.OUT/'predictions.parquet')
    for fold in old['folds']:
        if max(fold['train_days'])>=fold['test_day']:raise ValueError('future training')
    for name in ('market','queue'):
        score=np.column_stack([pred[name+'_keep_score'],np.zeros(len(pred)),pred[name+'_rejoin_score']])
        close(pred[name+'_action'],np.argmax(score,axis=1),'saved action')
    output=[];selected_sha={};n_fills=0
    for day in S.DAYS[3:]:
        build=json.loads((S.OUT/f'{day}_build.json').read_text())
        for suffix,key in [('.parquet','data_sha'),('_fills.parquet','fills_sha')]:
            if S.sha(S.OUT/f'{day}{suffix}')!=build[key]:raise ValueError('generation file changed')
        rows=pd.read_parquet(S.OUT/f'{day}.parquet');potential=pd.read_parquet(S.OUT/f'{day}_fills.parquet')
        identity=rows.drop_duplicates('mid').set_index('mid')
        for config in S.CONFIGS:
            for arm in original.arm.unique():
                path=S.OUT/f'{day}_{config}_{arm}_selected.parquet';chosen=pd.read_parquet(path)
                verify_selection(chosen,potential);selected_sha[path.name]=S.sha(path);n_fills+=len(chosen)
                rr=original[(original.day==day)&(original.config==config)&(original.arm==arm)]
                output.append(correct_windows(rr,chosen,identity))
        pair=rows.merge(rows[['config','mid','episode','side','reference_admitted','keep_net']],on=['config','mid','episode'],suffixes=('','_other'))
        g=pair[(pair.side!=pair.side_other)&pair.keep_q.notna()]
        origin=g.initial_net.to_numpy()+np.where(g.reference_admitted_other,g.keep_net_other,0.)
        for action in S.ACTIONS:
            close(g[action+'_utility'],g[action+'_value']-S.RISK_C*((origin+g[action+'_net'])**2-origin**2),'neutral utility')
    r=pd.concat(output,ignore_index=True);OUT.mkdir()
    r.to_parquet(OUT/'portfolio_windows.parquet',index=False)
    report=dict(old);report['summary']=[dict(x) for x in old['summary']];report['deltas']=[]
    for s in report['summary']:
        g=r[(r.config==s['config'])&(r.arm==s['arm'])]
        s.update(stop_value_usd=float(g.stop_value_c.sum()/100),settle_value_usd=float(g.settle_value_c.sum()/100),
                 missing_stop=int(g.stop_value_c.isna().sum()),missing_settlement=int(g.settle_value_c.isna().sum()),
                 settlement_carry_usd=float((g.settle_value_c-g.stop_value_c).sum()/100),
                 inventory_square_s=float(g.inventory_square_s.sum()),terminal_abs_mean=float(g.terminal_abs.mean()))
    for config in S.CONFIGS:
        g=r[r.config==config]
        for ref in ('market','always_keep','always_cancel','legacy_score'):
            p=g[g.arm=='queue'].merge(g[g.arm==ref],on=['day','mid'],suffixes=('_q','_b'),validate='one_to_one')
            for metric in ('stop_value_c','settle_value_c','inventory_square_s','terminal_abs'):
                if p[[metric+'_q',metric+'_b']].isna().any().any():
                    report['deltas'].append(dict(config=config,ref=ref,metric=metric,status='missing_mark'));continue
                x=(p[metric+'_q']-p[metric+'_b']).groupby(p.day).sum();scale=100 if metric.endswith('_c') else 1
                report['deltas'].append(dict(config=config,ref=ref,metric=metric,sum=float(x.sum()/scale),days_positive=int((x>0).sum()),
                    per_day={str(k):float(v/scale) for k,v in x.items()},interval=bootstrap_day(x/scale)))
    receipt=dict(as_of=dt.datetime.now(dt.timezone.utc).isoformat(),source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        source_sha=S.sha(__file__),supersedes=str(S.OUT/'report.json'),supersedes_sha=S.sha(S.OUT/'report.json'),
        reason='Normalize economically flat inventory using fsum and1e-9-share tolerance, so missing prices do not erase known cash. No policy/model/label/fill changes.',
        restored_stop_marks=int(original.stop_value_c.isna().sum()-r.stop_value_c.isna().sum()),
        portfolio_windows_checked=len(r),selected_fill_records_checked=n_fills,prediction_generations_checked=len(pred),selected_sha=selected_sha,status='PASS')
    report.update(as_of=receipt['as_of'],accounting_correction=receipt,portfolio_sha=S.sha(OUT/'portfolio_windows.parquet'))
    for folder in (OUT,REVIEW):S.write(folder/'report.json',report);S.write(folder/'verification.json',receipt)
    print('ACCOUNTING_V2_VERIFIED',len(r),'windows',n_fills,'fills; restored',receipt['restored_stop_marks'],'flat marks',flush=True)


if __name__=='__main__':run()
