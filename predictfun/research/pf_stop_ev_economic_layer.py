"""Frozen-model, per-resting-order economic calibration experiment.

Five previously consumed chronological folds; no new LGBM fits. Calibration
uses ALL certified neutral actions on the fold's reserved earlier day. Target,
latency, features and saved models unchanged. Predictions are checked against
both saved heads. No collector tape, protected dates or other-session jobs.

Five score setups: raw, global mean offset, global exposure-scaled correction,
order-state exposure correction, and pressure x order-state correction.
The exposure scale p_fill * client-known remaining is a scale for calibration,
not a claim that every fill consumes the entire order. All residual rates are
estimated from ALL actions, including zeros. No filled-only calibration.
Order cells: side x queue fraction (<=.5, >.5, unknown) x edge (<=1c, >1c).
Pressure: joint adverse Predict/Binance depth OFI at 250ms; persistent when
also joint at 1000ms; other; unknown/stale (either age >250ms). Fixed bins,
200-action minimum, shrinkage equivalent to 500 average-exposure actions.
Order cells shrink to global; pressure cells shrink to their order cell.

Policies keep score >= -margin * known remaining, margins 0/.1/.5c per share.
Margins are declared switching-cost SENSITIVITIES, not estimated queue value.
No threshold selected after results, no timer, no portfolio/churn claim.
At each policy's counts compare raw-score selection and exact expected random
selection within day/hour/side/rung/exact known remaining. Also compare the
pressure score to order-only correction at its own counts. Analytical
expectations, no sampled null/significance test. All 15 variants reported.
Primary next-stage screen (not promotion): pressure m=.1 beats raw m=.1 and
order-state at equal counts on >=4/5 days, pooled both positive, with positive
kept local value pooled and >=4/5 days. Otherwise no policy replay is warranted
by this local screen; diagnose failure instead. Replay, if warranted, requires
separate declared pipeline and measures queue/re-entry costs explicitly.
"""
import argparse
from datetime import datetime, timezone
import gc
import hashlib
import json
from pathlib import Path
import resource
import subprocess

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from pf_stop_ev_generalization_review import validate, selftest as parity_controls

ROOT=Path('/home/yuqing/ctaNew/data/predict_fun/research/stop_ev_model_20261115')
TESTS=('20261027','20261031','20261101','20261102','20261103')
KEY=['day','m','side','rung','generation','decision_ms']
GROUP=['day','hour','side','rung','x_known_remaining']
SCORES=('raw','global','exposure','order','pressure')
MARGINS=(0.,.1,.5)
MIN_N=200; PRIOR_N=500.


def require(ok,msg):
    if not bool(ok):raise ValueError(msg)


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def guard():require(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss<2048*1024,'RSS cap')


def states(d):
    q=d.o_q_frac.to_numpy(float);edge=d.o_edge.to_numpy(float)
    require(np.isfinite(edge).all(),'nonfinite edge')
    require(d.side.isin([0,1]).all(),'invalid side')
    z=pd.DataFrame(index=np.arange(len(d)))
    z['queue']=np.where(np.isfinite(q),np.where(q<=.5,0,1),2)
    z['edge']=(edge>1).astype(int);z['side']=d.side.to_numpy(int)
    f=d[['adv_ofi250','adv_bnd_ofi250','adv_ofi1000','adv_bnd_ofi1000','pf_age','bnd_age']].to_numpy(float)
    known=np.isfinite(f).all(axis=1)&(f[:,4]>=0)&(f[:,5]>=0)&(f[:,4]<=250)&(f[:,5]<=250)
    joint=(f[:,0]>0)&(f[:,1]>0);persist=joint&(f[:,2]>0)&(f[:,3]>0)
    z['pressure']=np.where(~known,3,np.where(persist,2,np.where(joint,1,0)))
    z['order_cell']=z.side*6+z.queue*2+z.edge
    z['pressure_cell']=z.order_cell*4+z.pressure
    return z


def exposure(d):
    p=d.p_fill.to_numpy(float);q=d.x_known_remaining.to_numpy(float)
    require(np.isfinite(p).all() and ((p>=0)&(p<=1)).all(),'invalid probability')
    require(np.isfinite(q).all() and (q>0).all(),'invalid remaining')
    return p*q


def fit_layer(c):
    require(not c.duplicated(KEY).any(),'duplicate calibration action')
    y=c.value_cents.to_numpy(float);p=c.pred.to_numpy(float);e=exposure(c)
    require(np.isfinite(y).all() and np.isfinite(p).all() and e.sum()>0,'invalid calibration')
    require((np.abs(y[c.filled_quantity.to_numpy()==0])<1e-8).all(),'nonzero no-fill value')
    residual=y-p;s=states(c);global_rate=float(residual.sum()/e.sum())
    prior_exposure=PRIOR_N*float(e.mean())
    a=dict(cal_day=str(c.day.max()),n=len(c),global_offset=float(residual.mean()),
           global_rate=global_rate,prior_exposure=prior_exposure,order={},pressure={})
    for level in ['order','pressure']:
        for cell,ix in s.groupby(level+'_cell',sort=True).indices.items():
            ix=np.asarray(ix);parent=global_rate if level=='order' else a['order'][str(int(cell)//4)]['rate']
            supported=len(ix)>=MIN_N
            rate=float((residual[ix].sum()+prior_exposure*parent)/(e[ix].sum()+prior_exposure)) if supported else parent
            a[level][str(int(cell))]=dict(n=len(ix),filled=int((c.filled_quantity.to_numpy()[ix]>0).sum()),
                exposure=float(e[ix].sum()),residual_cents=float(residual[ix].sum()),rate=rate,
                parent_rate=parent,supported=supported)
    return a


def apply_layer(d,a):
    require(str(d.day.min())>a['cal_day'],'calibration must precede evaluation')
    e=exposure(d);s=states(d);p=d.pred.to_numpy(float)
    require(np.isfinite(p).all(),'invalid raw prediction')
    order=np.array([a['order'].get(str(k),dict(rate=a['global_rate']))['rate'] for k in s.order_cell])
    rate=np.array([a['pressure'].get(str(k),dict(rate=r))['rate'] for k,r in zip(s.pressure_cell,order)])
    return dict(raw=p,global_=p+a['global_offset'],exposure=p+e*a['global_rate'],order=p+e*order,pressure=p+e*rate)


def population(d):
    y=d.value_cents.to_numpy(float);f=d.filled_quantity.to_numpy(float)
    return dict(n=len(d),filled=int((f>0).sum()),good=int((y>1e-8).sum()),bad=int((y< -1e-8).sum()),
        filled_zero=int(((f>0)&(np.abs(y)<=1e-8)).sum()),value_cents=float(y.sum()),
        good_value_cents=float(np.maximum(y,0).sum()),gross_loss_cents=float(np.maximum(-y,0).sum()),
        filled_shares=float(f.sum()),at_price_cents=float(d.at_price_value_cents.sum()),through_price_cents=float(d.through_price_value_cents.sum()))


def matched(d,keep,score):
    keep=np.asarray(keep,bool);score=np.asarray(score,float)
    require(len(d)==len(keep)==len(score) and np.isfinite(score).all(),'bad matched score')
    sel=np.zeros(len(d),bool);expect=0.
    y=d.value_cents.to_numpy(float)
    for ix in d.groupby(GROUP,sort=False,dropna=False).indices.values():
        ix=np.asarray(ix);n=int(keep[ix].sum())
        sel[ix[np.argsort(-score[ix],kind='stable')[:n]]]=True
        expect+=n*float(y[ix].mean())
    require(int(sel.sum())==int(keep.sum()),'matched count')
    return sel,float(expect)


def policy_metrics(d,score,margin):
    y=d.value_cents.to_numpy(float);q=d.x_known_remaining.to_numpy(float)
    keep=np.asarray(score)>=-margin*q;raw=d.raw.to_numpy(float)
    matched_raw,expect=matched(d,keep,raw);matched_order,_=matched(d,keep,d.order)
    rawkeep=raw>=-margin*q;extra_pull=rawkeep&~keep;extra_keep=~rawkeep&keep
    terms=dict(kept=population(d.iloc[np.flatnonzero(keep)]),pulled=population(d.iloc[np.flatnonzero(~keep)]),
        extra_pulled_vs_raw=population(d.iloc[np.flatnonzero(extra_pull)]),
        extra_kept_vs_raw=population(d.iloc[np.flatnonzero(extra_keep)]))
    terms.update(margin_cents_per_share=margin,predicted_kept_cents=float(np.asarray(score)[keep].sum()),
        delta_vs_raw_cents=float(y[keep].sum()-y[rawkeep].sum()),
        matched_raw_value_cents=float(y[matched_raw].sum()),matched_order_value_cents=float(y[matched_order].sum()),
        matched_expected_value_cents=expect,selection_vs_raw_cents=float(y[keep].sum()-y[matched_raw].sum()),
        selection_vs_order_cents=float(y[keep].sum()-y[matched_order].sum()),
        selection_vs_expected_cents=float(y[keep].sum()-expect))
    return terms


def summary(d):
    y=d.value_cents.to_numpy(float);p=d.pred.to_numpy(float);f=d.filled_quantity.to_numpy()>0
    pp=d.p_fill.to_numpy();v=d.v_fill.to_numpy();keep=p>=0
    fill_term=(pp-f)*v;value_term=np.where(f,v-y,0.)
    require(np.allclose(p-y,fill_term+value_term,rtol=0,atol=1e-8),'residual accounting identity')
    loss=np.maximum(-y,0);tail=y/d.x_known_remaining.to_numpy()<=d.tail_cut.to_numpy()
    rec=dict(population=population(d),raw_nonnegative=population(d.iloc[np.flatnonzero(keep)]),
        predicted_nonnegative_value_cents=float(p[keep].sum()),fill_auc=float(roc_auc_score(f,pp)) if len(np.unique(f))==2 else None,
        mean_pred_fill=float(pp.mean()),actual_fill_rate=float(f.mean()),
        filled_head_mean_cents=float(v[f].mean()) if f.any() else None,filled_actual_mean_cents=float(y[f].mean()) if f.any() else None,
        raw_kept_error=dict(total_cents=float((p-y)[keep].sum()),fill_term_cents=float(fill_term[keep].sum()),
            value_term_cents=float(value_term[keep].sum()),interpretation='Exact decomposition, not causal attribution.'),
        calibration_tail=dict(n=int(tail.sum()),gross_loss_cents=float(loss[tail].sum()),total_gross_loss_cents=float(loss.sum()),
            raw_kept_n=int((tail&keep).sum()),raw_kept_loss_cents=float(loss[tail&keep].sum())),
        scores={})
    for name in SCORES:
        s=d[name].to_numpy(float)
        rec['scores'][name]=dict(mean_error_cents=float((s-y).mean()),mse=float(np.mean((s-y)**2)),
            policies={str(m):policy_metrics(d,s,m) for m in MARGINS})
    return rec


def state_tables(d):
    out=[]
    for key,g in d.groupby(['side','queue','edge','pressure'],sort=True):
        y=g.value_cents.to_numpy();p=g.pred.to_numpy();keep=p>=0
        out.append(dict(side=int(key[0]),queue=int(key[1]),edge=int(key[2]),pressure=int(key[3]),
            **population(g),pred_mean=float(p.mean()),realized_mean=float(y.mean()),
            raw_positive_n=int(keep.sum()),raw_positive_value=float(y[keep].sum()),
            raw_positive_pred=float(p[keep].sum()),pressure_score_mean=float(g.pressure_score.mean())))
    return out


def selftest():
    checks=parity_controls();n=600
    d=pd.DataFrame({c:np.zeros(n,dtype=int) for c in KEY+['hour','x_known_remaining']})
    d['generation']=np.arange(n);d['day']='20261023';d['x_known_remaining']=30.
    d['pred']=0.;d['p_fill']=.5;d['filled_quantity']=30.;d['value_cents']=np.r_[np.full(n//2,30.),np.full(n//2,-30.)]
    d['o_q_frac']=.2;d['o_edge']=.5
    for c in ['adv_ofi250','adv_bnd_ofi250','adv_ofi1000','adv_bnd_ofi1000']:d[c]=np.r_[np.full(n//2,-1.),np.ones(n//2)]
    d['pf_age']=10.;d['bnd_age']=10.
    a=fit_layer(d);test=d.copy();test['day']='20261024';scores=apply_layer(test,a)
    require((scores['pressure'][:300]>0).all() and (scores['pressure'][300:]<0).all(),'planted state correction missing')
    changed=test.copy();changed['value_cents']*= -100.;changed['filled_quantity']=0
    require(np.array_equal(scores['pressure'],apply_layer(changed,a)['pressure']),'application reads future outcome')
    refused=0
    for mode in ['remaining','probability','chronology','no_fill']:
        b=d.copy()
        if mode=='remaining':b.loc[0,'x_known_remaining']=0
        if mode=='probability':b.loc[0,'p_fill']=2
        if mode=='no_fill':b.loc[0,'filled_quantity']=0
        try:
            apply_layer(b,a) if mode=='chronology' else fit_layer(b)
        except ValueError:refused+=1
    require(refused==4,'bad calibrator input accepted')
    oracle,ex=matched(test,np.arange(n)<100,test.value_cents)
    require(oracle.sum()==100 and test.value_cents[oracle].sum()==3000 and abs(ex)<1e-8,'matched oracle')
    checks.update(state_correction_positive=True,outcome_blind_application=True,calibrator_refusals=refused,matched_oracle=True)
    return checks


def run(out):
    require(not out.exists(),'output exists');out.mkdir(parents=True);checks=selftest();hashes={};folds={};panels=[]
    def read(p):
        h=sha(p);require(str(p) not in hashes or hashes[str(p)]==h,'source changed');hashes[str(p)]=h
        return pd.read_parquet(p) if p.suffix=='.parquet' else json.loads(p.read_text())
    for day in TESTS:
        fit=read(ROOT/f'fit2_{day}.json');cal=fit['cal'];F=fit['features']
        require(max(fit['train_days'])<cal<day,'fold chronology')
        frames=[]
        heads={}
        for h in ['p','v']:
            meta=fit['artifacts']['full_'+h];path=Path(meta['path']);hashes[str(path)]=sha(path)
            require(hashes[str(path)]==meta['sha256'],'model hash');heads[h]=lgb.Booster(model_file=str(path))
        counts={}
        for dd in [cal,day]:
            path=ROOT/f'ev_x2_{dd}.parquet';d=read(path);p=read(ROOT/f'pred2_{day}_{dd}.parquet');validate(d,p)
            require(hashes[str(path)]==fit['inputs_sha256'][dd],'prepared input hash')
            for h,c in [('p','p_fill'),('v','v_fill')]:
                require(np.array_equal(heads[h].predict(d[F].to_numpy(np.float32)),p[c]),'saved prediction parity')
            require(np.allclose(d.value_cents,d.at_price_value_cents+d.through_price_value_cents,atol=1e-8,rtol=0),'components differ')
            for c in ['pred','p_fill','v_fill','b2']:d[c]=p[c]
            counts[dd]=dict(original=len(d),cert_est=int(d.cert_est.sum()),cert_ws=int(d.cert_ws.sum()),
                source_status=read(ROOT/f'ev_x_{dd}.json')['status'],certificate=read(ROOT/f'ev_x2_{dd}.json')['checks'])
            frames.append(d[d.cert_est].reset_index(drop=True))
        c,d=frames;a=fit_layer(c);fd=out/day;fd.mkdir();(fd/'calibrator.json').write_text(json.dumps(a,indent=2)+'\n')
        reloaded=json.loads((fd/'calibrator.json').read_text());ss=apply_layer(d,a);ss['global']=ss.pop('global_')
        rr=apply_layer(d,reloaded);rr['global']=rr.pop('global_')
        for name in SCORES:require(np.array_equal(ss[name],rr[name]),'calibrator reload')
        z=states(d)
        for name in z:d[name]=z[name]
        # Preserve the categorical pressure column; score table has a distinct name.
        d['pressure_state']=d.pressure;d['pressure_score']=ss['pressure']
        for name in SCORES:d[name]=ss[name]
        d['tail_cut']=float(np.quantile(c.value_cents/c.x_known_remaining,.01))
        rec=summary(d)
        tab=d.copy();tab['pressure']=tab.pressure_state
        tables=state_tables(tab);(fd/'states.json').write_text(json.dumps(tables,indent=2,allow_nan=False)+'\n')
        rec.update(day=day,cal_day=cal,cal_n=len(c),calibration_tail_cut_cents_per_remaining_share=float(d.tail_cut.iloc[0]),
            statuses=counts,strict=summary(d[d.cert_ws].reset_index(drop=True)),
            side={str(s):summary(d[d.side==s].reset_index(drop=True)) for s in [0,1]},
            calibration_supported_order_cells=sum(v['supported'] for v in a['order'].values()),
            calibration_supported_pressure_cells=sum(v['supported'] for v in a['pressure'].values()))
        cols=KEY+['hour','x_known_remaining','value_cents','filled_quantity','at_price_value_cents','through_price_value_cents',
            'cert_ws','pred','p_fill','v_fill','tail_cut','o_edge','o_q_frac','o_depth','o_age','queue','edge','pressure_state',
            'adv_ofi250','adv_bnd_ofi250','pf_age','bnd_age']+list(SCORES)
        d[cols].to_parquet(fd/'actions.parquet',index=False)
        cases=d[(d.raw>=0)&(d.value_cents<0)].sort_values('value_cents').head(10)
        cases[cols].to_csv(fd/'worst_raw_positive_cases.csv',index=False)
        (fd/'report.json').write_text(json.dumps(rec,indent=2,allow_nan=False)+'\n');folds[day]=rec;panels.append(d[cols])
        print(json.dumps(dict(day=day,n=len(d),raw=rec['raw_nonnegative'],
            pressure={m:z['kept']['value_cents'] for m,z in rec['scores']['pressure']['policies'].items()})),flush=True)
        del c,d,tab,frames,heads;gc.collect();guard()
    d=pd.concat(panels,ignore_index=True);pooled=summary(d);tab=d.copy();tab['pressure_score']=tab.pressure;tab['pressure']=tab.pressure_state
    (out/'states_pooled.json').write_text(json.dumps(state_tables(tab),indent=2,allow_nan=False)+'\n')
    vals=[folds[day]['scores']['pressure']['policies']['0.1'] for day in TESTS]
    criteria=dict(delta_vs_raw_positive_days=sum(z['delta_vs_raw_cents']>0 for z in vals),
        matched_order_positive_days=sum(z['selection_vs_order_cents']>0 for z in vals),
        kept_positive_days=sum(z['kept']['value_cents']>0 for z in vals),
        pooled_delta_vs_raw_cents=sum(z['delta_vs_raw_cents'] for z in vals),
        pooled_matched_order_cents=sum(z['selection_vs_order_cents'] for z in vals),
        pooled_kept_value_cents=sum(z['kept']['value_cents'] for z in vals))
    criteria['warrants_separately_declared_replay']=bool(all(criteria[k]>=4 for k in ['delta_vs_raw_positive_days','matched_order_positive_days','kept_positive_days'])
        and all(criteria[k]>0 for k in ['pooled_delta_vs_raw_cents','pooled_matched_order_cents','pooled_kept_value_cents']))
    for p,h in hashes.items():require(sha(p)==h,'input changed during run')
    result=dict(asof=datetime.now(timezone.utc).isoformat(),source_sha256=sha(__file__),
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=Path(__file__).resolve().parents[2],text=True).strip(),
        controls=checks,input_sha256=hashes,folds=folds,pooled=pooled,screen=criteria,
        strict_pooled=summary(d[d.cert_ws].reset_index(drop=True)),
        new_lgbm_fits=0,calibration_setups=4,score_setups=5,margin_sensitivities=3,
        portfolio_replay=False,queue_cost_estimated=False,peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
        interpretation='Consumed-data local economic screen. All actions, neutral continuation, no queue-cost or portfolio claim; zero new LGBM fits.')
    (out/'report.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(asof=result['asof'],screen=criteria,pooled=pooled)),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path);ap.add_argument('--selftest',action='store_true');args=ap.parse_args()
    if args.selftest:print(json.dumps(selftest()))
    else:require(args.out is not None,'--out required');run(args.out)
