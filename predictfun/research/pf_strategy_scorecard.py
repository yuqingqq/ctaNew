"""Reproduce ten existing strategy/clock cases and attach common accounting.

No refits, policy changes, threshold search, new dates or inferential claims.
Every window must reproduce its named prior artifact before publication.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parent))
import pf_entry_arrival as A
import pf_entry_arrival_fit as F
import pf_strategy_metrics as Q

L,P,V,G,M,K=A.L,A.P,A.V,A.G,A.M,A.K
np,pd,lgb=A.np,A.pd,A.lgb
HERE=Path(__file__).resolve().parent
REVIEW=HERE.parent/'reviews/strategy_scorecard_20261116'
OUT=Path(K.IO)/'strategy_scorecard_20261116_v1'
CATALOG={
    'KEEP_ALL':dict(name='Always-rest ladder',old='KA',model='none',entry='Post whenever the shared ladder permits',resting='No learned cancellation; shared structural handling remains'),
    'FILL_GUARD':dict(name='Fill-probability guard',old='QR',model='STOP-EV v3 fill head only; original saved QR thresholds',entry='Wait when fresh P(fill) exceeds the saved threshold',resting='Pull when P(fill) exceeds the saved threshold'),
    'SWEEP_GUARD':dict(name='Sweep-risk guard',old='MD1000',model='STOP-MODEL v6, 1s sweep-risk heads; original saved thresholds',entry='Wait when predicted through-price risk is too high',resting='Pull the order when its depth-adjusted sweep risk is too high'),
    'ORDER_EV':dict(name='Order-state EV',old='B2',model='STOP-EV v3 order-state-only hurdle',entry='Post at nonnegative fresh EV',resting='Pull at negative resting EV'),
    'STOP_EV':dict(name='STOP-EV strategy',old='EV / baseline',model='STOP-EV v3 full 71 market +12 order inputs; P(fill) times mean filled value',entry='Post at nonnegative fresh EV',resting='Pull at negative resting EV'),
    'ARRIVAL_EV':dict(name='Arrival forecast + STOP-EV guard',old='forecast',model='Side-specific arrival residual models + frozen STOP-EV v3 guard',entry='Post when forecast of same-price arrival teacher value is nonnegative',resting='Unchanged STOP-EV negative-value pull rule'),
    'FUTURE_ENTRY':dict(name='Future-informed entry diagnostic',old='arrival_oracle',model='Future arrival teacher score + frozen STOP-EV v3 guard',entry='Post using future information; NONCAUSAL diagnostic',resting='Unchanged STOP-EV negative-value pull rule')}
CASES=[
    dict(id=f'{name}__legacy40',strategy=name,clock='legacy40',prior='latency',cfg='current',arm=arm)
    for name,arm in [('KEEP_ALL','KA'),('FILL_GUARD','QR'),('SWEEP_GUARD','MD1000'),('ORDER_EV','B2'),('STOP_EV','EV')]]+[
    dict(id=f'{name}__recorded58',strategy=name,clock='recorded58',prior='arrival',cfg=None,arm=arm)
    for name,arm in [('STOP_EV','baseline'),('ARRIVAL_EV','forecast'),('FUTURE_ENTRY','arrival_oracle')]]+[
    dict(id='STOP_EV__zero_orders',strategy='STOP_EV',clock='zero_orders',prior='latency',cfg='order_zero',arm='EV'),
    dict(id='STOP_EV__zero_all',strategy='STOP_EV',clock='zero_all',prior='latency',cfg='modeled_zero',arm='EV')]
CLOCKS={
    'legacy40':dict(placement_ms=40,cancel_ms=22,pending_cancel_ms=95,notification_ms=100,feed='recorded observation clocks'),
    'recorded58':dict(placement_ms=58,cancel_ms=37,pending_cancel_ms=95,notification_ms=100,feed='recorded observation clocks; deterministic recorded order medians'),
    'zero_orders':dict(placement_ms=0,cancel_ms=0,pending_cancel_ms=0,notification_ms=0,feed='recorded observation clocks'),
    'zero_all':dict(placement_ms=0,cancel_ms=0,pending_cancel_ms=0,notification_ms=0,feed='zero modeled feed delay; recorded events; 50ms grid remains')}


def selftest():
    checks=Q.selftest();checks.update(G.selftest())
    W=M.toy_lane_window([100.,150.],[50,50],[130.,30.])
    for w in W.values():w['yup']=1.
    with G.clocks(40.,22.):r=L.simulate(W,shift=50.)
    src={s:np.zeros(len(w['anc']),bool) for s,w in W.items()}
    v,f=Q.account(W,r,src,src,50.,'toy','toy')
    L.require(len(f)>0,'metrics engine positive control has no fills')
    caught=False
    try:Q.account(W,r,src,src,0.,'toy','bad-clock')
    except ValueError:caught=True
    L.require(caught,'unshifted anchor passed modeled fill reconciliation')
    checks.update(engine_fill_parity=True,wrong_fill_clock_refused=caught)
    return checks


def freeze():
    REVIEW.mkdir(parents=True,exist_ok=True);p=REVIEW/'freeze_v1.json';L.require(not p.exists(),'freeze exists')
    A.verify_freeze()
    sources={n:L.sha(HERE/n) for n in ['pf_strategy_metrics.py','pf_strategy_scorecard.py','pf_strategy_scorecard_report.py']}
    old=json.loads((A.REVIEW/'freeze_v2.json').read_text())
    A.dump(p,dict(asof=A.stamp(),source_commit=A.commit(),sources=sources,dependency_sources=old['dependency_sources'],
        catalog=CATALOG,cases=CASES,clocks=CLOCKS,days=list(L.DAYS),review_ms=50,trade_shift_ms=50,queue='M unknown-depth zero',size=30,rungs_per_side=6,
        population='Same 1434 first-minute windows, five consumed dates, six Oct31 boundary exclusions; all fills. Registry outcome used only for accounting.',
        valuation='Gross YES-equivalent cash/inventory; pre-fill last valid book strictly before modeled fill time; post-fill last valid book <= fill+h at100/500/1000/5000ms; terminal60.2s; remaining inventory held to recorded binary resolution.',
        missing='Missing required marks/outcome refuse completed output; stale (>1s) marks counted without outcome-based filtering. Prior loader exclusions retained with counts.',
        controls=selftest(),nulls='No inferential null, significance claim or policy selection; descriptive accounting only.',
        multiplicity=dict(existing_replay_cases=10,new_fits=0,new_thresholds=0,new_policies=0),output=str(OUT),
        limits='Gross simulated accounting; fees/rewards/impact not added. Only first-minute trading followed by passive settlement hold. M50/queueM; no new robustness or physical validation claim. Old reported markouts use unshifted anchors and are separately reconciled.'))


def verify():
    p=REVIEW/'freeze_v1.json';f=json.loads(p.read_text())
    for n,h in {**f['dependency_sources'],**f['sources']}.items():L.require(L.sha(HERE/n)==h,'frozen source changed '+n)
    L.require(f['cases']==CASES and f['catalog']==CATALOG,'case identities changed')
    return f


def run(day,limit=None):
    declaration=verify();L.require(day in L.DAYS,'undeclared day')
    out=OUT/('smoke_'+day if limit else day);L.require(not out.exists(),'output exists');out.mkdir(parents=True)
    start=time.time();checks=selftest();tracked={};counts=Counter();rows=[];ledgers=[]
    def track(p):
        p=Path(p);L.require(p.exists(),'missing input '+str(p));tracked[str(p)]=L.sha(p)
    prior_path=G.PREVIOUS/f'receipt_{day}_v3.json';track(prior_path);prior=json.loads(prior_path.read_text())
    for path,z in prior['inputs'].items():
        L.require(Path(path).exists()==z['exists'],'input existence changed')
        if z['exists']:track(path);L.require(tracked[path]==z['sha256'],'original input changed')
    paths=dict(latency=G.DATA/day/'windows.parquet',arrival=A.ROOT/'replay'/day/'windows.parquet')
    old=json.loads((G.PREVIOUS/'attribution_v3.json').read_text());track(G.PREVIOUS/'attribution_v3.json')
    ar=A.ROOT/'replay'/day/'receipt.json';track(ar);arj=json.loads(ar.read_text())
    for p in paths.values():track(p)
    L.require(tracked[str(paths['latency'])]==old['input_hashes'][day]['windows.parquet'],'prior latency windows identity')
    L.require(tracked[str(paths['arrival'])]==arj['windows_sha256'],'prior arrival windows identity')
    refs={name:pd.read_parquet(p) for name,p in paths.items()}
    refs['latency']=refs['latency'].set_index(['m','cfg','arm']);refs['arrival']=refs['arrival'].set_index(['m','arm'])
    heads,tfit=A.teacher(day);models={'EV':heads,'QR':heads,'B2':[lgb.Booster(model_file=tfit['artifacts'][f'b2_{h}']['path']) for h in ['p','v']]}
    fp=A.ROOT/'fit'/day/'receipt.json';track(fp);fit=json.loads(fp.read_text());students={}
    L.require(tracked[str(fp)]==arj['fit_sha256'],'student fit receipt changed')
    for s in (0,1):
        z=fit['heads'][str(s)];track(z['path']);L.require(tracked[z['path']]==z['sha256'],'student changed');students[s]=lgb.Booster(model_file=z['path'])
    dp=A.ROOT/'data'/day/'receipt.json';track(dp);dj=json.loads(dp.read_text())
    L.require(tracked[str(dp)]==arj['data_receipt_sha256'],'arrival target source changed')
    D,windows,imputation=A.load(day);ZERO,zstat=L.ZI.build(day,D,track)
    PA=L.SP.pred_arrays(str(Path(M.OUTD)/f'pred_fold{day}_{day}_v6.parquet'),M.HP)[0]
    for W,X,src,im in windows():
        if limit is not None and counts['windows']>=limit:break
        m=int(W[0]['m']);fresh=L.fresh_predictions(models,W,X);entry=F.fresh_scores(heads,students,W,X)
        L.require(np.array_equal(entry['baseline'],fresh['EV'],equal_nan=True),'entry baseline vector identity')
        op=A.ROOT/'data'/day/'oracle'/f'{m}.npz';track(op);L.require(tracked[str(op)]==dj['oracle_hashes'][op.name],'oracle changed')
        oracle=np.load(op)['score'];entry['arrival_oracle']=np.where(np.isfinite(oracle),oracle,entry['baseline'])
        Wz,Xz=L.ZI.window(W,ZERO[m]);fz=G.fresh(heads,Wz,Xz)
        for c in CASES:
            strategy=c['strategy'];zero=c['clock'].startswith('zero');w,x=(Wz,Xz) if c['clock']=='zero_all' else (W,X)
            ctx=L.clock(True) if zero else G.clocks(CLOCKS[c['clock']]['placement_ms'],CLOCKS[c['clock']]['cancel_ms'])
            with ctx:
                dec=gate=None
                if strategy=='SWEEP_GUARD':
                    pp={s:PA.get((m,s)) for s in (0,1)};choice=prior['md_choice']
                    dec={(s,d):L.SP.model_dec(pp[s],w[s]['best'],s,choice[0]) for s in (0,1) for d in range(P.NR)}
                    gate={(s,d):L.SP.model_gate(pp[s],d,choice[0]*choice[1]) for s in (0,1) for d in range(P.NR)}
                elif strategy!='KEEP_ALL':
                    arm={'ORDER_EV':'B2','FILL_GUARD':'QR'}.get(strategy,'EV')
                    cache=fz if c['clock']=='zero_all' else entry[c['arm']] if c['prior']=='arrival' else fresh[arm]
                    pol=L.Policy(models[arm],w,x,arm,prior['qr_choice'],Counter(),fresh=cache)
                    dec={(s,d):pol.dec(s,d) for s in (0,1) for d in range(P.NR)};gate={(s,d):pol.gate(s,d) for s in (0,1) for d in range(P.NR)}
                r=L.simulate(w,dec,gate,50.);oldvalue,live=L.value(w,r,src,im)
                oldrow=dict(**oldvalue,**{f'live{j}':v for j,v in enumerate(live)})
                ref=refs[c['prior']].loc[(m,c['cfg'],c['arm']) if c['prior']=='latency' else (m,c['arm'])]
                G.parity(oldrow,ref)
                for field in ['pnl_old','shares_young','markout1_cents','markout5_cents','pend','pend_shares','ws','e2']:
                    Q.close(oldvalue[field],ref[field],'previous output parity '+field)
                v,ledger=Q.account(w,r,src,im,50.,day,c['id'])
                Q.close(v['terminal_pnl_usd'],oldvalue['pnl'],'new accounting terminal parity');Q.close(v['shares'],oldvalue['shares'],'new accounting shares parity')
                v.update(day=day,m=m,case=c['id'],strategy=strategy,clock=c['clock'],legacy_anchor_markout1_usd=oldvalue['markout1_cents']/100,
                    legacy_anchor_markout5_usd=oldvalue['markout5_cents']/100,legacy_mean_rung_exposure=float(np.mean(live)))
                rows.append(v);ledgers.append(ledger);counts['replays']+=1;counts['prior_window_parity']+=1;counts['fills']+=len(ledger)
        counts['windows']+=1
        if counts['windows']%24==0:
            print(json.dumps(dict(day=day,counts=dict(counts),seconds=round(time.time()-start),rss_mib=K.rss_mib())),flush=True);A.E.mem_guard('strategy scorecard')
    frame=pd.DataFrame(rows);ledger=pd.concat(ledgers,ignore_index=True)
    L.require(not frame.duplicated(['day','m','case']).any(),'duplicate strategy windows')
    L.require(not ledger.duplicated(['day','m','case','side','rung','fill_seq']).any(),'duplicate fills')
    if limit is None:L.require(counts['windows']==len(refs['arrival'])//3,'missing windows')
    for p,h in tracked.items():L.require(L.sha(p)==h,'changed input '+p)
    frame.to_parquet(out/'windows.parquet',index=False);ledger.to_parquet(out/'fills.parquet',index=False)
    receipt=dict(asof=A.stamp(),source_commit=A.commit(),day=day,counts=dict(counts),checks=checks,freeze_sha256=L.sha(REVIEW/'freeze_v1.json'),
        input_hashes=tracked,outputs={p.name:L.sha(p) for p in [out/'windows.parquet',out/'fills.parquet']},loader=D['stat'],imputation=imputation,
        zero_inputs=zstat,qr_choice=prior['qr_choice'],md_choice=prior['md_choice'],smoke_only=limit is not None,seconds=time.time()-start)
    A.dump(out/'receipt.json',receipt);print(json.dumps(dict(day=day,counts=dict(counts),seconds=receipt['seconds'])),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('task',choices=['freeze','selftest','run']);p.add_argument('--day');p.add_argument('--limit',type=int);a=p.parse_args()
    if a.task=='freeze':freeze()
    elif a.task=='selftest':print(json.dumps(selftest()))
    else:run(a.day,a.limit)
