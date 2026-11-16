"""Frozen STOP_EV transfer to later minutes and aligned observed-maker values.

No fits or tuning. The first-minute-trained model runs continuously to
299.8 seconds, with the existing 200 ms cancellation tail ending at expiry.
All five previously consumed test dates, same 1,434 scorecard markets.
"""
import argparse
from collections import Counter
import gc
import json
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parent))
import pyarrow as pa
import pyarrow.parquet as pq
import pf_stop_ev_minutes_inputs as I
import pf_stop_ev_minutes_metrics as V
S,A,L,G,Q,M,K,P=I.S,I.S.A,I.S.L,I.S.G,I.S.Q,I.M,I.K,I.S.P
np,pd=I.np,I.pd
HERE=Path(__file__).resolve().parent
REVIEW=HERE.parent/'reviews/stop_ev_minutes_20261116'
OUT=Path(K.IO)/'stop_ev_minutes_20261116_v2'
SOURCE_NAMES=['pf_stop_ev_minutes_inputs.py','pf_stop_ev_minutes_metrics.py','pf_stop_ev_minutes.py','pf_stop_ev_minutes_report.py']
BATCH=24


def verify():
    S.verify()
    f=json.loads((REVIEW/'freeze_v2.json').read_text())
    for name,h in f['sources'].items():L.require(L.sha(HERE/name)==h,'changed source '+name)
    return f


def freeze():
    S.verify();REVIEW.mkdir(parents=True,exist_ok=True);p=REVIEW/'freeze_v2.json'
    L.require(not p.exists(),'freeze exists')
    A.dump(p,dict(asof=A.stamp(),source_commit=A.commit(),sources={n:L.sha(HERE/n) for n in SOURCE_NAMES},
        inherited_freeze_sha256=L.sha(S.REVIEW/'freeze_v1.json'),days=list(L.DAYS),
        supersedes=dict(freeze='freeze_v1.json',sha256=L.sha(REVIEW/'freeze_v1.json'),reason='One-market smoke passed all replay/accounting controls but receipt serialization rejected NaN for the absent non-hour-boundary subgroup. Missing subgroup diagnostics now serialize as null. No policy, features, fills, or valuation change. Version 1 smoke artifacts retained.'),
        strategy='STOP_EV corrected v3; unchanged heads and sign thresholds',
        execution=dict(place_ms=58,cancel_ms=37,pending_cancel_ms=95,notice_ms=100,review_ms=50,
            trade_shift_ms=50,queue='M unknown-depth zero',rungs_per_side=6,shares=30,
            quote_end_ms=299800,terminal_ms=300000,inventory_cap=None,cooldown_ms=0),
        population='Same 1434 scorecard markets, six Oct31 boundary exclusions; all fills with statuses; no new dates.',
        makers='Per-day rank by raw filled shares in 0<=t_rel<240 on the same eligible markets, top five vs all other observed makers. Not profit-ranked. All depths.',
        maker_clock='Same inherited group anchor/imputation and M50 as simulation. Real executed quantities; assumed execution time, not a validated match-time observation.',
        markouts=dict(horizons_ms=list(V.HORIZONS),expiry_censor='Fill+h beyond market expiry has missing value and explicit censored quantity; no endpoint substitution.',
            midpoint='Last valid public midpoint strictly before fill for execution edge; at or before fill+h for markouts; stale >1s counted.'),
        partitions=['minute','day','maker_group','depth_group','WS/E2','imputed','order_age_under100ms'],
        controls=V.selftest(),
        mandatory=['first-minute input bit identity','60.2s control replay/ledger parity','continuous-prefix parity to59.2s',
            'engine-fill identity','minute inventory and value reconciliation','raw-source markout audit','positive and known-bad controls'],
        nulls='Descriptive transfer/comparison only; no inference, selection, profitability or promotion claim.',
        multiplicity=dict(new_models=0,new_thresholds=0,primary_strategy_cases=1,control_replays=1),
        limits='Later minutes are outside first-minute training-phase coverage. Public maker fill value excludes taker/hedge/reward/cost and starting inventory. Simulated vs observed fill quantities remain different populations. Fill realism unvalidated.',output=str(OUT)))


def parity_features(W,X,reference):
    count=0
    for s in (0,1):
        ref=reference[(int(W[0]['m']),s)];cur=X[s];n=len(ref[0])
        L.require(n<=1200 and np.array_equal(cur[0][:n],ref[0]),'feature grid parity')
        L.require((cur[0][n:1200]<0).all(),'unexpected first-minute model row')
        nr=len(ref[1])
        for field in (1,2,3):
            L.require(np.array_equal(cur[field][:nr],ref[field],equal_nan=True),f'first-minute feature identity side{s} field{field}')
        count+=nr
    return count


def run_policy(W,X,heads):
    with G.clocks(58.,37.):
        fresh=G.fresh(heads,W,X)
        pol=L.Policy(heads,W,X,'EV',(0.,1.),Counter(),fresh=fresh)
        dec={(s,d):pol.dec(s,d) for s in (0,1) for d in range(P.NR)}
        gate={(s,d):pol.gate(s,d) for s in (0,1) for d in range(P.NR)}
        return L.simulate(W,dec,gate,50.)


def check_control(W,r,D,day,old,oldfills):
    src={};im={}
    for s in (0,1):
        g=D['LT'].iloc[D['LI'].get((int(W[0]['m']),s),np.zeros(0,int))]
        g=g.iloc[np.lexsort(((-g.ptick.to_numpy() if s==0 else g.ptick.to_numpy()),g.anc.to_numpy()))]
        src[s]=g.ws_timed.to_numpy(bool);im[s]=g.imputed.to_numpy(bool)
    v,f=Q.account(W,r,src,im,50.,day,'STOP_EV')
    for field,value in v.items():Q.close(value,old[field],'control '+field)
    keys=['side','rung','fill_ms','price_cents','shares']
    L.require(Counter(f[keys].itertuples(index=False,name=None))==Counter(oldfills[keys].itertuples(index=False,name=None)),'first-minute exact fill ledger parity')
    return len(f)


def raw_audit(f,W):
    """Recompute from raw window frames and long Up/Down payout accounting."""
    if not len(f):return
    times,mids,_,_=V.book(W);t=f.fill_ms.to_numpy();q=f.shares.to_numpy();side=f.side.to_numpy()
    p=f.price_cents.to_numpy()/100;sg=1-2*side;expiry=W[0]['ws']+300000
    pre_idx=np.searchsorted(times,t,side='left')-1
    pre=np.where(pre_idx>=0,mids[np.maximum(pre_idx,0)],np.nan)
    L.require(np.allclose(f.pre_mid_cents,pre,atol=1e-9,rtol=0,equal_nan=True),'raw pre-fill provenance')
    L.require(np.allclose(f.spread_capture_usd,q*sg*(pre/100-p),atol=1e-8,rtol=0,equal_nan=True),'raw execution edge')
    for h in V.HORIZONS:
        ix=np.searchsorted(times,t+h,side='right')-1
        valid=(t+h<=expiry)&(ix>=0)
        expected=np.where(valid,q*sg*(mids[np.maximum(ix,0)]/100-p),np.nan)
        L.require(np.allclose(f[f'markout_{h}_usd'],expected,atol=1e-8,rtol=0,equal_nan=True),'raw horizon value')
        L.require(np.array_equal(f[f'markout_{h}_status']=='expiry_censored',t+h>expiry),'raw expiry coverage')
    y=W[0]['yup'];cost=np.where(side==0,p,1-p);payout=np.where(side==0,y,1-y)
    L.require(np.allclose(f.settlement_pnl_usd,q*(payout-cost),atol=1e-8,rtol=0),'independent Up/Down payout')
    term=mids[np.searchsorted(times,expiry,side='right')-1]
    L.require(np.allclose(f.terminal_pnl_usd,q*sg*(term/100-p),atol=1e-8,rtol=0),'raw terminal value')


def audit_controls(f,W):
    if not len(f):return False
    raw_audit(f,W)
    bad=f.copy();bad.loc[bad.index[0],'settlement_pnl_usd']+=1;caught=False
    try:raw_audit(bad,W)
    except ValueError:caught=True
    L.require(caught,'corrupted settlement not refused')
    bad=f.copy();bad.loc[bad.index[0],'markout_1000_status']='expiry_censored' if f.markout_1000_status.iloc[0]!='expiry_censored' else 'ok'
    caught=False
    try:raw_audit(bad,W)
    except ValueError:caught=True
    L.require(caught,'corrupted horizon status not refused')
    return True


def run(day,limit=None):
    freeze=verify();L.require(day in L.DAYS,'undeclared date');start=time.time()
    out=OUT/('smoke_'+day if limit else day);L.require(not out.exists(),'output exists');out.mkdir(parents=True)
    inputs={};counts=Counter();statuses=[];minutes=[];inventories=[];features=[];writer=None;controls=V.selftest()
    def track(p):
        p=Path(p);L.require(p.exists(),'missing input '+str(p));inputs[str(p)]=L.sha(p)
    prior=S.OUT/day;receipt_path=prior/'receipt.json';track(receipt_path);pr=json.loads(receipt_path.read_text())
    for path,h in pr['input_hashes'].items():track(path);L.require(inputs[path]==h,'changed inherited input')
    for name,h in pr['outputs'].items():track(prior/name);L.require(inputs[str(prior/name)]==h,'changed scorecard output')
    old=pd.read_parquet(prior/'windows.parquet',filters=[('case','==','STOP_EV__recorded58')]).set_index('m')
    oldfills=pd.read_parquet(prior/'fills.parquet',filters=[('case','==','STOP_EV__recorded58')]);oldgroups=oldfills.groupby('m').indices
    regpath=Path(K.IO)/'step1'/day/'registry.parquet';track(regpath);reg=pd.read_parquet(regpath).set_index('mid')
    L.require(reg.index.is_unique,'registry duplicate')
    selected=sorted(map(int,old.index),key=lambda m:float(reg.loc[m,'ws']))
    if limit:selected=selected[:limit]
    L.require(all(float(reg.loc[m,'ws'])*1000>=M.boundary_cut(day) for m in selected),'boundary population')
    W0={m:float(reg.loc[m,'ws'])*1000 for m in selected}
    fitpath=Path(S.V.OUTD)/f'fit3_{day}.json';track(fitpath);heads,fit=A.teacher(day)
    for h in ['p','v']:track(fit['artifacts'][f'full_{h}']['path'])
    reference=P.market_arrays(day)
    fair,fair_stats=I.fair_fix(day,W0)
    import pf_io as IO
    hours=I.X5.depth_hours(day)
    for hour in hours:
        for path in [Path(IO.BN_DIR)/(hour+'.csv.gz'),Path(IO.REF_RUN)/'reference_prices'/(hour+'.jsonl.gz'),Path(I.X5.DEPTH_DIR)/(hour+'.csv.gz')]:
            if path.exists():track(path)
    clock=json.loads((Path(K.IO)/'step1'/day/'clock_rl_d2.json').read_text())
    depth,depth_stats=I.X5.depth_arrays(hours,float(clock['bn_in'])+float(clock['d_dec']))
    L.require(depth is not None,'depth source unavailable')
    raw=pd.read_parquet(Path(K.IO)/'step1'/day/'legs_timed2.parquet',columns=['mid','signer','shares','t_rel'])
    raw=raw[raw.mid.isin(old.index)];groups=V.rank_makers(raw);del raw
    print(json.dumps(dict(day=day,phase='inputs_ready',seconds=round(time.time()-start),rss_mib=K.rss_mib())),flush=True)
    for offset in range(0,len(selected),BATCH):
        batch=selected[offset:offset+BATCH]
        D=I.load_day(day,set(batch),pr['imputation']['imp_off_ms'])
        seen=[]
        for W in I.pairs(D):
            m=int(W[0]['m']);seen.append(m)
            L.require(reg.loc[m,'outcome'] in ('Up','Down') and int(reg.loc[m,'up_index'])==1,'registry payout identity')
            X,fs=I.market_inputs(W,fair[m],depth)
            counts['feature_rows_first_minute_parity']+=parity_features(W,X,reference)
            features.extend(dict(day=day,m=m,**z) for z in fs)
            W60=I.truncate(W);control=run_policy(W60,X,heads)
            of=oldfills.iloc[oldgroups.get(m,np.zeros(0,int))]
            counts['control_fills']+=check_control(W60,control,D,day,old.loc[m],of)
            counts['first_minute_replay_parity']+=1
            r=run_policy(W,X,heads)
            cut=W[0]['ws']+59200.
            L.require(Counter(x for x in r['fl'] if x[2]<=cut)==Counter(x for x in control['fl'] if x[2]<=cut),'continuous common-prefix fills')
            L.require([x for x in r['log'] if x[0]<=cut]==[x for x in control['log'] if x[0]<=cut],'continuous common-prefix decisions')
            counts['continuous_prefix_parity']+=1
            inv=V.minute_inventory(W,r);inventories.extend(dict(day=day,m=m,**z) for z in inv)
            ours=V.engine_fills(W,r,D,day);makers=V.maker_fills(W,D,day,groups)
            f=pd.concat([ours,makers],ignore_index=True)
            for col in ['side','rung','fill_seq','row_id','signer','m']:f[col]=f[col].astype('int64')
            for col in ['fill_ms','anchor_ms','price_cents','shares','age_ms']:f[col]=f[col].astype(float)
            for col in ['ws_timed','imputed','cancel_pending','through']:f[col]=f[col].astype(bool)
            f=V.value_fills(f,W)
            raw_audit(f,W);counts['audited_fills']+=len(f)
            if not counts['audit_control']:counts['audit_control']=int(audit_controls(f,W))
            term=M.CL.portfolio_terminal(r,*W[0]['tf_mff']);mine=f[f.case=='STOP_EV']
            Q.close(mine.terminal_pnl_usd.sum(),term['pnl'],'engine/ledger terminal')
            Q.close(mine.settlement_pnl_usd.sum(),term['cash']+term['position']*W[0]['yup'],'engine/ledger settlement')
            for case in ['STOP_EV','TOP5','OTHER_MAKERS']:
                cf=f[f.case==case]
                for minute in range(1,6):minutes.append(dict(day=day,m=m,case=case,minute=minute,**V.summarize_fills(cf[cf.minute==minute])))
            # Explicit types keep empty/sparse side partitions from changing parquet schemas.
            for col in ['side','rung','fill_seq','row_id','signer','m','minute']:f[col]=f[col].astype('int64')
            for col in ['ws_timed','imputed','cancel_pending','through']:f[col]=f[col].astype(bool)
            table=pa.Table.from_pandas(f,preserve_index=False)
            if len(f):
                if writer is None:writer=pq.ParquetWriter(out/'fills.parquet',table.schema,compression='zstd')
                writer.write_table(table)
            counts['windows']+=1;counts['strategy_fills']+=len(ours);counts['maker_legs']+=len(makers)
            if counts['windows']%12==0:
                print(json.dumps(dict(day=day,counts=dict(counts),seconds=round(time.time()-start),rss_mib=K.rss_mib())),flush=True)
                A.E.mem_guard('later minutes')
        L.require(set(seen)==set(batch),'missing batch market')
        statuses.append(dict(batch=offset,**D['stat']))
        del D;gc.collect()
    if writer is not None:writer.close()
    L.require(counts['windows']==len(selected) and counts['audit_control']==1,'incomplete run')
    pd.DataFrame(minutes).to_parquet(out/'minute_values.parquet',index=False)
    pd.DataFrame(inventories).to_parquet(out/'inventory.parquet',index=False)
    pd.DataFrame(features).to_parquet(out/'feature_support.parquet',index=False)
    for path,h in inputs.items():L.require(L.sha(path)==h,'input changed during run')
    outputs={p.name:L.sha(p) for p in out.glob('*.parquet')}
    A.dump(out/'receipt.json',dict(asof=A.stamp(),day=day,source_commit=A.commit(),freeze_sha256=L.sha(REVIEW/'freeze_v2.json'),
        input_hashes=inputs,outputs=outputs,counts=dict(counts),controls=controls,loader_batches=statuses,
        top5_ids=[s for s,g in groups.items() if g=='TOP5'],selected_markets=selected,
        imputation=pr['imputation'],fair=fair_stats,depth=depth_stats,seconds=time.time()-start,smoke_only=limit is not None))
    print(json.dumps(dict(day=day,counts=dict(counts),seconds=round(time.time()-start))),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('task',choices=['selftest','freeze','run']);p.add_argument('--day');p.add_argument('--limit',type=int);a=p.parse_args()
    if a.task=='selftest':print(json.dumps(V.selftest()))
    elif a.task=='freeze':freeze()
    else:run(a.day,a.limit)
