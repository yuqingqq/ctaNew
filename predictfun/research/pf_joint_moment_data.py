"""Neutral action sampling, causal features and parity-gated local labels."""
import argparse
from collections import Counter
import gc
import json
from pathlib import Path
import time
import pyarrow as pa
import pyarrow.parquet as pq
import pf_joint_moment as J
import pf_flow_control_run as RUN
F,C,np,pd,I,A,V,D=J.F,J.C,J.np,J.pd,J.I,J.A,J.V,J.D


def metadata(day,training):
    raw=json.loads((C.OUT/day/'receipt.json').read_text())
    for p,h in raw['input_hashes'].items():C.require(C.sha(p)==h,'raw input changed '+p)
    reg=pd.read_parquet(Path(I.K.IO)/'step1'/day/'registry.parquet');reg=reg[reg.mid.isin(raw['selected_ids'])].sort_values('ws').copy()
    if not training:
        selected=json.loads((D.OUT/day/'receipt.json').read_text())['selected_markets'];reg=reg[reg.mid.isin(selected)]
        C.require(set(reg.mid)==set(selected),'evaluation markets changed')
    return raw,reg


def eval_rows(day):
    src=D.OUT/day;r=json.loads((src/'receipt.json').read_text())
    C.require(C.sha(src/'actual.parquet')==r['outputs']['actual.parquet'],'archived canonical labels changed')
    a=pd.read_parquet(src/'actual.parquet');a['hash']=[V.PATH.keyhash(day,m,t)+f':{s}:{d}:{g}'
        for m,t,s,d,g in a[['m','decision_ms','side','rung','generation']].itertuples(index=False,name=None)]
    return a


def compare_label(label,row):
    orientation=1. if row['kind']=='post' else -1.
    expected=dict(y_q=row['actual_affected_quantity'],y_m=orientation*row['actual_focal_motion'],
        y_w=orientation*(row['actual_focal_edge']+row['actual_focal_motion']),y_value=row['actual'],
        y_risk=row['actual_risk'],y_other=row['actual_other_edge']+row['actual_other_motion'])
    return {k:float(label[k]-v) for k,v in expected.items() if not np.isclose(label[k],v,rtol=0,atol=1e-7)}


def select_training(W,data,day):
    reference,rows=A.reference_candidates(W,data,day);a=pd.DataFrame(rows)
    eligible=a[a.status.isin(['eligible','eligible_estimated_timing'])]
    ids=set(eligible.sort_values('hash').groupby(['kind','side','phase'],sort=False).head(2).index)
    a['selected']=[i in ids for i in a.index]
    a['uid']=[f'{day}:{m}:{kind}:{s}:{d}:{g}' for m,kind,s,d,g in a[['m','kind','side','rung','generation']].itertuples(index=False,name=None)]
    return reference,a


def run(day,preflight=False):
    if not preflight:J.verify()
    C.require(day in F.TRAIN+(F.CAL,F.TEST),'undeclared day')
    training=day in F.TRAIN;start=time.time();out=J.OUT/(('preflight_' if preflight else '')+day)
    if preflight:
        attempt=1
        while out.exists():
            attempt+=1;out=J.OUT/f'preflight_{day}_attempt{attempt}'
    C.require(not out.exists(),'moment data output exists');out.mkdir(parents=True)
    raw,reg=metadata(day,training);markets=reg.mid.astype(int).tolist();archived=None if training else eval_rows(day)
    fair=depth=None
    if not preflight:
        fair,_=I.fair_fix(day,{int(r.mid):float(r.ws)*1000 for r in reg.itertuples()})
        clock=json.loads((Path(I.K.IO)/'step1'/day/'clock_rl_d2.json').read_text())
        depth,_=I.X5.depth_arrays(I.X5.depth_hours(day),float(clock['bn_in'])+float(clock['d_dec']))
    output=[];mismatches=[];controls={};candidate_statuses=Counter();writer=None;canonical_checked=0;canonical_positive=0;seen=set()
    for begin in range(0,len(markets),24):
        batch=markets[begin:begin+24];data=I.load_day(day,set(batch),raw['imputation_offset_ms'])
        for W in I.pairs(data):
            m=int(W[0]['m']);seen.add(m)
            if training:
                reference,all_rows=select_training(W,data,day);chosen=all_rows[all_rows.selected].copy()
                candidate_statuses.update({str(k):int(v) for k,v in all_rows.groupby(['kind','status']).size().items()})
                table=pa.Table.from_pandas(all_rows,preserve_index=False)
                if writer is None:writer=pq.ParquetWriter(out/'candidates.parquet',table.schema)
                writer.write_table(table)
            else:
                chosen=archived[archived.m==m].copy()
                with A.CLOCK.clocks(58.,37.):reference=A.AC.replay_neutral(W,size=30.,shift_ms=50.,queue='M',nr=6,cap=F.CAP,mode='hard')
            bytime={float(t):g for t,g in chosen.groupby('decision_ms')}
            sparse=None
            if not preflight:
                ks=[int(np.searchsorted(W[0]['tg'],t)) for t in bytime]
                sparse=J.DATA.feature_rows(W,fair[m],depth,{0:ks,1:ks})
            checked_ids=set()
            if training and (m%17==0 or m==markets[0]):
                checked_ids=set(chosen.sort_values('hash').groupby(['kind','side'],sort=False).head(1).uid)
            def observe(k,t,ST,kpos):
                nonlocal canonical_checked,canonical_positive
                if t not in bytime:return
                state=V.P.snapshot(W,k,t,ST,kpos);j=int(W[0]['jk'][k]);mid=(W[0]['P5'][j,0]+W[1]['P5'][j,0])/2.
                for row in bytime[t].to_dict('records'):
                    indices=[i for i,x in enumerate(state['info']) if x['kind']==row['kind'] and x['side']==row['side'] and x['rung']==row['rung'] and (row['kind']=='post' or abs(x['activation']-(t-row['x_age_ms']))<.001)]
                    base={name:row[name] for name in ('uid','day','m','kind','side','rung','generation','phase','decision_ms','x_age_ms','hash')}
                    status='eligible'
                    if len(indices)!=1:status='not_available_in_client_state'
                    else:
                        index=indices[0];action=state['baseline'].copy();action[index]=1. if row['kind']=='post' else 2.
                        if row['kind']=='post' and not V.P.feasible(state,action):status='inventory_reservation_limit'
                    if status!='eligible':
                        C.require(training,'evaluation availability changed');output.append(dict(base,status=status));continue
                    price=state['orders'][index,2];base.update(status=status,mid=mid,price=price,known_remaining=state['orders'][index,3])
                    label=J.fast_label(W,k,t,ST,kpos,state,index,row['kind'],day)
                    if not training:
                        diff=compare_label(label,row);canonical_checked+=1;canonical_positive+=int(row['actual_affected_quantity']>1e-9)
                        if diff:mismatches.append(dict(uid=row['uid'],kind=row['kind'],difference=diff,label=label,known_remaining=base['known_remaining']))
                    elif row['uid'] in checked_ids:
                        check=J.canonical_label(W,row,reference,mid,price,label['y_remaining']);canonical_checked+=1;canonical_positive+=int(check['y_q']>1e-9)
                        diff={key:float(label[key]-check[key]) for key in ('y_q','y_m','y_w','y_value','y_risk','y_other') if not np.isclose(label[key],check[key],rtol=0,atol=1e-7)}
                        if diff:mismatches.append(dict(uid=row['uid'],kind=row['kind'],difference=diff))
                    if not preflight:
                        market,scale=sparse[int(row['side'])][k];x=J.features(W,k,t,state,index,market,scale)
                        base.update(zip(J.FEATURES,map(float,x)))
                    output.append(dict(base,**label))
            replay=RUN.run_lane(W,cap=F.CAP,observer=observe)
            for key in ('fl','stat','orders','cancels'):np.testing.assert_equal(replay[key],reference[key])
            controls[str(m)]=dict(neutral_replay_equal=True)
            C.dump(out/'progress.json',dict(asof=C.stamp(),markets=len(seen),checked=canonical_checked,mismatches=mismatches))
            print(json.dumps(dict(stage='preflight' if preflight else 'data',day=day,markets=len(seen),seconds=round(time.time()-start),mismatches=len(mismatches))),flush=True)
            del reference,replay,sparse;gc.collect();C.guard('joint moment data')
        del data;gc.collect()
    if writer is not None:writer.close()
    C.require(seen==set(markets),'missing selected markets');frame=pd.DataFrame(output)
    C.require(not frame.uid.duplicated().any(),'duplicate action');frame.to_parquet(out/'actions.parquet',index=False)
    if not training:C.require(set(frame.uid)==set(archived.uid),'evaluation identities changed')
    C.dump(out/'receipt.json',dict(asof=C.stamp(),source_manifest=None if preflight else C.sha(J.FREEZE),day=day,markets=markets,
        action_statuses=frame.groupby(['kind','status']).size().to_dict(),candidate_statuses=dict(candidate_statuses),
        controls=controls,canonical_checked=canonical_checked,canonical_positive=canonical_positive,mismatches=mismatches,
        raw_receipt_sha256=C.sha(C.OUT/day/'receipt.json'),outputs={p.name:C.sha(p) for p in out.glob('*.parquet')},seconds=time.time()-start))
    if not preflight:C.require(not mismatches and canonical_checked>0,'canonical label parity failed')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['preflight','build']);p.add_argument('day');a=p.parse_args();run(a.day,a.mode=='preflight')
