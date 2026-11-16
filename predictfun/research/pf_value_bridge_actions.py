"""Neutral post/wait and keep/cancel labels for frozen bridge comparisons."""
from collections import Counter
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import time
import pf_value_bridge as V
import pf_flow_control_run as RUN
import pf_flow_control_checks as CHECK
import pf_flow_control_bounded as B
import pf_flow_price_data as DATA
import pf_stop_queue_activation_lane as LANE
import pf_ev_lag_gap as CLOCK
F,C,np,pd,I,P,K=V.F,V.C,V.np,V.pd,V.I,V.P,V.K
AC=C.AC


def load(day):
    raw=json.loads((C.OUT/day/'receipt.json').read_text())
    for p,h in raw['input_hashes'].items():C.require(C.sha(p)==h,'raw input changed '+p)
    reg=pd.read_parquet(Path(I.K.IO)/'step1'/day/'registry.parquet');reg=reg[reg.mid.isin(raw['selected_ids'])].sort_values('ws').copy()
    day0=I._ms(f'{day[:4]}-{day[4:6]}-{day[6:]}');reg['block']=((reg.ws*1000-day0)//7200000).astype(int)
    selected=reg.groupby('block',sort=True).head(1).mid.astype(int).tolist();C.require(len(selected)==12,'missing hour block')
    windows={int(x.mid):float(x.ws)*1000 for x in reg[reg.mid.isin(selected)].itertuples()}
    fair,_=I.fair_fix(day,windows);clock=json.loads((Path(I.K.IO)/'step1'/day/'clock_rl_d2.json').read_text())
    depth,_=I.X5.depth_arrays(I.X5.depth_hours(day),float(clock['bn_in'])+float(clock['d_dec']))
    data=I.load_day(day,set(selected),raw['imputation_offset_ms'])
    return selected,data,fair,depth,raw


def reference_candidates(W,data,day):
    with CLOCK.clocks(58.,37.):ref=AC.replay_neutral(W,size=30.,shift_ms=50.,queue='M',nr=6,cap=F.CAP,mode='hard')
    info,_=DATA.provenance(data,W);tf=np.asarray(W[0]['tf']);ws=W[0]['ws'];m=int(W[0]['m'])
    gaps=np.flatnonzero(np.diff(tf)>500.)
    coverage=AC.Coverage(trade_start_ms=ws-2050.,trade_end_ms=ws+300950.,mark_end_ms=float(tf[-1]),expected_legs=len(info),book_start_ms=float(tf[0]),book_end_ms=float(tf[-1]),
        book_gaps=tuple(AC.MissingLeg(f'{m}_{j}',float(tf[j]),float(tf[j+1]),'book_gap') for j in gaps))
    ledger=AC.Ledger(W,ref,submitted_quantity=30.,shift_ms=50.,leg_info=info,coverage=coverage);rows=[]
    def base(key,t,kind):
        s,d,g=key;order=ledger.orders[key]
        return dict(day=day,m=m,side=s,rung=d,generation=g,decision_ms=t,x_age_ms=t-order['ta'],kind=kind,
            phase=min(5,int((t-ws)//60000)+1),hash=V.PATH.keyhash(day,m,t)+f':{s}:{d}:{g}')
    for key,t in AC.generation_probes(ledger,day=day,ages_ms=C.AGES):
        row=base(key,t,'cancel')
        if t+2000.>ws+298000.:row['status']='shutdown_horizon'
        else:
            z=AC.label_action(ledger,key,t,day=day,settings=C.SETTINGS);row['status']=z['status']
            support=DATA.mark_support(W[key[0]],t)
            if row['status'] in ('eligible','eligible_estimated_timing') and support!='ok':row['status']=support
        rows.append(row)
    for key,order in ledger.orders.items():
        t=float(order['ts']);row=base(key,t,'post');support=DATA.mark_support(W[key[0]],t)
        if t+2000.>ws+298000.:row['status']='shutdown_horizon'
        elif support!='ok':row['status']=support
        elif any(tf[j]<=t+1000. and tf[j+1]>=t for j in gaps):row['status']='incomplete_book_coverage'
        else:row['status']='eligible_estimated_timing'
        rows.append(row)
    return ref,rows


def truth(W,row,reference,corrupt=False):
    t=float(row['decision_ms']);end=t+1000.;s=int(row['side']);d=int(row['rung']);ispost=row['kind']=='post'
    activation=t+58. if ispost else t-row['x_age_ms'];values=[];prefixes=[];chosen_orders=[]
    for act in (False,True):
        z=I.truncate(W,end-W[0]['ws'])
        for side,w in z.items():
            mask=np.asarray(w['anc'])-50.<=end
            for field in ('anc','ptl','shl'):w[field]=np.asarray(w[field])[mask]
        def gate(side,rung):
            return lambda k:bool(float(z[0]['tg'][k])>=t and not (ispost and act and side==s and rung==d and float(z[0]['tg'][k])==t))
        gates={(side,rung):gate(side,rung) for side in (0,1) for rung in range(6)}
        dec={(s,d):(lambda k,now,o:bool(not ispost and act and now>=t and abs(o['ta']-activation)<.001))}
        with CLOCK.clocks(58.,37.):r=LANE.sim_joint_g(z,'M',50.,F.CAP,{side:[False]*z[side]['n'] for side in (0,1)},decide=dec,gate=gates,size=30.,nr=6,mode='hard',cooldown=0.)
        C.require(not any(r['stat']['viol'].values()),'branch engine violation')
        past=Counter(x for x in r['fl'] if x[2]<=t);expected=Counter(x for x in reference['fl'] if x[2]<=t)
        if corrupt:expected[(0,0,t-1.,50.,.01)]+=1
        C.require(past==expected,'branch historical prefix differs')
        prefixes.append(Counter(x for x in r['fl'] if x[2]<(t+58. if ispost else t+37.)))
        mt,mm=AC.valid_marks(W[0],0);j=np.searchsorted(mt,end,side='right')-1
        C.require(j>=0 and end-mt[j]<=1000.,'missing common mark')
        pos=sum((1-2*side)*q for side,rung,ft,p,q in r['fl']);cash=-sum((1-2*side)*p*q for side,rung,ft,p,q in r['fl'])
        own=[o for o in r['order_states'][s,d] if abs(o['ta']-activation)<.001]
        quantity=sum(q for o in own for ft,q in o['fl'] if ft>t)
        if ispost:C.require(len(own)==int(act),'post/wait target submission identity')
        values.append(dict(wealth=cash+pos*mm[j],risk=-F.RISK*pos**2,quantity=quantity))
        chosen_orders.append(len(own))
    C.require(prefixes[0]==prefixes[1],'unavoidable branch exposure differs')
    wealth=values[1]['wealth']-values[0]['wealth'];risk=values[1]['risk']-values[0]['risk']
    return dict(actual=wealth+risk,actual_wealth=wealth,actual_risk=risk,actual_quantity=values[int(ispost)]['quantity'],label_origin='canonical_replay')


def prediction(state,index,kind,ctx):
    packed,w,mid,diag=ctx;a=state['baseline'].copy();a[index]=1. if kind=='post' else 2.
    out,q=K.evaluate(packed,state['orders'],np.stack([state['baseline'],a]),state['position'],mid,1000.)
    u=P.utility(out,state['position']);dv=u[:,1]-u[:,0];v=float(w@dv)
    wealth=float(w@(out[:,1,0]-out[:,0,0]));own=q[:,int(kind=='post'),index]
    return dict(predicted=v,predicted_wealth=wealth,predicted_risk=v-wealth,predicted_quantity=float(w@own),
        scenario_advantage_sd=float(np.sqrt(w@((dv-v)**2))),**diag)


def run(day):
    declaration=V.verify();start=time.time();out=V.OUT/('actions_'+day);C.require(not out.exists(),'action output exists');out.mkdir(parents=True)
    selected,data,fair,depth,raw=load(day);paths=V.PATH.load();candidates=[];values=[];states=[];checks={};cache={};cache_hash=None
    if day==F.TEST:
        src=F.OUT/'bounded_probe_v1';receipt=json.loads((src/'receipt.json').read_text())
        C.require(C.sha(src/'probes.parquet')==receipt['outputs']['probes.parquet'],'cached label hash')
        cache_hash=C.sha(src/'receipt.json')
        for row in pd.read_parquet(src/'probes.parquet').to_dict('records'):
            cache[row['m'],row['side'],row['rung'],row['generation'],row['decision_ms']]=row
    for W in I.pairs(data):
        m=int(W[0]['m']);ref,rows=reference_candidates(W,data,day);bytime={}
        for row in rows:
            if row['status'] in ('eligible','eligible_estimated_timing'):bytime.setdefault(row['decision_ms'],[]).append(row)
        X,_=I.market_inputs(W,fair[m],depth);prob=P.forecasts(W,X);controllers={case:V.Controller(W,paths,prob,case) for case in V.CASES};saved={}
        def observe(k,t,ST,kpos):
            if t not in bytime:return
            state=P.snapshot(W,k,t,ST,kpos)
            for row in bytime[t]:
                matches=[i for i,info in enumerate(state['info']) if info['kind']==row['kind'] and info['side']==row['side'] and info['rung']==row['rung'] and (row['kind']=='post' or abs(info['activation']-(t-row['x_age_ms']))<.001)]
                if len(matches)!=1:row['status']='not_available_in_client_state';continue
                i=matches[0]
                if row['kind']=='post':
                    a=state['baseline'].copy();a[i]=1.
                    if not P.feasible(state,a):row['status']='inventory_reservation_limit';continue
                C.require(abs(kpos)<=F.CAP+1e-7,'neutral state beyond cap')
                saved[row['kind'],row['side'],row['rung'],row['generation']]=(k,state,i)
        replay=RUN.run_lane(W,cap=F.CAP,observer=observe)
        for field in ('fl','stat','orders','cancels'):np.testing.assert_equal(replay[field],ref[field])
        frame=pd.DataFrame(rows);eligible=frame[frame.status.isin(['eligible','eligible_estimated_timing'])]
        chosen=eligible.sort_values('hash').groupby(['kind','side','phase'],sort=False).head(2);ids=set(chosen.index)
        frame['selected']=[i in ids for i in frame.index];candidates.append(frame)
        for row in chosen.to_dict('records'):
            key=(row['kind'],row['side'],row['rung'],row['generation']);k,state,index=saved[key]
            estimates={}
            for case,controller in controllers.items():
                ctx,status=controller.context(k,row['decision_ms'])
                estimates[case]=dict(evaluation_status=status)
                if ctx is not None:estimates[case].update(prediction(state,index,row['kind'],ctx))
            cached=cache.get((m,row['side'],row['rung'],row['generation'],row['decision_ms'])) if row['kind']=='cancel' else None
            if cached is not None:
                actual=dict(actual=cached['realized_advantage_cents'],actual_wealth=cached['realized_wealth_advantage_cents'],actual_risk=cached['realized_risk_advantage_cents'],actual_quantity=cached['actual_keep_quantity'],label_origin='audited_exact_identity_cache')
            else:actual=truth(W,row,ref)
            uid=f"{day}:{m}:{row['kind']}:{row['side']}:{row['rung']}:{row['generation']}"
            for case,est in estimates.items():values.append(dict(**row,uid=uid,case=case,known_inventory=state['position'],**actual,**est))
            if not checks.get(row['kind']+'_wrong_prefix_refused'):
                checks[row['kind']+'_wrong_prefix_refused']=CHECK.refused(lambda:truth(W,row,ref,True),'corrupt action prefix')
        C.guard('bridge action market');print(json.dumps(dict(day=day,m=m,selected=len(chosen),seconds=round(time.time()-start))),flush=True)
    pd.concat(candidates,ignore_index=True).to_parquet(out/'candidates.parquet',index=False);df=pd.DataFrame(values)
    C.require(not df.duplicated(['uid','case']).any(),'duplicate action evaluation');df.to_parquet(out/'values.parquet',index=False)
    C.dump(out/'receipt.json',dict(asof=C.stamp(),source_manifest=C.sha(V.FREEZE),day=day,selected_markets=selected,
        raw_receipt_sha256=C.sha(C.OUT/day/'receipt.json'),cached_labels_receipt_sha256=cache_hash,controls=checks,
        outputs={p.name:C.sha(p) for p in out.glob('*.parquet')},seconds=time.time()-start))
