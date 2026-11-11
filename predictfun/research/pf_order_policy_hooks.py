"""Opt-in policy hooks and read-only generation tracing around frozen v3."""
import collections
import copy
import math
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from .pf_order_policy_spec import HOLD_MS,BAND_C,RISK_C,CONT_MARGIN_C,OUT,TIMINGS
from .pf_order_policy_inputs import Inputs,Window,FEATURES,MARKET,micro_price
from .pf_order_policy_shadow import extend

NUMERIC=['mid','ws','generation','initial_size','t_send','t_act','decision','cancel_effective','entry_calm','entry_value1','entry_value5',
         'entry_settle','entry_q','entry_fills','label','label_q','label_value1','label_value5','known_remaining',
         'shadow_end','postcancel_rem','score','kept','entry_inventory']+FEATURES
NUMERIC=list(dict.fromkeys(NUMERIC))


class Manager:
    def __init__(self,day,timing,folder,phase,model=None):
        self.day=day;self.timing=timing;self.folder=Path(folder);self.phase=phase;self.inputs=Inputs(day);self.model=model
        self.rows=collections.defaultdict(list);self.writers={};self.windows=[];self.decision_rows=[]

    def window(self,mid,ws,rules,F,kf,trades,events,guard,value,line):
        mode=('calm_seed_continuation' if 'OLEARN' in rules else 'calm_seed') if 'OSEED' in rules else 'continuation' if 'OLEARN' in rules else 'hold_rule' if 'OHOLD' in rules else 'calm_filter' if 'OFILTER' in rules else 'reference'
        return Hook(self,mode,mid,ws,F,kf,trades,events,guard,value,line)

    def save(self,mode,rows):
        self.rows[mode].extend(rows)
        if len(self.rows[mode])>=10000:self.flush(mode)

    def flush(self,mode):
        rows=self.rows[mode]
        if not rows:return
        frame=pd.DataFrame(rows,columns=['id','day','mode','status']+NUMERIC)
        for k in NUMERIC:frame[k]=frame[k].astype(float)
        table=pa.Table.from_pandas(frame,preserve_index=False)
        if mode not in self.writers:
            p=self.folder/'generations'/f'{mode}.parquet';p.parent.mkdir(parents=True,exist_ok=True)
            self.writers[mode]=pq.ParquetWriter(p,table.schema,compression='snappy')
        self.writers[mode].write_table(table);rows.clear()

    def close(self):
        for mode in list(self.rows):self.flush(mode)
        for w in self.writers.values():w.close()
        pd.DataFrame(self.windows).to_parquet(self.folder/'independent_windows.parquet',index=False)


class Hook:
    def __init__(self,parent,mode,mid,ws,F,kf,trades,events,guard,value,line):
        self.parent=parent;self.mode=mode;self.mid=int(mid);self.ws=ws;self.w=Window(parent.inputs,mid,F,kf,ws)
        self.trades=trades;self.tt=np.array([t[0] for t in trades]);self.events=np.array(sorted({e[0] for e in events if e[3]=='D'}))
        self.guard=guard;self.value=value;self.line=line;self.records={};self.objects={};self.genfills=collections.defaultdict(list)
        self.seed_plan={};self.snapshots=[];self.holds={};self.count=0
        self.c=TIMINGS[parent.timing]
        self.wakeups=[]
        if mode in ('calm_filter','calm_seed','calm_seed_continuation') and len(self.w.rt):
            change=np.r_[True,self.w.calms[1:]!=self.w.calms[:-1]]
            self.wakeups=self.w.rt[change & (self.w.rt>=ws)&(self.w.rt<ws+60000)].tolist()

    def seed_flag(self,t,side,cur,v,line):
        if self.mode not in ('calm_seed','calm_seed_continuation') or not self.ws<=t<self.ws+60000:return False
        if v is None or not np.isfinite(v) or not line-BAND_C<=v<=line or not self.w.calm(t):return False
        if cur is None:
            self.seed_plan[(t,side)]=True;return True
        keep=bool(cur.get('_seed') and t<cur.get('t0',cur.get('t_act',-np.inf))+HOLD_MS)
        if keep:self.seed_plan[(t,side)]=True
        return keep

    def seed_size(self,t,side,normal):
        return 5. if self.seed_plan.get((t,side)) else normal

    def allow_new(self,t,side,net):
        return self.mode!='calm_filter' or (1 if side==0 else -1)*net<0 or self.w.calm(t)

    def sent(self,rec,net=0.):
        self.count+=1;gid=f'{self.parent.day}:{self.mid}:{self.mode}:{self.count}'
        rec['_gid']=gid;rec['_seed']=bool(self.seed_plan.get((rec['t_send'],rec['side'])))
        row={k:np.nan for k in NUMERIC};row.update(id=gid,day=self.parent.day,mode=self.mode,status='sent_no_cancel',mid=self.mid,ws=self.ws,
            generation=self.count,initial_size=rec['size'],t_send=rec['t_send'],t_act=rec['t_act'],side=rec['side'],order_price=rec['pt'],seed=float(rec['_seed']),entry_calm=float(self.w.calm(rec['t_send'])),entry_inventory=float(net),kept=0.)
        self.records[gid]=row;self.objects[gid]=rec

    def activated(self,o,rec):
        o['_gid']=rec['_gid'];o['_seed']=rec['_seed'];self.objects[o['_gid']]=o
        if self.records[o['_gid']]['status']=='sent_no_cancel':self.records[o['_gid']]['status']='accepted_no_cancel'

    def rejected(self,rec):self.records[rec['_gid']]['status']='postonly_refused'

    def hard(self,t,o,net=None,refs=()):
        i=self.w.index(t)
        if i<0 or not self.ws<=t<self.ws+240000:return False
        j=self.w.row(t)
        if j<0:return False
        from .pf_model_spec import SETS
        age=self.w.x[j,SETS['ALL'].index('bn_age')]
        if not np.isfinite(age) or age+t-self.w.rt[j]>1000:return False
        side=o['side'];f=self.w.F[i]
        if micro_price(f,side)!=o['pt'] or self.guard(t)[side]:return False
        v=self.value(t,side)
        if v is None or not np.isfinite(v) or v<self.line(t)-BAND_C:return False
        if any(r is not o and r.get('side')==side and r.get('pt')==o['pt'] and r.get('size',0)-r.get('kq',0)>1e-12 for r in refs):return False
        if net is not None:
            sign=1 if side==0 else -1;remaining=o['size']-o.get('kq',0)
            if sign*net>=10-1e-9:return False
            if sign*net>=0 and abs(net)+sum(max(0.,r['size']-r.get('kq',0)) for r in refs if r.get('side')==side)+o['size']>125+1e-9:return False
            if t>=self.ws+60000 and (sign*net>=0 or remaining>abs(net)+1e-9):return False
        return True

    def action(self,t,lane,p,size,cur,pending,net,refs):
        if cur is None:return p,size,False
        gid=cur.get('_gid');row=self.records[gid]
        if gid in self.holds:
            if p is None and t<self.holds[gid] and self.hard(t,cur,net,refs):return cur['pt'],cur['size'],False
            self.holds.pop(gid,None)
        if p is not None and p==cur['pt'] and abs(cur['size']-size)<1e-9:return p,size,False
        if np.isfinite(row['decision']):return p,size,False
        row['decision']=t;row['known_remaining']=cur['size']-cur.get('kq',0)
        if pending:row['status']='pending_create_cancel';return p,size,False
        if lane>=2:row['status']='rung_cancel';return p,size,False
        if p is not None:row['status']='replace_or_resize';return p,size,False
        if not 1e-9<row['known_remaining']<=5+1e-9:row['status']='size_or_known_fill';return p,size,False
        if not self.hard(t,cur,net,refs):row['status']='hard_guard_or_outside_band';return p,size,False
        v=self.value(t,lane);line=self.line(t)
        if v>line:row['status']='non_model_pull';return p,size,False
        other=sum(max(0.,r['size']-r.get('kq',0)) for r in refs if r.get('side')!=lane)
        x=self.w.features(t,cur,net,other,v,line)
        if x is None:row['status']='unknown_causal_state';return p,size,False
        row.update(x);row['status']='eligible'
        keep=self.mode=='hold_rule'
        if self.mode in ('continuation','calm_seed_continuation'):
            if self.parent.model is None:raise ValueError('missing continuation model')
            score=float(self.parent.model.predict(np.array([[x[k] for k in FEATURES]]),num_threads=1)[0]);row['score']=score
            keep=score>CONT_MARGIN_C*x['remaining']
        if keep:
            self.holds[gid]=min(t+HOLD_MS,self.ws+240000);row['kept']=1.
            return cur['pt'],cur['size'],True
        return p,size,False

    def cancelling(self,cur,t_rm):
        row=self.records[cur['_gid']]
        if row['status']=='eligible' and not np.isfinite(row['cancel_effective']):
            row['cancel_effective']=t_rm
            return self.parent.phase=='build'
        return False

    def snapshot(self,t,cur,others,pending):
        row=self.records[cur['_gid']]
        self.snapshots.append((row,copy.copy(cur),[copy.copy(o) for o in others if o is not cur and o['side']==cur['side']],
                               [copy.copy(o) for o in pending if o['side']==cur['side']],t))

    def fill(self,o,t,q):self.genfills[o['_gid']].append((t,o['side'],o['pt'],q))

    def label(self,row,target,others,pending,start):
        end=min(row['decision']+HOLD_MS,self.ws+240000)
        times=np.unique(np.r_[self.events[(self.events>row['decision'])&(self.events<end)],np.arange(row['decision']+50,end+1,50.)])
        for t in times:
            if not self.hard(t,target):end=min(end,t);break
        end+=self.c['cancel']
        lo=np.searchsorted(self.tt,start);hi=np.searchsorted(self.tt,end,side='left')
        fs=extend(self.w,target,others,pending,self.trades[lo:hi],start,end,self.c['queue'])
        row['postcancel_rem']=target['rem'];row['shadow_end']=end;row['label_q']=sum(q for _,q,_,_ in fs)
        for h in (1,5):
            value=0.;missing=False
            for t,q,p,_ in fs:
                mid=self.w.mid(t+h*1000)
                if not np.isfinite(mid):missing=True;break
                value+=(1 if target['side']==0 else -1)*q*(mid-p)
            row[f'label_value{h}']=np.nan if missing else value
        if not np.isfinite(row['label_value1']) or not np.isfinite(row['label_value5']):row['status']='missing_label_mark';return
        dq=(1 if target['side']==0 else -1)*row['label_q'];net=row['inventory']
        row['label']=.5*(row['label_value1']+row['label_value5'])-RISK_C*((net+dq)**2-net**2)

    def finish(self,rawfills,pnl,net_end,outcome):
        if sum(map(len,self.genfills.values()))!=len(rawfills):raise ValueError('generation fill count differs from engine')
        events=sorted([f for fs in self.genfills.values() for f in fs]);signed=[];cashflows=[];square=0.;last=self.ws;net=0.;maximum=0.
        for t,side,p,q in events:
            if t<last:raise ValueError('fill clock')
            square+=(t-last)/1000*net**2;last=t;sg=1 if side==0 else -1
            signed.append(sg*q);cashflows.append(-sg*q*p/100);net=math.fsum(signed)
            if abs(net)<1e-9:net=0.
            maximum=max(maximum,abs(net))
        cash=math.fsum(cashflows);stop=self.ws+240100;square+=max(0.,stop-last)/1000*net**2
        if not np.isclose(net,net_end,atol=1e-7,rtol=0) or not np.isclose(cash+net*outcome,pnl,atol=1e-7,rtol=0):raise ValueError('engine cash/inventory reconciliation')
        mark=self.w.mid(stop);marked=cash+net*mark/100 if net else cash
        self.parent.windows.append(dict(day=self.parent.day,mode=self.mode,mid=self.mid,ws=self.ws,pnl=pnl,cash=cash,net_end=net,
            marked_strict=marked,inventory_square_s=square,terminal_abs=abs(net),max_inventory=maximum,n_fills=len(events)))
        for row,target,others,pending,start in self.snapshots:self.label(row,target,others,pending,start)
        for gid,row in self.records.items():
            fs=self.genfills[gid];row['entry_q']=sum(f[3] for f in fs);row['entry_fills']=len(fs)
            for h in (1,5):
                values=[(1 if s==0 else -1)*q*(self.w.mid(t+h*1000)-p) for t,s,p,q in fs]
                row[f'entry_value{h}']=sum(values) if all(np.isfinite(values)) else np.nan
            row['entry_settle']=sum((1 if s==0 else -1)*q*(outcome*100-p) for t,s,p,q in fs)
        self.parent.save(self.mode,list(self.records.values()))
