"""Paired local continuation/reset counterfactuals; hidden state is label-only."""
import copy
import numpy as np

def paired_leg(w,target,others,pending,trades,start,end,rule,reset_at=None,reset_size=None):
    if end<=start:return []
    if rule not in ('M','L'):raise ValueError('unknown queue rule')
    live=[copy.copy(o) for o in others]
    live.sort(key=lambda o:(0 if 't_end' in o else 1,o.get('lane',o['side'])))
    side=target['side'];p=target['pt'];fills=[]
    if reset_at is None:
        o=copy.copy(target);o.update(t_end=end,_target=True);live.append(o)
    arrivals=[copy.copy(r) for r in pending if start<=r['t_act']<end and r.get('t_rm',end)>r['t_act']]
    if reset_at is not None and start<=reset_at<end:
        arrivals.append(dict(side=side,pt=p,size=reset_size,t_act=reset_at,t_rm=end,_target=True))
    arrivals.sort(key=lambda o:o['t_act']);added=0;last=start
    for t,ts,price,amount,_ in trades:
        if t<start:continue
        if t>=end:break
        if t<last or amount<0:raise ValueError('invalid trade clock/amount')
        last=t
        while added<len(arrivals) and arrivals[added]['t_act']<=t:
            r=arrivals[added];added+=1;i=int(np.searchsorted(w.tf,r['t_act'],side='left')-1)
            if i<0 or r['t_act']-w.tf[i]>1000:continue
            f=w.F[i];px=r['pt']
            if not 0<f['b1_p']<f['a1_p']<100:continue
            if px>=f['a1_p'] if side==0 else px<=f['b1_p']:continue
            ahead=w.shown(i,side,px)
            if not np.isfinite(ahead):continue
            if i+1<len(w.tf) and w.tf[i+1]-r['t_act']<=100:
                z=w.shown(i+1,side,px)
                if np.isfinite(z):ahead=max(ahead,z)
            if (px if side==0 else 100-px)*r['size']<100-1e-9:continue
            live.append(dict(side=side,pt=px,rem=r['size'],ahead=ahead,ci=i,t0=r['t_act'],
                t_end=r.get('t_rm',end),_target=r.get('_target',False)))
        if ts!=side:continue
        live=[o for o in live if o.get('t_end',end)>t and o['rem']>1e-12]
        live.sort(key=lambda o:-o['pt'] if side==0 else o['pt'])
        left=amount
        for o in live:
            if rule=='M':
                ci=int(np.searchsorted(w.tf,t,side='left')-1)
                for k in range(o['ci']+1,ci+1):
                    shown=w.shown(k,side,o['pt'])
                    if np.isfinite(shown):o['ahead']=min(o['ahead'],shown)
                o['ci']=max(o['ci'],ci)
            through=price<o['pt'] if side==0 else price>o['pt']
            if price==o['pt']:
                use=min(left,o['ahead']);o['ahead']-=use;q=min(left-use,o['rem'])
            elif through:q=min(left,o['rem']);o['ahead']=0.
            else:q=0.
            if q>1e-12:
                left-=q;o['rem']-=q
                if o.get('_target'):fills.append((t,q,o['pt'],bool(through)))
    return fills
