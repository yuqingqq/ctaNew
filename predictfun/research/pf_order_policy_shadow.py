"""Neutral extension after the reference cancel becomes effective; latent state is label-only."""
import copy
import numpy as np


def extend(window,target,others,pending,trades,start,end,rule):
    if end<=start or target['rem']<=1e-12:return []
    target=copy.copy(target);target['t_end']=end;target['_target']=True
    live=[copy.copy(o) for o in others]+[target]
    live.sort(key=lambda o:(0 if 't_end' in o and not o.get('_target') else 1,o.get('lane',o['side'])))
    arrivals=sorted([copy.copy(o) for o in pending if start<=o['t_act']<end and o.get('t_rm',end)>max(start,o['t_act'])],key=lambda o:o['t_act'])
    added=0;fills=[];side=target['side']
    for t,ts,price,amount,_ in trades:
        if t<start:continue
        if t>=end:break
        while added<len(arrivals) and arrivals[added]['t_act']<=t:
            r=arrivals[added];added+=1;i=int(np.searchsorted(window.tf,r['t_act'],side='left')-1)
            if i<0:continue
            f=window.F[i];p=r['pt']
            if (p>=f['a1_p']>=0 if side==0 else p<=f['b1_p']):continue
            ahead=window.shown(i,side,p)
            if not np.isfinite(ahead):ahead=0.
            if i+1<len(window.tf) and window.tf[i+1]-r['t_act']<=100:
                v=window.shown(i+1,side,p)
                if np.isfinite(v):ahead=max(ahead,v)
            live.append(dict(side=side,pt=p,rem=r['size'],ahead=ahead,ci=i,t0=r['t_act'],t_end=r.get('t_rm',end)))
        if ts!=side:continue
        live=[o for o in live if o.get('t_end',end)>t and o['rem']>1e-12]
        live.sort(key=lambda o:-o['pt'] if side==0 else o['pt'])
        left=amount
        for o in live:
            if rule=='M':
                ci=int(np.searchsorted(window.tf,t,side='left')-1)
                for k in range(o['ci']+1,ci+1):
                    shown=window.shown(k,side,o['pt'])
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
