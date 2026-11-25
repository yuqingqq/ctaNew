"""One-day passive inventory controller comparison. Read protocol.md."""
import argparse
import bisect
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'take_exit_venue_20261125'))
import vm_replay as VM
import pf_mt_decision as BTC

MM, X, LB, BE = VM.MM, VM.X, VM.LB, VM.BE
OUT = X.DATA / 'leftover_inventory_20261125'
CENSUS = MM.OUT2 / 'census.json'
ARMS = ('A', 'B', 'C')
TRAIN = ('20261024', '20261025')
EVAL = '20261026'
EPS = 1e-8


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as f:
        json.dump(obj, f, allow_nan=False, separators=(',', ':'))


def max_new(signed, reserved, capacity, unit):
    return max(0, math.floor(min(220, 440 - reserved, 1000 - signed - reserved,
                                 capacity + 50 / unit - signed - reserved) + EPS))


def key_for(side, left_ms, flow, blocked):
    time_band = min(5, max(0, int((left_ms - 1e-7) // 50000)))
    return (side, time_band, 0 if flow < EPS else 1 if flow <= 220 else 2, int(blocked))


def groups(key):
    side, left, flow, blocked = key
    return ['/'.join(map(str, parts)) for parts in [
        (side, left, flow, blocked), (side, left, flow), (side, left)]]


def forecast(model, key):
    for level, name in enumerate(groups(key)):
        group = model.get(name)
        if group and group['markets'] >= 8:
            return group['q25'], level
    return 0., 3


class Branch(X.FixedBranch):
    """Cache only delivered fill notices; all order and venue transitions are inherited."""
    def __init__(self, path, end, ident):
        super().__init__(path, end, ident, False, 'ASYNC_FAST10')
        self.prof = dict(VM.VM_PROFILE)
        self.E = VM.VMEngine(path, end, 200., self.prof, 'L')
        self.q0, self.cap = 220., 1000.
        self.inv = dict(self.inv, Qside=440., S=500., R=250.)
        self.cursor = 0
        self.by_gid = defaultdict(float)
        self.times = [[], []]
        self.sums = [[0.], [0.]]
        self.clock = -math.inf

    def consume(self, g):
        assert g >= self.clock
        self.clock = g
        fills = self.E.fills
        while self.cursor < len(fills) and fills[self.cursor][0] + BE.FILL_NOTICE <= g:
            t, gid, s, q, _, _ = fills[self.cursor]
            self.by_gid[gid] += q
            self.times[s].append(t + BE.FILL_NOTICE)
            self.sums[s].append(self.sums[s][-1] + q)
            self.cursor += 1

    def notices(self, g):
        self.consume(g)
        super().notices(g)

    def known_filled(self, gid, g):
        return self.by_gid.get(gid, 0.)

    def reserve(self, g):
        position = self.sums[0][-1] - self.sums[1][-1]
        return (position, *(sum(self.E.o[i]['size'] - self.by_gid.get(i, 0.) for i in self.unclosed[s]) for s in (0, 1)))

    def flow(self, s, g):
        return self.sums[s][-1] - self.sums[s][bisect.bisect_right(self.times[s], g - 30000.)]


class Policy(LB.LadderInventory):
    def __init__(self, branch, on, trigger, arm, model):
        super().__init__(branch)
        self.b, self.on, self.trigger, self.arm, self.model = branch, on, trigger, arm, model
        self.opened = None
        self.first_side = None
        self.proposals = []
        self.limits = Counter()

    def context(self, k, s):
        g = k * BE.STEP
        return key_for(s, self.b.end - g, self.b.flow(1-s, g), self.on[1-s][k])

    def live(self, v, order):
        if self.arm != 'C':
            action, why = super().live(v, order)
            if action == 'PULL':
                return action, why
        if self.arm != 'A' and self.on[v['side']][self.b.k]:
            return 'PULL', 'pressure'
        return 'KEEP', 'keep'

    def slot(self, v, rung, price):
        s, k, g = v['side'], self.b.k, v['t']
        if not (0 <= v['bid'] < v['ask'] <= 100 and 0 <= g-v['received'] <= 1000):
            return 'WAIT', None, 'book'
        if self.opened is None or self.opened == k:
            if not self.trigger[1-s][k] or (self.first_side is not None and s != self.first_side):
                return 'WAIT', None, 'opening'
        if self.arm != 'A' and self.on[s][k]:
            return 'WAIT', None, 'pressure'
        signed = (1-2*s) * v['position']
        reserved = v['B'] if s == 0 else v['A']
        if self.arm == 'C':
            key = self.context(k, s)
            cap, level = forecast(self.model, key)
            mid = (v['bid']+v['ask']) / 200
            unit = mid if s == 0 else 1-mid
            q = max_new(signed, reserved, cap, unit)
            plain = max(0, math.floor(min(220, 440-reserved, 1000-signed-reserved)+EPS))
            self.limits['risk_reduced_slots'] += int(q < plain)
            if not q:
                return 'WAIT', None, 'risk'
        else:
            action, q, why = super().slot(v, rung, price)
            if action == 'WAIT':
                return action, q, why
        # Native complementary buy representation for all quotes. Passive fees zero.
        if q * (price if s == 0 else 100-price) < 100-EPS:
            return 'WAIT', None, 'buy_min'
        if g + self.b.prof['place'] >= self.b.end:
            return 'WAIT', None, 'near_end'
        if self.opened is None:
            self.opened, self.first_side = k, s
        if self.arm == 'C':
            self.proposals.append((g,s,cap,level,max(0.,signed+reserved+q),unit))
        return 'POST', q, 'post'


def triggers(cache, edges):
    n = len(cache['x'][0])
    on = MM.TR.on_arrays(cache, edges, n)
    tr = MM.TR.triggers(on)
    for s in (0,1):
        good = cache['status'][s] == 0
        prefix = np.r_[0, np.cumsum(~good)]
        i = np.arange(n)
        tr[s] &= (i >= 40) & ((prefix[i+1]-prefix[np.maximum(i-40,0)]) == 0)
    return on, tr


def one(path, info, ident, flags, arm, model, z, payout, parity=False):
    b = Branch(path, info['end_rel'], ident)
    on, tr = flags
    policy = Policy(b, on, tr, arm, model)
    both = 0
    anchors = []
    for k in range(LB.n_ticks(b)):
        LB.step(b,k,policy)
        g = k * BE.STEP
        if parity and k % 20 == 0:
            assert np.allclose(b.reserve(g), (sum((1-2*s)*q for t,i,s,q,p,it in b.E.fills if t+BE.FILL_NOTICE<=g),
                *(sum(b.E.o[i]['size']-LB.LadderBranch.known_filled(b,i,g) for i in b.unclosed[s]) for s in (0,1))), atol=1e-7)
        live = {s for s in (0,1) for i in b.unclosed[s] if not b.E.o[i]['refused'] and
                b.E.o[i]['act']<=g<b.E.o[i]['remove'] and b.E.o[i]['rem']>EPS}
        both += len(live)==2
        if k % 100 == 0:
            for s in (0,1):
                anchors.append((g,s,policy.context(k,s),None if z is None or not np.isfinite(z[k]) else float(z[k])))
    LB.finish(b)
    fills = b.E.fills
    cash,pos,settled = b.E.value(100*payout)
    fifo = X.fifo(fills,100*payout)
    peak = max(abs(np.cumsum([0.]+[(1-2*s)*q for t,i,s,q,p,it in fills])))
    assert peak<=1000+EPS and abs(settled-fifo['completed_c']-fifo['residual_c']) < 1e-5
    assert all(it>=0 and b.E.o[i]['act']<=t<b.E.o[i]['remove'] for t,i,s,q,p,it in fills)
    if b.E.o:
        k = policy.opened
        assert k >= 40 and tr[1-policy.first_side][k] and all(o['side']==policy.first_side for o in b.E.o if o['submit']==k*BE.STEP)
    times = [[t for t,i,fs,q,p,it in fills if fs==s] for s in (0,1)]
    sums = [np.cumsum([0.]+[q for t,i,fs,q,p,it in fills if fs==s]) for s in (0,1)]
    def future(g,s):
        return float(sums[s][-1]-sums[s][bisect.bisect_right(times[s],g)])
    labels = [[list(key),future(g,1-s),zv] for g,s,key,zv in anchors]
    prop = []
    if arm=='C':
        assert len(policy.proposals)==len(b.E.o)
        for g,s,cap,level,exposure,unit in policy.proposals:
            actual=future(g,1-s)
            prop.append([cap,actual,level,exposure,unit])
    metrics = dict(settle=settled/100, turnover=fifo['completed_c']/100, residual=fifo['residual_c']/100,
                   marked=(cash+pos*b.mark)/100, end_pos=pos, peak=float(peak), both=both/LB.n_ticks(b),
                   shares=float(sum(a[-1] for a in sums)), matched=fifo['matched_shares'], posts=len(b.E.o),
                   pressure_pulls=sum(c[7:] == ('PULL','pressure') for c in b.cmd_log),
                   all_pulls=sum(o['req'] is not None for o in b.E.o), first_s=None if policy.opened is None else
                   (ident['start_abs_ms']-ident['opening_ms']+policy.opened*BE.STEP)/1000,
                   risk_reduced_slots=policy.limits['risk_reduced_slots'])
    return dict(metrics=metrics, anchors=labels, proposals=prop)


def run(phase):
    cen=json.loads(CENSUS.read_text())['rows']
    days = TRAIN if phase=='train' else (EVAL,)
    model = {} if phase=='train' else json.loads((HERE/'fit.json').read_text())['cells']
    if phase=='eval':
        import subprocess
        assert subprocess.check_output(['git','show','HEAD:'+str((HERE/'fit.json').relative_to(HERE.parents[2]))]) == (HERE/'fit.json').read_bytes()
    edges=MM.TR.T.P.frozen_edges()
    src=MM.KP.Tape()
    for day in days:
        reg=MM.registry(day).set_index('mid')
        rg=pd.read_parquet(MM.S1/day/'registry.parquet').set_index('mid')
        off=src.offset(day)
        idx={int(r['mid']):r for r in cen if r['day']==day}
        assert len(idx)==len(reg)==288
        btc=None if phase=='train' else BTC.load_btc(X.DATA/'maker_trace_v2_20261123/btc',day,Counter())
        for h,group in reg.groupby('hour'):
            want=sorted((int(m) for m in group.index if idx[m]['status']=='complete'),key=lambda m:(reg.loc[m,'ws'],m))
            if phase=='train':
                want=want[:1]
            want=[m for m in want if not (OUT/phase/f'{day}_{m}.json').exists()]
            if not want: continue
            for m,st,p,info in MM.worlds(day,want,src,off,reg):
                assert st=='complete' and idx[m]['tape_id']==BE.tape_id(p['frames'],p['trades'],p['mark'])
                ws=float(reg.loc[m,'ws'])*1000
                ident=MM.ident_of(day,m,ws,info,p)
                cache=MM.make_cache(info['t_start'],info['end_rel'])
                flags=triggers(cache,edges)
                n=len(cache['x'][0]); t=info['t_start']+BE.STEP*np.arange(n)
                z=None if btc is None else BTC.btc_feats(btc,np.floor(t-1000).astype(np.int64),np.full(n,int(ws)),np.full(n,int(ws)+300000))['z']
                payout=VM.V.payout_of(rg.loc[m])
                data={a:one(p,info,ident,flags,a,model,z,payout,h%6==0) for a in (('B',) if phase=='train' else ARMS)}
                write(OUT/phase/f'{day}_{m}.json',dict(source=sha(Path(__file__)),mid=m,day=day,hour=int(h),arms=data))
            print(phase,day,h,len(want),flush=True)


def fit():
    census=json.loads(CENSUS.read_text())['rows']
    buckets=defaultdict(list)
    markets=defaultdict(set)
    inputs={}
    expected=set()
    for day in TRAIN:
        reg=MM.registry(day).set_index('mid')
        for h,grp in reg.groupby('hour'):
            ok=[r['mid'] for r in census if r['day']==day and r['hour']==h and r['status']=='complete']
            if ok:
                expected.add((day,min(ok,key=lambda m:(reg.loc[m,'ws'],m))))
    used=set()
    for path in sorted((OUT/'train').glob('*.json')):
        d=json.loads(path.read_text()); m=d['mid']; assert d['source']==sha(Path(__file__))
        used.add((d['day'],m))
        inputs[path.name]=sha(path)
        for key,y,_ in d['arms']['B']['anchors']:
            for name in groups(key):
                buckets[name].append(y); markets[name].add(m)
    assert used==expected and all(d in TRAIN for d,m in used)
    cells={name:dict(q25=float(np.quantile(vals,.25,method='lower')), rows=len(vals), markets=len(markets[name])) for name,vals in buckets.items()}
    write(HERE/'fit.json',dict(days=TRAIN,inputs=inputs,cells=cells,census_sha=sha(CENSUS)))


def selftest():
    assert max_new(400,100,600,.5)==200
    assert max_new(-400,100,0,.5)==220 and max_new(950,25,5000,.5)==25
    assert max_new(400,100,0,.5)==0
    key=key_for(0,50000,220,True)
    assert key==(0,0,1,1) and forecast({},key)==(0.,3)
    assert forecast({groups(key)[2]:dict(markets=8,q25=10)},key)==(10,2)
    for k in (0,39,40,41):
        a=np.zeros(80,np.float32); a[k]=10
        cache=dict(x={0:a,1:-a},status={s:np.zeros(80,dtype=int) for s in (0,1)})
        _,tr=triggers(cache,MM.TR.T.P.frozen_edges())
        assert bool(tr[0][k])==(k>=40)
        cache['status'][0][30]=1
        assert not triggers(cache,MM.TR.T.P.frozen_edges())[1][0][k]
    print('size, support and opening clock controls passed',flush=True)


if __name__=='__main__':
    command=argparse.ArgumentParser()
    command.add_argument('stage',choices=['selftest','train','fit','eval'])
    stage=command.parse_args().stage
    if stage=='selftest': selftest()
    elif stage=='fit': fit()
    else: run(stage)
