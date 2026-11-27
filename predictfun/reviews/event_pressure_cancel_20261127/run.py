"""Paired replay of off-grid pressure cancellation under the frozen feed clock."""
from collections import Counter, defaultdict
from pathlib import Path
from unittest.mock import patch
import json
import sys

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PRIOR = HERE.parent / 'near_holdout_nov19_23_20261127'
sys.path.insert(0, str(PRIOR))
import runpy
from types import SimpleNamespace
H = SimpleNamespace(**runpy.run_path(str(PRIOR / "replay.py")))
import pf_mt_btc as source

R = H.R
STEP = R.BE.STEP
DAYS = H.DAYS


def update_times(day):
    """All valid observed update milliseconds, including those with unchanged mid."""
    base = source.day_ms(day)
    manifest = json.loads((H.OUT / f'grid_{day}.json').read_text())['inputs']
    assert len(manifest) == 25
    hours = []
    for stem, digest in manifest.items():
        rec, raw = source.load(source.Roots().bn / f'{stem}.csv.gz')
        assert rec['sha256'] == digest and rec['state'] == 'read'
        arr, _ = source.parse_bn(raw)
        good = ((arr['recv_ns'] >= 1787579334881534478) & np.isfinite(arr['ask'])
                & (arr['bid'] > 0) & (arr['ask'] > arr['bid']))
        times = arr['E_ms'][good] - base
        assert np.all(np.abs(times) < 2**31)
        hours.append(np.unique(times.astype(np.int32)))
        del raw, arr
    return np.unique(np.concatenate(hours)).astype(np.int64) + base + R.MM.B_IN


def adverse(clock, mid, at, edges):
    j = clock.searchsorted(at, side='right') - 1
    prev = clock.searchsorted(at - 250, side='right') - 1
    assert (prev >= 0).all()
    ret = (10000 * np.log(mid[j] / mid[prev])).astype(np.float32).astype(float)
    return {0: -ret >= edges[0][-1], 1: ret >= edges[1][-1]}


def events(updates, clock, mid, start, end, edges, flags):
    grid = np.arange(len(flags[0][0])) * STEP
    grid_on = adverse(clock, mid, start + grid, edges)
    assert all(np.array_equal(grid_on[s], flags[0][s]) for s in (0, 1))
    left, right = updates.searchsorted([start, start + end])
    times = updates[left:right]
    local = times - start
    off = local % STEP != 0
    times, local = times[off], local[off]
    on = adverse(clock, mid, times, edges)
    assert not (on[0] & on[1]).any()
    keep = on[0] | on[1]
    return list(zip(local[keep].tolist(), on[1][keep].astype(int).tolist()))


def cancel_live(b, g, side):
    # Skip engine advancement when no known-active generation could receive a request.
    if not any(b.E.o[i]['req'] is None and b.E.o[i]['act'] <= g for i in b.unclosed[side]):
        return []
    b.E.advance(g)
    b.notices(g)
    if not R.LB.book_ok_side(b.E.f[b.E.known], side):
        return []
    pulled = []
    for gid in b.unclosed[side]:
        if b.gid_state(gid, g) == 'live':
            b.E.request(gid, g, 'event_pressure')
            assert b.E.o[gid]['remove'] == g + 37
            pulled.append((gid, g, side))
    return pulled


def simulate(args, limits, offgrid):
    step, finish = R.LB.step, R.LB.finish
    pos, extra = 0, []
    result = {}

    def before(b, stop):
        nonlocal pos
        while pos < len(offgrid) and offgrid[pos][0] < stop:
            t, s = offgrid[pos]
            extra.extend(cancel_live(b, t, s))
            pos += 1

    def grid(b, k, pol):
        before(b, k * STEP)
        step(b, k, pol)

    def end(b):
        before(b, b.end)
        val = finish(b)
        assert pos == len(offgrid)
        orders = b.E.o
        assert all(o['submit'] % STEP == 0 for o in orders)
        assert all(o['req'] is None or o['req'] % STEP == 0 or o['why'] == 'event_pressure' for o in orders)
        assert sum(o['why'] == 'event_pressure' for o in orders) == len(extra)
        assert all(orders[i]['act'] <= t < orders[i]['remove'] for t,i,*_ in b.E.fills)
        last = {}
        reposts = 0
        for o in orders:
            key = o['side'], o['rung']
            old = last.get(key)
            if old and old['req'] is not None and old['req'] <= o['submit']:
                reposts += 1
            last[key] = o
        grid_pulls = sum(c[7:]==('PULL','pressure') for c in b.cmd_log)
        result.update(fills=list(b.E.fills), requests=sum(o['req'] is not None for o in orders),
                      reposts=reposts, grid_pressure_pulls=grid_pulls, event_pulls=len(extra),
                      lead_sum=sum(STEP-g % STEP for _,g,_ in extra))
        return val

    with patch.object(R.LB,'step',grid), patch.object(R.LB,'finish',end):
        row = H.pilot['simulate']('nominal', args, limits)
    fills = result.pop('fills')
    return dict(row, **result), fills


def exposures(fills, path):
    out = Counter()
    for t, _, side, q, price, it in fills:
        assert it >= 0 and tuple(path['trades'][it][:2]) == (t,side)
        out[(it,side,price)] += q
    return out


def valued(shares, path, end, payout):
    d = Counter(shares=0., win_sh=0., loss_sh=0., flat_sh=0., censored=0., favorable=0., adverse=0.,
                payout_win=0., payout_loss=0.)
    frames = path['frames']
    for (it,s,p),q in shares.items():
        d['shares'] += q
        final = q * (1-2*s) * (100*payout-p)/100
        d['payout_win' if final>=0 else 'payout_loss'] += final
        at = path['trades'][it,0]+1000
        i = np.searchsorted(frames[:,1],at,'right')-1
        if at>end or i<0 or not 0 <= frames[i,2] < frames[i,12] <= 100:
            d['censored'] += q
            continue
        unit = (1-2*s)*((frames[i,2]+frames[i,12])/2-p)
        d['win_sh' if unit>0 else 'loss_sh' if unit<0 else 'flat_sh'] += q
        d['favorable' if unit>=0 else 'adverse'] += q*unit/100
    assert abs(d['win_sh']+d['loss_sh']+d['flat_sh']+d['censored']-d['shares'])<1e-6
    return d


def one_day(day):
    assert day in DAYS
    baseline_path = PRIOR / f'markets_{day}.csv'
    census_path = PRIOR / f'census_{day}.csv'
    archive = json.loads((PRIOR/'result.json').read_text())
    for p in (baseline_path,census_path):
        assert R.sha(p)==archive['hashes'][p.name]
    baseline = pd.read_csv(baseline_path,float_precision='round_trip').query("policy=='near' and arm=='nominal'").set_index('mid')
    census = pd.read_csv(census_path).set_index('mid')
    assert len(census)==288 and baseline.index.is_unique
    directory=H.OUT/'step1'/day
    reg=pd.read_parquet(directory/'registry.parquet').set_index('mid')
    reg['hour']=(reg.ws.astype('int64')%86400)//3600
    fit=H.U.S.read(R.HERE/'fit_v2.json')['cells']
    edges=R.MM.TR.T.P.frozen_edges()
    hashes=json.loads((PRIOR/'inputs.json').read_text())
    for rel,digest in hashes.items():
        if day in rel or rel==f'btc/btc_{H.prev_day(day)}.parquet':
            assert R.sha(H.OUT/rel)==digest
    pressure=pd.read_parquet(H.OUT/f'pressure_{day}.parquet')
    k,mid=pressure.k_ms.to_numpy(),pressure.mid.to_numpy()
    clock=k+R.MM.B_IN
    opening=source.day_ms(day)
    updates=update_times(day)
    ix=updates.searchsorted(clock)
    assert (ix<len(updates)).all() and np.array_equal(updates[ix],clock)
    with patch.multiple(R.BTC,BTC_DAYS=(H.prev_day(day),day),PAD_LO_MS=opening-3600000,BTC_END_MS=opening+86400000):
        btc=R.BTC.load_btc(H.BTC,day,Counter())
    records=[]
    for m,status,path,info in H.worlds(day,directory,reg.sort_values('ws'),False):
        assert status==census.at[m,'status']
        if status!='complete':
            continue
        ws=reg.at[m,'ws']*1000.
        ident=R.MM.ident_of(day,m,ws,info,path)
        assert ident['tape_id']==baseline.at[m,'tape_id']
        with patch.object(R.MM,'bn',lambda:(k,mid)):
            flags,z=H.U.S.signal_data(btc,info,ws,edges)
        start,end=info['t_start'],info['end_rel']
        _,limits=H.strong['curves']((ws+300000-start-STEP*np.arange(len(z)))/1000,z)
        payout=R.VM.V.payout_of(reg.loc[m])
        path=dict(path,frames=np.asarray(path["frames"]),trades=np.asarray(path["trades"]))
        args=(path,info,ident,flags,fit,payout)
        feed=events(updates,clock,mid,start,end,edges,flags)
        control,old=simulate(args,limits,[])
        for key in baseline.columns.difference(['day','arm','policy','tape_id']):
            assert abs(control[key]-baseline.at[m,key])<1e-6,(day,m,key)
        candidate,new=simulate(args,limits,feed)
        a,b=exposures(old,path),exposures(new,path)
        loss,add=valued(a-b,path,end,payout),valued(b-a,path,end,payout)
        assert abs((candidate['settle']-control['settle'])-(add['payout_win']+add['payout_loss']-loss['payout_win']-loss['payout_loss']))<1e-6
        assert abs(candidate['shares']-control['shares']-(add['shares']-loss['shares']))<1e-6
        row=dict(day=day,mid=m,tape_id=ident['tape_id'])
        for prefix,metrics in [('grid',control),('event',candidate),('lost',loss),('added',add),('grid_marks',valued(a,path,end,payout)),('event_marks',valued(b,path,end,payout))]:
            row.update({f'{prefix}_{key}':v for key,v in metrics.items()})
        records.append(row)
    assert len(records)==len(baseline) and {r['mid'] for r in records}==set(baseline.index)
    pd.DataFrame(records).to_csv(HERE/f'paired_{day}.csv',index=False)
    print(day,len(records),'verified pairs',flush=True)


if __name__=='__main__':
    one_day(sys.argv[1])
