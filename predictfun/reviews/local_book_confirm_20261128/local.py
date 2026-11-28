"""Precompute causal price-level confirmations on received frames and alert events."""
from collections import defaultdict
import numpy as np


def flags(clock, mask, ts):
    j = clock.searchsorted(ts, 'right') - 1
    return np.where((j >= 0) & (ts - clock[j] <= 300), mask[j], 0)


def displayed(frames, side, price):
    start = 2 + side * 10
    levels = frames[:, start:start+5]
    sizes = frames[:, start+5:start+10]
    populated = levels >= 0
    worst = (np.where(populated, levels, 101).min(1) if side == 0 else
             np.where(populated, levels, -1).max(1))
    covered = (price >= worst if side == 0 else price <= worst) | ~populated.all(1)
    good = (frames[:, 2] >= 0) & (frames[:, 12] <= 100) & (frames[:, 2] < frames[:, 12])
    return np.where(covered & good, np.where(levels == price, sizes, 0).sum(1), np.nan)


def build(frames, start, end, clock, mask, grid, increments):
    # increments is a map from price to a list of exact relative frame-receipt times.
    rec = frames[:, 1]
    assert (np.diff(rec) >= 0).all()
    lo, hi = clock.searchsorted([start, start+end])
    times = np.unique(np.r_[clock[lo:hi][mask[lo:hi] != 0]-start, grid, rec])
    times = times[(times >= 0) & (times < end)]
    bits = flags(clock, mask, start+times)
    table = defaultdict(dict)
    per_price = {}
    for side in (0, 1):
        t = times[(bits & (1 << side)) != 0]
        cur = rec.searchsorted(t, 'right')-1
        old = rec.searchsorted(t-300, 'right')-1
        valid = (old >= 0) & (t-rec[cur] <= 300) & (t-300-rec[old] <= 300)
        for price, receipts in increments.items():
            pubs = np.unique(receipts)
            j = pubs.searchsorted(t, 'right')-1
            eligible = valid & (j >= 0) & (t-pubs[j] < 300)
            if not eligible.any():
                continue
            d = displayed(frames, side, price)
            hits = eligible & (d[cur] < d[old])
            at, qty = t[hits], d[cur[hits]]
            per_price[side, price] = (at, qty)
            for tick, size in zip(at, qty):
                table[tick, side][price] = size
    return table, per_price


def fixture():
    f = np.full((5,22), -1., dtype=float)
    f[:,0] = f[:,1] = [0,100,300,320,620]
    f[:,2], f[:,12], f[:,7], f[:,17] = 49,51,100,100
    f[2:,7] = 20
    c = np.array([0,299,319,600,620])
    m = np.array([0,1,1,1,0])
    def mk(fr):
        return build(fr,0,630,c,m,np.arange(0,630,50),{49:[320],48:[320]})[0]
    tab = mk(f)
    assert tab[320,0] == {49:20} and not tab.get((300,0)) and not tab.get((320,1))
    assert tab[320,0][49] <= 50 and tab[320,0][49] > 10
    assert not tab.get((620,0))
    other=f.copy()
    other[-1,7]=10000
    assert all(mk(other).get(k)==v for k,v in tab.items() if k[0]<620)
    filled = np.tile(f[2],(1,1))
    filled[0,2:7] = [49,48,47,46,45]
    assert np.isnan(displayed(filled,0,44)[0])
