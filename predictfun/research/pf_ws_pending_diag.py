"""Diagnostic: why WS settlementsPending increases cover only ~55% of REST trade volume.

settlementsPending is ONE aggregate USD value per (book side, price): trades matched but not yet settled
on-chain. A frame-to-frame change at a price = new matches - settlements at that price, so it is not a
trade list. Per day, live t_rel 0..300 s, USD at the YES price (shares x p):
  A. volume: REST live USD vs WS increases and decreases
  B. pending lag: from an increase at a (side, price) level to the next decrease of that level
  C. per REST cell (market, YES price, executedAt second S): WS increases at that price (either side)
     in seconds S-1..S+1 vs REST USD in S-1..S+1 -> visible (>= 95%), partial, invisible; split by
     whether the level fell in that span, whether it was already non-zero, and whether frames were missing
  D. invisible/partial share by taker type and by leg size
Usage: python3 -m predictfun.research.pf_ws_pending_diag 20261023
Writes step1/<day>/pending_diag.json.
"""
import sys, collections
import numpy as np, pandas as pd
from . import pf_io as io

day = sys.argv[1]; S1 = f'{io.OUT}/step1/{day}'
reg = pd.read_parquet(f'{S1}/registry.parquet', columns=['mid', 'ws', 'we'])
win = {int(m): (int(ws) * 1000 - 2000, int(we) * 1000 + 3000) for m, ws, we in zip(reg.mid, reg.ws, reg.we)}
live_ms = {int(m): (int(ws) * 1000, int(we) * 1000) for m, ws, we in zip(reg.mid, reg.ws, reg.we)}

prev, ev, frt = {}, collections.defaultdict(list), collections.defaultdict(list)
init = {}          # (mid, side, pt) -> level in the first frame seen (deltas start after it)
n_backwards = 0
for h in io.hours_of(day, extra_after=1, extra_before=1):
    for recv, d in io.ws_books(h):
        mid = d.get('marketId'); w = win.get(mid); t = d['updateTimestampMs']
        if w is None or not (w[0] <= t <= w[1]):
            continue
        if frt[mid] and t < frt[mid][-1]:
            n_backwards += 1
        frt[mid].append(t)
        cur = {}
        for si, key in ((0, 'bids'), (1, 'asks')):
            for p, v in (d.get('settlementsPending') or {}).get(key) or []:
                k = (si, int(round(float(p) * 100))); cur[k] = cur.get(k, 0.0) + float(v)
        old = prev.get(mid)
        if old is None:
            for k, v in cur.items():
                init[(mid,) + k] = v
        else:
            for k in set(cur) | set(old):
                dl = cur.get(k, 0.0) - old.get(k, 0.0)
                if abs(dl) > 1e-9:
                    ev[(mid, k[1])].append((t, k[0], dl))
        prev[mid] = cur
    io.mem_check(h)

# ---- A + B ----
pos = neg = 0.0; lags = []; censored = 0
for (mid, pt), lst in ev.items():
    a, b = live_ms[mid]
    by_side = collections.defaultdict(list)
    for t, s, dl in lst:
        by_side[s].append((t, dl))
        if a <= t < b:
            if dl > 0: pos += dl
            else: neg -= dl
    for s, xs in by_side.items():
        xs.sort()
        tn = np.array([t for t, dl in xs if dl < 0], dtype=np.int64)
        for t, dl in xs:
            if dl > 0 and a <= t < b:
                j = np.searchsorted(tn, t, side='right')
                if j < len(tn): lags.append(tn[j] - t)
                else: censored += 1
lags = np.array(lags, dtype=np.float64)

L = pd.read_parquet(f'{S1}/legs_timed.parquet', columns=['mid', 'exec_s', 'ptick', 'shares', 'taker_index', 'taker_quote'])
L['usd'] = L.shares * L.ptick / 100.0
rest_usd = float(L.usd.sum())

# ---- C ----
W = collections.defaultdict(float); N = collections.defaultdict(float)
for (mid, pt), lst in ev.items():
    for t, s, dl in lst:
        b = int(round(t / 1000.0))
        if dl > 0: W[(mid, pt, b)] += dl
        else: N[(mid, pt, b)] -= dl
lvl_ts = {}        # (mid, pt) -> (times, level summed over sides) for the level just before a time
for (mid, pt), lst in ev.items():
    lst.sort()
    lvl_ts[(mid, pt)] = (np.array([t for t, _, _ in lst], dtype=np.int64), np.cumsum([dl for _, _, dl in lst]))
init_pt = collections.defaultdict(float)
for (m, s, p), v in init.items():
    init_pt[(m, p)] += v
frt = {m: np.sort(np.array(v, dtype=np.int64)) for m, v in frt.items()}

C = L.groupby(['mid', 'exec_s', 'ptick'], as_index=False).agg(usd=('usd', 'sum'), shares=('shares', 'sum'))
R = {(m, p, s): u for m, s, p, u in zip(C.mid.values, C.exec_s.values, C.ptick.values, C.usd.values)}
cat = []
for m, s, p, u in zip(C.mid.values, C.exec_s.values, C.ptick.values, C.usd.values):
    w3 = sum(W.get((m, p, s + k), 0.0) for k in (-1, 0, 1))
    n3 = sum(N.get((m, p, s + k), 0.0) for k in (-1, 0, 1))
    r3 = sum(R.get((m, p, s + k), 0.0) for k in (-1, 0, 1))
    wc = sum(W.get((m, 100 - p, s + k), 0.0) for k in (-1, 0, 1))
    tt, lv = lvl_ts.get((m, p), (np.zeros(0, np.int64), np.zeros(0)))
    j = np.searchsorted(tt, (s - 1.5) * 1000) - 1
    before = init_pt.get((m, p), 0.0) + (lv[j] if j >= 0 else 0.0)
    ft = frt.get(m)
    if ft is None or len(ft) == 0:
        gap = 1e9
    else:
        x = ft[np.searchsorted(ft, (s - 1.5) * 1000):np.searchsorted(ft, (s + 1.5) * 1000, side='right')]
        gap = 3000.0 if len(x) == 0 else float(max(np.diff(np.r_[(s - 1.5) * 1000, x, (s + 1.5) * 1000])))
    vis = 'visible' if w3 >= 0.95 * r3 else ('partial' if w3 > 1e-9 else 'invisible')
    cat.append((vis, n3 > 1e-9, before > 1e-6, gap > 1000, wc > 1e-9))
C['vis'] = [c[0] for c in cat]; C['fell'] = [c[1] for c in cat]; C['nonzero_before'] = [c[2] for c in cat]
C['frame_gap_1s'] = [c[3] for c in cat]; C['complement_inc'] = [c[4] for c in cat]
tot = C.usd.sum()
def share(mask): return round(float(C.usd[mask].sum() / tot), 4)
outC = {v: share(C.vis == v) for v in ('visible', 'partial', 'invisible')}
for v in ('partial', 'invisible'):
    mv = C.vis == v
    outC[f'{v}_split'] = dict(
        level_fell_in_span=share(mv & C.fell), level_nonzero_before=share(mv & C.nonzero_before),
        no_change_level_zero_before=share(mv & ~C.fell & ~C.nonzero_before),
        no_change_level_nonzero_before=share(mv & ~C.fell & C.nonzero_before),
        frame_gap_over_1s=share(mv & C.frame_gap_1s), increase_at_complement_price=share(mv & C.complement_inc))
# ---- D: by taker type and leg size (leg-level, using its cell's category) ----
Lc = L.merge(C[['mid', 'exec_s', 'ptick', 'vis']], on=['mid', 'exec_s', 'ptick'], how='left')
byt = {}
for (ti, tq), g in Lc.groupby(['taker_index', 'taker_quote']):
    byt[f'taker_index{ti}_quote{tq}'] = dict(usd_share=round(float(g.usd.sum() / Lc.usd.sum()), 4),
        **{v: round(float(g.usd[g.vis == v].sum() / g.usd.sum()), 4) for v in ('visible', 'partial', 'invisible')})
Lc['sz'] = pd.cut(Lc.shares, [0, 1, 5, 50, 1e12], right=False, labels=['<1', '1-5', '5-50', '>=50'])
bys = {str(k): {v: round(float(g.usd[g.vis == v].sum() / g.usd.sum()), 4) for v in ('visible', 'partial', 'invisible')}
       for k, g in Lc.groupby('sz', observed=True)}
cells_ps = C.groupby(['mid', 'exec_s']).ptick.nunique()

gaps = np.concatenate([np.diff(v) for v in frt.values() if len(v) > 1])
out = dict(day=day, n_rest_cells=int(len(C)), n_frames=int(sum(len(v) for v in frt.values())), frames_out_of_order=n_backwards,
           frame_gap_ms_pct=dict(zip(['p50', 'p90', 'p99'], np.percentile(gaps, [50, 90, 99]).round(1).tolist())),
           A_volume_usd=dict(rest_live=round(rest_usd), ws_increases=round(pos), ws_decreases=round(neg),
                             ws_increase_over_rest=round(pos / rest_usd, 4)),
           B_increase_to_next_decrease_ms=dict(n=int(len(lags)), censored=censored,
               pct=dict(zip(['p10', 'p25', 'p50', 'p75', 'p90'], np.percentile(lags, [10, 25, 50, 75, 90]).round(0).tolist())),
               share_le_100ms=round(float((lags <= 100).mean()), 4), share_le_1s=round(float((lags <= 1000).mean()), 4)),
           C_cells_usd_share=outC, D_by_taker=byt, D_by_leg_size=bys,
           seconds_with_trades_at_2plus_prices=round(float((cells_ps >= 2).mean()), 4))
io.write_json(f'{S1}/pending_diag.json', out)
import json; print(json.dumps(out, indent=1))

# ---- E: where WS increases exceed REST (live, USD) ----
# REST taker side per match record: (market, taker YES-space limit price, second) -> USD; and any trade per (market, second)
TB = collections.defaultdict(float); seen = set()
for h in io.hours_of(day, extra_after=1, extra_before=1):
    for recv, params, path, pl in io.rest_rows(h, '/v1/orders/matches'):
        for x in pl.get('data') or []:
            mid = (x.get('market') or {}).get('id')
            if mid not in live_ms:
                continue
            k = x['transactionHash'] + x['taker']['hash']
            if k in seen:
                continue
            seen.add(k)
            try:
                s_ = int(io.iso_s(x['executedAt']))
            except (KeyError, TypeError, ValueError):
                continue
            side_, py = io.yes_side(x['taker']); tp = int(round(py * 100))
            TB[(mid, tp, s_)] += int(x['amountFilled']) / io.E18 * tp / 100.0
del seen
ANY = set(zip(C.mid.values, C.exec_s.values))
RB = collections.defaultdict(float)
for (m, p, s), u in R.items():
    RB[(m, p, s)] += u
orph = orph_flick = flick = both_sides = 0.0; inc_tot = 0.0
orph_taker = orph_other_price = orph_no_trade = 0.0
orph_by_win = collections.defaultdict(float); inc_by_win = collections.defaultdict(float)
orph_by_side = collections.defaultdict(float)
for (mid, pt), lst in ev.items():
    a, b = live_ms[mid]
    last_dec = {}                                # side -> (t, size) of the latest decrease
    by_t = collections.defaultdict(dict)
    for t, s, dl in lst:
        by_t[t][s] = dl
    for t, s, dl in lst:                         # lst is sorted by t
        if dl < 0:
            last_dec[s] = (t, -dl); continue
        if not (a <= t < b):
            continue
        inc_tot += dl; inc_by_win[mid] += dl
        ld = last_dec.get(s)
        fl = ld is not None and t - ld[0] <= 200 and abs(ld[1] - dl) <= 0.01 * dl + 1e-6
        if fl: flick += dl
        o = by_t[t].get(1 - s)
        if o is not None and o > 0 and abs(o - dl) <= 0.01 * dl + 1e-6: both_sides += dl
        bk = int(round(t / 1000.0))
        if sum(RB.get((mid, pt, bk + k), 0.0) for k in (-1, 0, 1)) <= 1e-9:
            orph += dl; orph_by_win[mid] += dl; orph_by_side[s] += dl
            if fl: orph_flick += dl
            if any(TB.get((mid, pt, bk + k), 0.0) > 0 for k in (-1, 0, 1)): orph_taker += dl
            elif any((mid, bk + k) in ANY for k in (-1, 0, 1)): orph_other_price += dl
            else: orph_no_trade += dl
ow = pd.Series(orph_by_win).sort_values(ascending=False)
rest_by_win = L.groupby('mid').usd.sum()
top = [dict(mid=int(m), orphan_usd=round(float(v)), ws_inc_usd=round(float(inc_by_win[m])), rest_usd=round(float(rest_by_win.get(m, 0.0))))
       for m, v in ow.head(8).items()]
out['E_ws_increase_excess'] = dict(
    ws_inc_usd=round(inc_tot), rest_usd=round(rest_usd), excess_usd=round(inc_tot - rest_usd),
    orphan_inc_usd=round(orph), orphan_share_of_inc=round(orph / inc_tot, 4),
    orphan_by_side={str(k): round(v) for k, v in orph_by_side.items()},
    readd_after_equal_decrease_le200ms_usd=round(flick), orphan_that_are_readd_usd=round(orph_flick),
    same_frame_equal_increase_on_other_side_usd=round(both_sides),
    orphan_at_a_taker_limit_price_usd=round(orph_taker), orphan_other_rest_trade_same_market_pm1s_usd=round(orph_other_price),
    orphan_no_rest_trade_in_market_pm1s_usd=round(orph_no_trade),
    orphan_top5pct_windows_share=round(float(ow.head(max(1, len(ow) // 20)).sum() / max(ow.sum(), 1e-9)), 4),
    top_orphan_windows=top)
io.write_json(f'{S1}/pending_diag.json', out)
print(json.dumps(out['E_ws_increase_excess'], indent=1))
