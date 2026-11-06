"""How are freed prices re-posted so fast? (2026-11-04; user: "How do they repost so fast? Do they send cancel and repost
together"). Descriptive, read days, nothing selected.

Part 1 (8 days): book publication cadence: gaps between consecutive frames of one market (venue updateTimestampMs), live.
Part 2 (8 days): new best prices taken in the same book update that freed them (as pf_vacate_gap), split by how the old
  order at that price left: 'lifted' = trades at that price on the other side in the interval >= 90% of its shown size
  there in the frame before; 'partly traded'; 'pulled' = no trade there. Also the change in the market's resting order count
  in that update, and whether the new size equals the removed size (pulled ones).
Part 3 (10-23 and 11-01, raw REST matches with order hashes): an aggressive order that is not filled in full can rest as a
  limit order; its hash then appears later as a maker leg. Share of maker volume from such remainders, and for new best
  prices taken in the same update after being lifted: is the first fill at the new price a remainder of the order that
  lifted it (same side as the lifting buyer / seller)?
Usage: python3 -m predictfun.research.pf_repost_anatomy
"""
import os, collections
import numpy as np, pandas as pd
from . import pf_io as io

DAYS = ['20261023', '20261024', '20261025', '20261026', '20261027', '20261031', '20261101', '20261102']; RAW_DAYS = ['20261023', '20261101']
SKIP = ('20261028', '20261029', '20261030')
S4 = f'{io.OUT}/step4'; OUTD = f'{S4}/repost_anatomy'; os.makedirs(OUTD, exist_ok=True)
GAP_EDGES = [5.0, 20.0, 40.0, 60.0, 100.0, 250.0]; GAP_LAB = ['<5', '5-20', '20-40', '40-60', '60-100', '100-250', '>=250']


def raw_legs(day):
    """Maker legs with hashes from the raw REST matches (one first receipt per leg) and each transaction's taker hash/side."""
    legs = {}
    for h in io.hours_of(day):
        if h[:8] in SKIP or not os.path.exists(f'{io.PF_RUN}/rest/{h}.jsonl.gz'):
            continue
        for recv, _, _, pl in io.rest_rows(h, '/v1/orders/matches'):
            for x in pl.get('data') or []:
                mid = (x.get('market') or {}).get('id'); t = x['taker']; tx = x['transactionHash'][2:18]
                tside, _ = io.yes_side(t)
                for m in x['makers']:
                    k = tx + m['hash'][2:18]
                    if k in legs:
                        if recv < legs[k][1]:
                            legs[k] = (legs[k][0], recv) + legs[k][2:]
                        continue
                    side, py = io.yes_side(m)
                    legs[k] = (mid, recv, 0 if side == 'bid' else 1, int(round(py * 100)), int(m['amount']) / io.E18, m['hash'][2:18], t['hash'][2:18], 0 if tside == 'bid' else 1)
        io.mem_check(f'rest {h}')
    return pd.DataFrame(list(legs.values()), columns=['mid', 'first_recv_ns', 'side', 'ptick', 'shares', 'mhash', 'thash', 'tside'])


gaps = collections.Counter(); ev = []; FIRST = []
for d in DAYS:
    S1 = f'{io.OUT}/step1/{d}'
    cols = ['mid', 't_ms', 'order_count'] + [f'{s}{i}_{k}' for s in 'ba' for i in range(1, 6) for k in 'ps']
    bk = pd.concat([pd.read_parquet(f'{S1}/books/{f}', columns=cols) for f in sorted(os.listdir(f'{S1}/books'))]).sort_values(['mid', 't_ms'], kind='stable')
    reg = pd.read_parquet(f'{S1}/registry.parquet', columns=['mid', 'ws']).set_index('mid')
    lt = pd.read_parquet(f'{S1}/legs_timed2.parquet', columns=['mid', 'side', 'ptick', 'shares', 't_ms', 'how', 't_e2', 't_rel', 'first_recv_ns'])
    lt = lt[(lt.how.values > 0) | np.isfinite(lt.t_e2.values)].copy(); lt['ta'] = np.where(lt.how.values > 0, lt.t_ms.values, lt.t_e2.values)
    LB = {k: x for k, x in lt.groupby(['mid', 'side'], sort=False)}
    for mid, g in bk.groupby('mid', sort=False):
        if mid not in reg.index:
            continue
        ws = float(reg.loc[mid].ws) * 1000.0; tf = g.t_ms.values.astype(float); lw = (tf >= ws) & (tf < ws + 240000.0)
        dg = np.diff(tf)[lw[1:]]
        for lab, n in zip(*np.unique(np.array(GAP_LAB, dtype=object)[np.searchsorted(GAP_EDGES, dg, side='right')], return_counts=True)):
            gaps[lab] += int(n)
        oc = g.order_count.values
        BK = {s: (g[[f'{p}{i}_p' for i in range(1, 6)]].values, g[[f'{p}{i}_s' for i in range(1, 6)]].values) for s, p in ((0, 'b'), (1, 'a'))}
        for side in (0, 1):
            P, S = BK[side]; Po, So = BK[1 - side]; best = P[:, 0]; opp = Po[:, 0]
            imp = np.r_[False, (best[1:] >= 0) & (best[:-1] >= 0) & ((best[1:] > best[:-1]) if side == 0 else (best[1:] < best[:-1]))] & lw
            for k in np.flatnonzero(imp):
                p = best[k]
                if not (opp[k - 1] >= 0 and ((opp[k - 1] <= p) if side == 0 else (opp[k - 1] >= p))):
                    continue                                                  # not freed in this update (gap > 0 or wide)
                old = float(np.where(Po[k - 1] == p, So[k - 1], 0.0).sum())  # the other side's size at p just before
                new = float(np.where(P[k] == p, S[k], 0.0).sum())
                h = LB.get((mid, 1 - side))
                V = float(h.shares.values[(h.ptick.values == p) & (h.ta.values > tf[k - 1]) & (h.ta.values <= tf[k])].sum()) if h is not None else 0.0
                how = 'lifted' if (old > 0 and V >= 0.9 * old) else ('partly traded' if V > 0 else ('pulled' if old > 0 else 'crossed'))
                rec = dict(day=d, mid=mid, side=side, p=int(p), t=tf[k], dt=tf[k] - tf[k - 1], how=how, old=old, new=new, V=V, doc=int(oc[k] - oc[k - 1]))
                ev.append(rec)
                hs = LB.get((mid, side))
                if d in RAW_DAYS and hs is not None:                         # first fill at the new price while it rests
                    later = np.flatnonzero(~np.isin(np.arange(k + 1, len(tf)), np.flatnonzero(np.any(P == p, 1))))
                    kend = k + 1 + later[0] if len(later) else len(tf)
                    tend = tf[kend] if kend < len(tf) else np.inf
                    f = hs[(hs.ptick.values == p) & (hs.ta.values >= tf[k]) & (hs.ta.values <= tend)]
                    if len(f):
                        j = int(np.argmin(f.ta.values)); FIRST.append(dict(day=d, mid=mid, side=side, p=int(p), how=how, first_recv_ns=int(f.first_recv_ns.values[j]),
                                                                           shares=float(f.shares.values[j]), t_lift=tf[k]))
    del bk, lt, LB
    io.mem_check(f'day {d}')
E = pd.DataFrame(ev); E.to_parquet(f'{OUTD}/same_update_events.parquet', index=False)
tot = sum(gaps.values())
print('part 1: gaps between book frames of one market (venue ms), live 0-240 s: ' + '  '.join(f'{k} {gaps[k] / tot:.1%}' for k in GAP_LAB) + f'  (n {tot})')
print(f'\npart 2: new best prices taken in the same update that freed them: {len(E)}')
for how, g in E.groupby('how'):
    pulled_same = (g.how == 'pulled') & (np.abs(g.new - g.old) <= 0.01 * np.maximum(g.old, 1e-9))
    print(f"  {how:13s} {len(g) / len(E):6.1%}  update interval median {g.dt.median():4.0f} ms | resting orders change: {' '.join(f'{k:+d} {v:.0%}' for k, v in g.doc.clip(-3, 3).value_counts(normalize=True).sort_index().items())}"
          + (f" | new size = removed size (1%): {pulled_same.mean():.1%}" if how == 'pulled' else ''))
# part 3: remainders
F = pd.DataFrame(FIRST); out3 = {}
for d in RAW_DAYS:
    R = raw_legs(d)
    lt = pd.read_parquet(f'{io.OUT}/step1/{d}/legs_timed2.parquet', columns=['mid', 'first_recv_ns', 'side', 'ptick', 'shares', 't_rel'])
    J = lt.merge(R.drop_duplicates(['mid', 'first_recv_ns', 'side', 'ptick', 'shares']), on=['mid', 'first_recv_ns', 'side', 'ptick', 'shares'], how='left')
    first_taker = R.groupby('thash').first_recv_ns.min()                          # when each order was first seen as a taker
    seen = J.mhash.map(first_taker); J['remainder'] = seen.notna() & (seen <= J.first_recv_ns)
    live = J[(J.t_rel >= 0) & (J.t_rel < 240)]
    rem_share = float(live.shares[live.remainder].sum() / live.shares.sum())
    f = F[F.day == d].merge(J[['mid', 'first_recv_ns', 'side', 'ptick', 'shares', 'mhash', 'remainder']].rename(columns={'ptick': 'p'}),
                            on=['mid', 'first_recv_ns', 'side', 'p', 'shares'], how='left')
    row = dict(unjoined=int(J.mhash.isna().sum()), maker_volume_from_remainders=rem_share)
    for how, g in f.groupby('how'):
        row[f'first_fill_remainder|{how}'] = float(g.remainder.fillna(False).mean()); row[f'n|{how}'] = int(len(g))
    out3[d] = row
    print(f"\npart 3 {d}: maker volume (live 0-240 s) from orders that first traded as takers: {rem_share:.1%}; unjoined legs {row['unjoined']}")
    for how, g in f.groupby('how'):
        print(f"  first fill at a price taken in the same update, after it was {how:13s}: from a remainder {g.remainder.fillna(False).mean():5.1%} (n {len(g)})")
io.write_json(f'{OUTD}/report.json', dict(gaps={k: gaps[k] / tot for k in GAP_LAB}, same_update=E.how.value_counts(normalize=True).round(4).to_dict(), remainders=out3))
