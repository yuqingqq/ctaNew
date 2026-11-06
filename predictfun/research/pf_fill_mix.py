"""Spread vs informed cost by fill type: the other makers vs our replay fills (2026-11-03; user: "what the main dif of our
strategy against theirs"). Descriptive, 8 read days.

Groups as pf_maker_behavior (makers ranked per day by live-window volume: top 5 / 6-20 / rest; ours: ladder lad1q arm and
single quote S60 + CAP10). Per timed fill (1 s-stamped legs and fills with no frame before, an empty side or inside the
spread are statuses): position = at best / behind (frame strictly before); swept = the frame after the trade shows the best
beyond the fill price (the level was cleared; same definition for our fills, on the market book). "After" = the first
frame at or after the fill time (a leg timed by its book drop carries the drop frame's time; the frame strictly after it
was used before 22:40Z and is reported as a check, with the replay's own through flag for our fills); spread = s (m0 - p),
informed = s (m1 - m0), c/sh, venue mids strictly before and 1 s after. Window part: first 60 s (our arms quote only then).
Swept, transaction rule (main, 22:55Z): a maker leg is swept through when its transaction has a leg on the same side at a
worse price for the maker (the taker went beyond it) - the replay's own through rule, so market and our fills compare
like for like. Transaction key (legs_timed2 has no hash): (mid, exec_s, first receipt, taker token, taker quote, price
executed), validated against the hash by pf_match_type PF_MT_TXCHECK=1. The book rule is kept as a check: it reads a
frame that can predate the trade and under-counts sweeps.
Usage: python3 -m predictfun.research.pf_fill_mix
"""
import os
import numpy as np, pandas as pd
from . import pf_io as io

DAYS = ['20261023', '20261024', '20261025', '20261026', '20261027', '20261031', '20261101', '20261102']; FRESH = ('20261101', '20261102')
S4 = f'{io.OUT}/step4'; OUTD = f'{S4}/fill_mix'; os.makedirs(OUTD, exist_ok=True)
COLS = ['mid', 't_ms', 'b1_p', 'a1_p']
OURS = [('ours: ladder', lambda d: f"{S4}/ladder_{'preds_c3' if d in FRESH else 'preds_o3'}_lad1q/{d}/fills/K3_z5_S60_TH0.5_CAP50_BUD75_RF1000_THIN30_CD500.parquet"),
        ('ours: single quote', lambda d: f"{S4}/live_tokyoreal_custom_e2_{'preds_c3' if d in FRESH else 'preds_o3'}_lc/{d}/fills/"
                                         "MICRO_G4_G2_G6_G5_G3_G1_M025_XNA_XMIN1_XS0_XV_S60_CAP10__mid_szA1-5-25-0.parquet")]
GR = ['top 5', '6-20', 'rest', 'ours: ladder', 'ours: single quote']
AGG, st = [], dict(legs_1s_only=0, unusable=0)
for d in DAYS:
    S1 = f'{io.OUT}/step1/{d}'
    bk = pd.concat([pd.read_parquet(f'{S1}/books/{f}', columns=COLS) for f in sorted(os.listdir(f'{S1}/books'))], ignore_index=True)
    bk.sort_values(['mid', 't_ms'], inplace=True, kind='stable')
    reg = pd.read_parquet(f'{S1}/registry.parquet', columns=['mid', 'ws']).set_index('mid')
    lt = pd.read_parquet(f'{S1}/legs_timed2.parquet', columns=['mid', 'exec_s', 'side', 'ptick', 'shares', 't_ms', 'how', 't_e2', 'signer', 't_rel',
                                                              'first_recv_ns', 'taker_index', 'taker_quote', 'price_executed'])
    lt = lt[lt.mid.isin(reg.index)].copy()
    TK = ['mid', 'exec_s', 'first_recv_ns', 'taker_index', 'taker_quote', 'price_executed']
    lo = lt.groupby(TK, sort=False).ptick.transform('min').values; hi = lt.groupby(TK, sort=False).ptick.transform('max').values
    lt['thr_tx'] = np.where(lt.side.values == 0, lt.ptick.values > lo, lt.ptick.values < hi)
    lt = lt[(lt.t_rel >= 0) & (lt.t_rel < 60)].copy()
    vol = pd.read_parquet(f'{S1}/legs_timed2.parquet', columns=['mid', 'signer', 'shares', 't_rel'])
    vol = vol[vol.mid.isin(reg.index) & (vol.t_rel >= 0) & (vol.t_rel < 240)].groupby('signer').shares.sum().sort_values(ascending=False)
    rk = pd.Series(np.arange(1, len(vol) + 1), index=vol.index); r_ = lt.signer.map(rk).fillna(1e9).values
    lt['grp'] = np.where(r_ <= 5, 'top 5', np.where(r_ <= 20, '6-20', 'rest'))
    timed = (lt.how.values > 0) | np.isfinite(lt.t_e2.values); st['legs_1s_only'] += int((~timed).sum()); lt = lt[timed]
    lt['ta'] = np.where(lt.how.values > 0, lt.t_ms.values, lt.t_e2.values)
    lt['thr'] = lt.thr_tx.values; lt['tie'] = lt.how.values > 0
    parts = [lt[['mid', 'side', 'ptick', 'shares', 'ta', 'grp', 'thr', 'tie']]]
    for g_, f_ in OURS:
        f = pd.read_parquet(f_(d), columns=['mid', 't', 'side', 'pt', 'q', 'through'])
        parts.append(pd.DataFrame(dict(mid=f.mid.values, side=f.side.values, ptick=f.pt.values.astype(int), shares=f.q.values,
                                       ta=reg.loc[f.mid.values].ws.values * 1000.0 + f.t.values * 1000.0, grp=g_, thr=f.through.values, tie=False)))
    U = pd.concat(parts, ignore_index=True); UB = {m: x for m, x in U.groupby('mid', sort=False)}
    rows = []
    for mid, g in bk.groupby('mid', sort=False):
        u = UB.get(mid)
        if u is None:
            continue
        tf = g.t_ms.values.astype(np.float64); b1 = g.b1_p.values; a1 = g.a1_p.values
        vm = (b1 >= 0) & (a1 > b1); tv = tf[vm]; mv = ((b1 + a1) / 2.0)[vm]
        ta = u.ta.values.astype(np.float64); p = u.ptick.values; side = u.side.values; sg = np.where(side == 0, 1.0, -1.0)
        j0 = np.searchsorted(tf, ta, side='left') - 1; jn = np.searchsorted(tf, ta, side='left'); jr = np.searchsorted(tf, ta, side='right')
        i0 = np.searchsorted(tv, ta, side='left') - 1; i1 = np.searchsorted(tv, ta + 1000.0, side='right') - 1
        ok = (j0 >= 0) & (jr < len(tf)) & (i0 >= 0) & (i1 >= 0)
        j0c = np.clip(j0, 0, len(tf) - 1); jnc = np.clip(jn, 0, len(tf) - 1); jrc = np.clip(jr, 0, len(tf) - 1)
        bestr = np.where(side == 0, b1[jrc], a1[jrc]); swept_r = np.where(side == 0, bestr < p, (bestr < 0) | (bestr > p))
        best0 = np.where(side == 0, b1[j0c], a1[j0c]); bestn = np.where(side == 0, b1[jnc], a1[jnc])
        depth = np.where(side == 0, best0 - p, p - best0); ok &= (best0 >= 0) & (depth >= 0)
        swept = np.where(side == 0, bestn < p, (bestn < 0) | (bestn > p))
        m0 = mv[np.clip(i0, 0, None)]; m1 = mv[np.clip(i1, 0, None)]
        rows.append(pd.DataFrame(dict(grp=u.grp.values, ok=ok, pos=np.where(depth == 0, 'at best', 'behind'), swept=u.thr.values.astype(bool), q=u.shares.values,
                                      q_book=u.shares.values * swept,
                                      q_swept_r=u.shares.values * swept_r, q_thr=u.shares.values * u.thr.values, q_tie=u.shares.values * (u.tie.values & (tf[jnc] == ta)),
                                      qs=u.shares.values * sg * (m0 - p), qi=u.shares.values * sg * (m1 - m0))))
    F = pd.concat(rows, ignore_index=True); st['unusable'] += int((~F.ok).sum()); F = F[F.ok]
    a = F.groupby(['grp', 'pos', 'swept'])[['q', 'qs', 'qi', 'q_book', 'q_swept_r', 'q_thr', 'q_tie']].sum().reset_index(); a['day'] = d; AGG.append(a)
    del bk, lt, U, UB, F
    io.mem_check(f'day {d}')
A = pd.concat(AGG, ignore_index=True); A.to_parquet(f'{OUTD}/agg.parquet', index=False)
T = A.groupby(['grp', 'pos', 'swept'])[['q', 'qs', 'qi', 'q_book', 'q_swept_r', 'q_thr', 'q_tie']].sum()
print(f'statuses {st}; fills in the first 60 s; c/sh; share = of the group\'s volume')
print(f"{'group':20s} {'all: spread':>11s} {'informed':>8s} {'net':>6s} | " + ' | '.join(f'{x:^33s}' for x in ('at best, at its price', 'at best, swept through', 'behind, at its price', 'behind, swept through')))
print(f"{'':20s} {'':>11s} {'':>8s} {'':>6s} | " + ' | '.join(f"{'share':>6s} {'spread':>7s} {'inform':>7s} {'net':>6s}  " for _ in range(4)))
out = {}
for gname in GR:
    if gname not in T.index.get_level_values(0):
        continue
    t = T.loc[gname]; v = t.q.sum(); r = dict(spread=float(t.qs.sum() / v), informed=float(t.qi.sum() / v), swept=float(t.q[t.index.get_level_values(1)].sum() / v),
             swept_book=float(t.q_book.sum() / v), swept_frame_after=float(t.q_swept_r.sum() / v), ws_tie=float(t.q_tie.sum() / v)); cells = []
    for pos in ('at best', 'behind'):
        for sw in (False, True):
            if (pos, sw) in t.index:
                c = t.loc[(pos, sw)]; cs = dict(share=float(c.q / v), spread=float(c.qs / c.q), informed=float(c.qi / c.q))
            else:
                cs = dict(share=0.0, spread=np.nan, informed=np.nan)
            r[f"{pos}|{'swept' if sw else 'at price'}"] = cs
            cells.append(f"{cs['share']:6.0%} {cs['spread']:7.2f} {cs['informed']:7.2f} {cs['spread'] + cs['informed']:6.2f}  ")
    out[gname] = r
    print(f"{gname:20s} {r['spread']:11.2f} {r['informed']:8.2f} {r['spread'] + r['informed']:6.2f} | " + ' | '.join(cells))
print('swept share by definition: transaction / replay through rule (used) | book, frame at or after | book, frame strictly after | WS-timed legs tied to a frame time')
for gname, r in out.items():
    print(f"  {gname:20s} {r['swept']:6.1%} | {r['swept_book']:6.1%} | {r['swept_frame_after']:6.1%} | {r['ws_tie']:6.1%}")
io.write_json(f'{OUTD}/report.json', dict(statuses=st, groups=out))
