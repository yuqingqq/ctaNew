"""Why are our resting orders at the back of the queue? Market side (2026-11-03; user: "Why we are still in the back?").

Per window, from the venue book frames (venue time t_ms; 5 shown levels a side; order_count = all resting orders):
  1. listing: first frame relative to window start (ws); order_count and shown size at first frame, +1 s, +10 s, +60 s, ws
  2. pre-window mid (last frame before ws-240 s, ws-60 s, ws)
  3. the queue at window open: for each shown level at ws (last frame strictly before ws), size Q and a lower bound of the
     part resting since T0 (min of the shown size at that price over every frame in [T0, ws); status 'hidden' if the price
     left the 5 shown levels in between), T0 = first frame + 10 s and ws - 60 s
  4. a newcomer at window open: a 5-share order joining each such level at ws + 40 ms behind the shown size (lowered when a
     later frame shows less), never moved, to ws + 60 s; trades at its price consume the queue ahead first, a trade through
     its price fills it (pf_ladder's rule for one order). Outcome: first fill from the queue, first fill swept, or none.
     The same order placed at the front (nothing ahead) is the incumbent's best case.
Exclusions are statuses with counts. Usage: python3 -m predictfun.research.pf_queue_priority
"""
import os
import numpy as np, pandas as pd
from . import pf_io as io

DAYS = os.environ.get('PF_QP_DAYS', '20261023,20261024,20261025,20261026,20261027,20261031,20261101,20261102').split(',')
S4 = f'{io.OUT}/step4'; Z = 5.0; L_PL = 40.0; HOLD = 60000.0
COLS = ['mid', 't_ms', 'order_count'] + [f'{s}{i}_{k}' for s in 'ba' for i in range(1, 6) for k in 'ps']


def shown(P, S, side, p):
    """Shown size at price p per frame; NaN where p is beyond the 5 shown levels (unknown)."""
    sz = np.where(P == p, S, 0.0).sum(1)
    last = P[:, 4]
    beyond = (last >= 0) & ((p < last) if side == 0 else (p > last))
    return np.where(beyond, np.nan, sz)


def newcomer(tf, sz, tr_t, tr_p, tr_q, side, p, ahead0, t0, t1):
    """One never-moved order of Z shares live at t0 until t1. Returns ('queue'|'swept'|'none', first-fill time)."""
    jl = np.searchsorted(tf, t0, side='left') - 1
    cm = np.minimum.accumulate(np.where(np.isnan(sz[jl + 1:]), np.inf, sz[jl + 1:])) if jl + 1 < len(sz) else np.empty(0)
    k = np.searchsorted(tr_t, t0, side='left'); k1 = np.searchsorted(tr_t, t1, side='left')
    jt = np.searchsorted(tf, tr_t[k:k1], side='left') - 1               # frames strictly before each trade
    ahead = ahead0
    for i, j in zip(range(k, k1), jt):
        pt, q = tr_p[i], tr_q[i]
        if j > jl and cm[j - jl - 1] < ahead:
            ahead = cm[j - jl - 1]
        if pt == p:
            use = min(q, ahead); ahead -= use
            if q - use > 1e-12:
                return 'queue', tr_t[i]
        elif (side == 0 and pt < p) or (side == 1 and pt > p):
            return 'swept', tr_t[i]
    return 'none', np.nan


rows_l, rows_q = [], []
for d in DAYS:
    S1 = f'{io.OUT}/step1/{d}'
    bk = pd.concat([pd.read_parquet(f'{S1}/books/{f}', columns=COLS) for f in sorted(os.listdir(f'{S1}/books'))], ignore_index=True)
    bk.sort_values(['mid', 't_ms'], inplace=True, kind='stable')
    reg = pd.read_parquet(f'{S1}/registry.parquet', columns=['mid', 'ws', 'first_seen_ns']).set_index('mid')
    lt = pd.read_parquet(f'{S1}/legs_timed2.parquet', columns=['mid', 'exec_s', 'side', 'ptick', 'shares', 't_ms', 'how', 't_e2'])
    lt['ta'] = np.where(lt.how.values > 0, lt.t_ms.values, np.where(np.isfinite(lt.t_e2.values), lt.t_e2.values, lt.exec_s.values * 1000.0))
    LTB = {m: g.sort_values('ta', kind='stable') for m, g in lt.groupby('mid', sort=False)}
    for mid, g in bk.groupby('mid', sort=False):
        if mid not in reg.index:
            continue
        ws = float(reg.loc[mid].ws) * 1000.0; tf = g.t_ms.values.astype(np.float64); oc = g.order_count.values
        BP = g[[f'b{i}_p' for i in range(1, 6)]].values; BS = g[[f'b{i}_s' for i in range(1, 6)]].values
        AP = g[[f'a{i}_p' for i in range(1, 6)]].values; AS = g[[f'a{i}_s' for i in range(1, 6)]].values
        tot = np.where(BP >= 0, BS, 0).sum(1) + np.where(AP >= 0, AS, 0).sum(1)
        mids = np.where((BP[:, 0] >= 0) & (AP[:, 0] > BP[:, 0]), (BP[:, 0] + AP[:, 0]) / 2.0, np.nan)
        t_first = tf[0]
        def at(t):
            i = np.searchsorted(tf, t, side='left') - 1
            return i if i >= 0 else None
        rec = dict(day=d, mid=mid, first_rel_s=(t_first - ws) / 1000.0, seen_rel_s=float(reg.loc[mid].first_seen_ns) / 1e6 / 1000.0 - ws / 1000.0)
        for lab, t in (('first', t_first + 1e-3), ('p1s', t_first + 1000.0), ('p10s', t_first + 10000.0), ('p60s', t_first + 60000.0),
                       ('ws-120s', ws - 120000.0), ('ws-30s', ws - 30000.0), ('ws-10s', ws - 10000.0), ('ws-2s', ws - 2000.0), ('ws', ws)):
            i = at(t)
            rec[f'oc_{lab}'] = float(oc[i]) if i is not None and t <= ws else np.nan
            rec[f'shown_{lab}'] = float(tot[i]) if i is not None and t <= ws else np.nan
        for lab, t in (('m240', ws - 240000.0), ('m60', ws - 60000.0), ('ws', ws)):
            i = at(t); rec[f'mid_{lab}'] = float(mids[i]) if i is not None and tf[i] >= t_first else np.nan
        rows_l.append(rec)
        iw = at(ws)
        if iw is None or tf[0] > ws - 60000.0:
            rows_q.append(dict(day=d, mid=mid, status='no_prewindow_book')); continue
        h = LTB.get(mid)
        for side, P, S in ((0, BP, BS), (1, AP, AS)):
            if h is None:
                tr_t = tr_p = tr_q = np.empty(0)
            else:
                hs = h[(h.side.values == side) & (h.ta.values >= ws) & (h.ta.values < ws + HOLD + L_PL)]
                tr_t = hs.ta.values.astype(np.float64); tr_p = hs.ptick.values; tr_q = hs.shares.values.astype(np.float64)
            for lvl in range(5):
                p = P[iw, lvl]
                if p < 0:
                    continue
                Q = float(S[iw, lvl]); sz = shown(P, S, side, p)
                r = dict(day=d, mid=mid, side=side, level=lvl + 1, p=int(p), Q=Q, status='ok')
                for lab, t0 in (('since_list10', t_first + 10000.0), ('since_m60', ws - 60000.0)):
                    a = np.searchsorted(tf, t0, side='left'); seg = sz[a:iw + 1]
                    r[f'Q_{lab}'] = np.nan if (len(seg) == 0 or np.isnan(seg).any()) else float(seg.min())
                ja = np.searchsorted(tf, ws + L_PL, side='left') - 1
                ahead_new = sz[ja] if np.isfinite(sz[ja]) else Q
                for who, a0 in (('back', ahead_new), ('front', 0.0)):
                    o, tt = newcomer(tf, sz, tr_t, tr_p, tr_q, side, p, a0, ws + L_PL, ws + HOLD)
                    r[f'{who}_outcome'] = o; r[f'{who}_t'] = (tt - ws) / 1000.0 if np.isfinite(tt) else np.nan
                rows_q.append(r)
    del bk, lt, LTB
    io.mem_check(f'day {d}')
L = pd.DataFrame(rows_l); Qd = pd.DataFrame(rows_q)
QD = f"{S4}/queue_priority{os.environ.get('PF_QP_SFX', '')}"; os.makedirs(QD, exist_ok=True)
L.to_parquet(f'{QD}/listing.parquet', index=False); Qd.to_parquet(f'{QD}/levels.parquet', index=False)

print(f'windows {len(L)} (8 days); level rows {len(Qd)}; statuses {Qd.status.value_counts().to_dict()}')
print('\n1) listing and build-up (medians over windows; size in shares, both sides, 5 shown levels each)')
print(f"   first book frame at {L.first_rel_s.median():.1f} s from window open (p10/p90 {L.first_rel_s.quantile(.1):.1f}/{L.first_rel_s.quantile(.9):.1f}); "
      f"collector first saw the market at {L.seen_rel_s.median():.1f} s")
for lab in ('first', 'p1s', 'p10s', 'p60s', 'ws-120s', 'ws-30s', 'ws-10s', 'ws-2s', 'ws'):
    print(f"   {lab:8s} orders resting {L[f'oc_{lab}'].median():7.0f}   shown size {L[f'shown_{lab}'].median():8.0f}   (n {L[f'oc_{lab}'].notna().sum()})")
print('2) pre-window mid (cents): ' + ', '.join(f"{lab} median {L[f'mid_{lab}'].median():.1f} p10/p90 {L[f'mid_{lab}'].quantile(.1):.1f}/{L[f'mid_{lab}'].quantile(.9):.1f}" for lab in ('m240', 'm60', 'ws')))
ok = Qd[Qd.status == 'ok']
print('3) the queue at window open, by level (1 = best); shares, medians; lower bound of the part resting since T0 (hidden = left the shown levels)')
for lvl, g in ok.groupby('level'):
    s10 = g.Q_since_list10; s60 = g.Q_since_m60
    print(f"   level {lvl}: n {len(g):5d}  Q {g.Q.median():6.0f}  since listing+10 s >= {s10.median():6.0f} (hidden {s10.isna().mean():.0%}; "
          f"share of Q {np.nansum(s10) / g.Q[s10.notna()].sum():.0%})  since ws-60 s >= {s60.median():6.0f} (share of Q {np.nansum(s60) / g.Q[s60.notna()].sum():.0%})")
print('4) a 5-share order joining each level at window open, never moved, held 60 s: first fill (share of levels)')
for lvl, g in ok.groupby('level'):
    b = g.back_outcome.value_counts(normalize=True); f = g.front_outcome.value_counts(normalize=True)
    print(f"   level {lvl}: at the back  queue {b.get('queue', 0):.0%} swept {b.get('swept', 0):.0%} none {b.get('none', 0):.0%}   |   "
          f"at the front  queue {f.get('queue', 0):.0%} swept {f.get('swept', 0):.0%} none {f.get('none', 0):.0%}")
io.write_json(f'{QD}/report.json', dict(
    n_windows=len(L), statuses=Qd.status.value_counts().to_dict(),
    listing={c: float(L[c].median()) for c in L.columns if c.startswith(('oc_', 'shown_', 'mid_', 'first_rel', 'seen_rel'))},
    by_level={int(lvl): dict(n=len(g), Q_med=float(g.Q.median()),
                             back=g.back_outcome.value_counts(normalize=True).round(4).to_dict(),
                             front=g.front_outcome.value_counts(normalize=True).round(4).to_dict()) for lvl, g in ok.groupby('level')}))
