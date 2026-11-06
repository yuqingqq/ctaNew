"""Do the other makers hold their resting orders through the early window? (2026-11-03; user: "Can you check if others hold
through the early window?")

Part 1, the queue resting at the open (all makers together; the book shows sizes per price, not orders). For each shown
level at the open (last frame strictly before ws; side s, price p, size Q0) the orders resting there can only leave by a
fill or a cancel, and fills at p are at most the volume traded at p. So for every later frame u where p is shown,
    cancelled from the open's queue by u  >=  Q0 - shown(u) - V_p(ws, u + slack]
and the running max of the right side is a LOWER bound on the open's queue cancelled (new orders joining behind can only
hide cancels, never fake them; the slack of 1 s absorbs trades stamped after the book shows them). Evaluated at 10 / 30 /
60 / 240 s, just before p first becomes the best (levels 2-5 at the open), and just before the first trade through p
(first leg on that side at a worse price for the maker). No upper bound exists from these data: holding cannot be proven,
only pulling.
Part 2, by maker (anonymised signer, ranked by live-window volume per day as docs §34): share of each band's fill volume
whose trade cleared the level (the frame after the trade shows the best beyond the fill price), by depth behind the best
at the trade. Legs with only a 1 s stamp are excluded (status count).
Usage: python3 -m predictfun.research.pf_hold_check            (selftest: ... pf_hold_check --selftest)
"""
import bisect, collections, os, sys
import numpy as np, pandas as pd

SLACK = float(os.environ.get('PF_HOLD_SLACK', '1000')); CHK = (10.0, 30.0, 60.0, 240.0)


def shown(P, S, side, p):
    """Shown size at price p per frame; NaN where p is beyond the 5 shown levels (unknown)."""
    sz = np.where(P == p, S, 0.0).sum(1)
    last = P[:, 4]
    beyond = (last >= 0) & ((p < last) if side == 0 else (p > last))
    return np.where(beyond, np.nan, sz)


def cancel_lb(tf, sz, tq, tv, Q0):
    """Running lower bound of the open's queue cancelled at each frame. tf: frame times (from the open on); sz: shown size
    at p there (NaN = unknown); tv / tq: sorted trade times at p and their shares."""
    cv = np.r_[0.0, np.cumsum(tq)]
    V = cv[np.searchsorted(tv, tf + SLACK, side='right')]
    x = np.where(np.isnan(sz), -np.inf, Q0 - sz - V)
    return np.maximum(np.maximum.accumulate(x), 0.0)


def selftest():
    tf = np.array([0., 100., 200., 300.])
    # 1) positive control: the size vanishes with no trades -> all of it cancelled
    assert np.allclose(cancel_lb(tf, np.array([100., 100., 0., 0.]), np.empty(0), np.empty(0), 100.0), [0, 0, 100, 100])
    # 2) consumed by a trade stamped 500 ms after the book shows it (inside the slack) -> no cancel
    assert np.allclose(cancel_lb(tf, np.array([100., 0., 0., 0.]), np.array([100.0]), np.array([600.0]), 100.0), 0)
    # 3) known-bad input: the same trade stamped 2 s late is outside the slack -> the bound wrongly shows a cancel (must fire)
    assert cancel_lb(tf, np.array([100., 0., 0., 0.]), np.array([100.0]), np.array([2000.0]), 100.0)[1] == 100.0
    # 4) cancels hidden by new orders: 50 join, all 100 originals cancel -> bound 50 (<= the true 100), never above
    assert np.allclose(cancel_lb(tf, np.array([100., 150., 50., 50.]), np.empty(0), np.empty(0), 100.0), [0, 0, 50, 50])
    # 5) unknown frames (price beyond the shown levels) are skipped; the bound never falls
    assert np.allclose(cancel_lb(tf, np.array([100., 40., np.nan, 100.]), np.empty(0), np.empty(0), 100.0), [0, 60, 60, 60])
    # 6) shown(): beyond the 5 shown bids is unknown, inside the range but absent is 0
    P = np.array([[50, 49, 48, 47, 46], [50, 49, 48, 47, -1]]); S = np.ones((2, 5)) * 7
    a = shown(P, S, 0, 44); assert np.isnan(a[0]) and a[1] == 0 and np.allclose(shown(P, S, 0, 48), 7)
    print('selftest PASS (6 cases)')


if __name__ == '__main__' and '--selftest' in sys.argv:
    selftest(); sys.exit(0)

from . import pf_io as io

DAYS = ['20261023', '20261024', '20261025', '20261026', '20261027', '20261031', '20261101', '20261102']
S4 = f'{io.OUT}/step4'; OUTD = f"{S4}/hold_check{'' if SLACK == 1000.0 else f'_slack{SLACK:g}'}"; os.makedirs(OUTD, exist_ok=True)
COLS = ['mid', 't_ms'] + [f'{s}{i}_{k}' for s in 'ba' for i in range(1, 6) for k in 'ps']
r1, r2, al, st = [], collections.Counter(), collections.Counter(), dict(no_open_book=0, legs_1s_only=0, legs_no_frame=0)
for d in DAYS:
    S1 = f'{io.OUT}/step1/{d}'
    bk = pd.concat([pd.read_parquet(f'{S1}/books/{f}', columns=COLS) for f in sorted(os.listdir(f'{S1}/books'))], ignore_index=True)
    bk.sort_values(['mid', 't_ms'], inplace=True, kind='stable')
    reg = pd.read_parquet(f'{S1}/registry.parquet', columns=['mid', 'ws']).set_index('mid')
    lt = pd.read_parquet(f'{S1}/legs_timed2.parquet', columns=['mid', 'exec_s', 'side', 'ptick', 'shares', 't_ms', 'how', 't_e2', 'signer', 't_rel'])
    lt = lt[lt.mid.isin(reg.index) & (lt.t_rel >= 0) & (lt.t_rel < 240)].copy()
    timed = (lt.how.values > 0) | np.isfinite(lt.t_e2.values)
    lt['ta'] = np.where(lt.how.values > 0, lt.t_ms.values, np.where(np.isfinite(lt.t_e2.values), lt.t_e2.values, lt.exec_s.values * 1000.0))
    LTB = {m: g.sort_values('ta', kind='stable') for m, g in lt.groupby('mid', sort=False)}
    BOOK = {}
    for mid, g in bk.groupby('mid', sort=False):
        if mid not in reg.index:
            continue
        ws = float(reg.loc[mid].ws) * 1000.0; tf = g.t_ms.values.astype(np.float64)
        BP = g[[f'b{i}_p' for i in range(1, 6)]].values; BS = g[[f'b{i}_s' for i in range(1, 6)]].values
        AP = g[[f'a{i}_p' for i in range(1, 6)]].values; AS = g[[f'a{i}_s' for i in range(1, 6)]].values
        BOOK[mid] = (tf, BP, BS, AP, AS)
        iw = np.searchsorted(tf, ws, side='left') - 1
        if iw < 0:
            st['no_open_book'] += 1; continue
        a = iw + 1; b = np.searchsorted(tf, ws + 240000.0, side='left'); fr = tf[a:b]
        h = LTB.get(mid)
        for side, P, S in ((0, BP, BS), (1, AP, AS)):
            hs = h[h.side.values == side] if h is not None else None
            best = P[a:b, 0]
            for lvl in range(5):
                p = P[iw, lvl]
                if p < 0:
                    continue
                Q0 = float(S[iw, lvl]); sz = shown(P[a:b], S[a:b], side, p)
                if hs is not None:
                    at = hs[hs.ptick.values == p]; tv = at.ta.values.astype(np.float64); tq = at.shares.values.astype(np.float64)
                    thr = hs.ta.values[(hs.ptick.values < p) if side == 0 else (hs.ptick.values > p)]
                    t_sw = float(thr.min()) if len(thr) else np.inf
                else:
                    tv = tq = np.empty(0); t_sw = np.inf
                lb = cancel_lb(fr, sz, tq, tv, Q0) if len(fr) else np.empty(0)
                def lb_at(t):
                    j = np.searchsorted(fr, t, side='left') - 1
                    return float(lb[j]) if j >= 0 else 0.0
                rec = dict(day=d, mid=mid, side=side, level=lvl + 1, Q0=Q0, n_frames=len(fr))
                for c in CHK:
                    rec[f'lb_{c:g}'] = lb_at(ws + c * 1000.0); rec[f'V_{c:g}'] = float(tq[tv < ws + c * 1000.0].sum())
                touch = np.nonzero(best == p)[0] if lvl > 0 else np.empty(0, int)
                rec['t_touch'] = (fr[touch[0]] - ws) / 1000.0 if len(touch) else np.nan
                rec['lb_touch'] = lb_at(fr[touch[0]]) if len(touch) else np.nan
                rec['t_sweep'] = (t_sw - ws) / 1000.0 if np.isfinite(t_sw) else np.nan
                if np.isfinite(t_sw):
                    j = np.searchsorted(fr, t_sw, side='left') - 1
                    rec['lb_sweep'] = lb_at(t_sw); rec['shown_pre_sweep'] = float(sz[j]) if j >= 0 and np.isfinite(sz[j]) else np.nan
                    rec['V_pre_sweep'] = float(tq[tv < t_sw].sum())
                else:
                    rec['lb_sweep'] = rec['shown_pre_sweep'] = rec['V_pre_sweep'] = np.nan
                r1.append(rec)
    # part 2: makers
    tot = lt.groupby('signer').shares.sum().sort_values(ascending=False); rank = pd.Series(np.arange(1, len(tot) + 1), index=tot.index)
    st['legs_1s_only'] += int((~timed).sum()); u = lt[timed]
    band = np.where(rank.values <= 5, 'top 5', np.where(rank.values <= 20, '6-20', '21+')); BAND = dict(zip(rank.index, band))
    for (mid, side), g in u.groupby(['mid', 'side'], sort=False):
        v = BOOK.get(mid)
        if v is None:
            st['legs_no_frame'] += len(g); continue
        tf, BP, BS, AP, AS = v; P, S = (BP, BS) if side == 0 else (AP, AS)
        ta = g.ta.values.astype(float); p = g.ptick.values; q = g.shares.values
        j0 = np.searchsorted(tf, ta, side='left') - 1; jn = np.searchsorted(tf, ta, side='right')
        ok = (j0 >= 0) & (jn < len(tf))
        j0c = np.clip(j0, 0, len(tf) - 1); jnc = np.clip(jn, 0, len(tf) - 1)
        b0 = P[j0c, 0]; bn = P[jnc, 0]; ok &= b0 >= 0; st['legs_no_frame'] += int((~ok).sum())   # no frame around it, or its side empty
        depth = np.clip((b0 - p) if side == 0 else (p - b0), 0, 2)
        cleared = (bn < p) if side == 0 else ((bn < 0) | (bn > p))
        for bd, dp, cl, qq in zip(g.signer.map(BAND).values[ok], depth[ok], cleared[ok], q[ok]):
            r2[(bd, int(dp), bool(cl))] += qq
        # trade-book alignment: at-best legs grouped per (price, frame pair); the frame after must show the size at that
        # price down by >= 90% of the traded shares, or the level gone
        at = ok & (depth == 0)
        if at.any():
            G = pd.DataFrame(dict(p=p[at], j0=j0c[at], jn=jnc[at], q=q[at])).groupby(['p', 'j0', 'jn'], sort=False).q.sum().reset_index()
            s0 = np.where(P[G.j0.values] == G.p.values[:, None], S[G.j0.values], 0.0).sum(1)
            sn = np.where(P[G.jn.values] == G.p.values[:, None], S[G.jn.values], 0.0).sum(1)
            good = (s0 - sn >= 0.9 * G.q.values - 1e-9) | (sn <= 1e-9)
            al['vol'] += float(G.q.sum()); al['vol_aligned'] += float(G.q.values[good].sum())
    del bk, lt, LTB, BOOK
    io.mem_check(f'day {d}')

A = pd.DataFrame(r1); M = pd.DataFrame([(k[0], k[1], k[2], v) for k, v in r2.items()], columns=['band', 'depth', 'cleared', 'q'])
A.to_parquet(f'{OUTD}/levels.parquet', index=False); M.to_parquet(f'{OUTD}/maker_volume.parquet', index=False)
print(f'part 1: {len(A)} levels at the open, {A.mid.nunique()} windows; statuses {st}')
print('  share of the open queue (share-weighted) - cancelled is a LOWER bound; traded = volume at that price')
out = {}
for lvl, g in A.groupby('level'):
    q0 = g.Q0.sum(); row = {f'cancel>= {c:g}s': g[f'lb_{c:g}'].sum() / q0 for c in CHK}
    row.update({f'traded {c:g}s': g[f'V_{c:g}'].sum() / q0 for c in (60.0, 240.0)})
    out[int(lvl)] = row
    print(f"  level {lvl}: Q0 {q0 / len(g):5.0f} sh/level  " + '  '.join(f"{k} {v:4.0%}" for k, v in row.items()))
print('  levels 2-5 that become the best: cancelled (lower bound) just before the touch; levels swept through: just before the sweep')
for lvl, g in A.groupby('level'):
    t = g[g.lb_touch.notna()]; s = g[g.lb_sweep.notna()]
    tt = f"touched {len(t) / len(g):4.0%} (median {t.t_touch.median():5.1f} s), cancel>= {t.lb_touch.sum() / max(t.Q0.sum(), 1e-9):4.0%}" if lvl > 1 else 'at best at the open'
    print(f"  level {lvl}: {tt} | swept {len(s) / len(g):4.0%} (median {s.t_sweep.median():5.1f} s): cancel>= {s.lb_sweep.sum() / max(s.Q0.sum(), 1e-9):4.0%}, "
          f"traded before {s.V_pre_sweep.sum() / max(s.Q0.sum(), 1e-9):4.0%}, shown just before / Q0 {s.shown_pre_sweep.sum() / max(s.Q0[s.shown_pre_sweep.notna()].sum(), 1e-9):4.0%}")
T = M.groupby(['band', 'depth']).apply(lambda g: pd.Series(dict(vol_k=g.q.sum() / 1e3, cleared=float(g.q[g.cleared].sum() / g.q.sum()))))
B = M.groupby('band').apply(lambda g: pd.Series(dict(vol_share=g.q.sum() / M.q.sum(), cleared=float(g.q[g.cleared].sum() / g.q.sum()))))
print(f"trade-book alignment: {al['vol_aligned'] / max(al['vol'], 1e-9):.1%} of timed at-best volume shows in the next frame (slack {SLACK:g} ms)")
print(f'part 2: {M.q.sum() / 1e6:.1f}M timed maker shares; share of volume whose trade cleared the level (depth 0 = at best, 2 = 2+ back)')
print(B.round(3).to_string()); print(T.round(3).unstack('depth').to_string())
io.write_json(f'{OUTD}/report.json', dict(statuses=st, slack_ms=SLACK, alignment=dict(al), part1=out, part2_band={k: v for k, v in B.round(4).to_dict('index').items()},
                                          part2_band_depth={f'{k[0]}|{k[1]}': v for k, v in T.round(4).to_dict('index').items()}))
