"""Per-day guard grid for pm_ladder_v2's daily refit (pf_v2_refit.py reads step4/badfill_info_tx_tk/days/<d>.parquet for days after the 8
read days). Same unit, label, statuses and features as pf_badfill_grid_tx.py with PF_TXKEY=tk (build() below is that file's build() with
two changes): book frames are cut at the end of the day being built (d + 1 day 00Z; the refit for T runs at T 00Z) instead of at the fixed
2026-11-04, and the 1 s prediction file is preds_o3 before 11-01, preds_c3 from 11-01 (that file's FRESH rule extended to later days).
Labels never need a frame after the day: rows end at t_rel < 240 s of the last window (23:59:00), sweeps within 544 ms, mid 1 s later.
Parity (PF_V2_GRID_CHECK=1): building 20261031 (sealed-day cut) and 20261102 (preds_c3) reproduces their rows of grid.parquet exactly,
every column; falsifier: the same build with the cancel latency 22 -> 23 ms must differ.
Usage: PF_RL_DDEC=2 python3 -m predictfun.research.pf_v2_grid_day <day> ...   |   PF_V2_GRID_CHECK=1 PF_RL_DDEC=2 python3 -m ...
"""
import os, sys
import numpy as np, pandas as pd
from . import pf_io as io
from . import pf_clock_rl as CK
from .pf_model_spec import SETS

assert CK.D_DEC == 2, 'run with PF_RL_DDEC=2'
FE = SETS['ALL']; W = 500.0
ms_ = lambda d: pd.Timestamp(d, tz='UTC').value // 1_000_000
SEAL_MS = (ms_('2026-10-28'), ms_('2026-10-31')); SEALED = ('20261028', '20261029', '20261030')
GDIR = f'{io.OUT}/step4/badfill_info_tx_tk'; COLS = ['day', 'mid', 'tau', 't_rel', 'y1000', 'pred_1000', 'v_bid', 'v_ask', 'bad_bid', 'bad_ask']
TXK = ['mid', 'exec_s', 'first_recv_ns', 'taker_index', 'taker_quote', 'price_executed']


def tx_times(lt, S1, st):
    ws_t = np.where(lt.how.values > 0, lt.t_ms.values.astype(float), np.nan); e2_t = np.where(lt.how.values == 0, lt.t_e2.values, np.nan)
    tk = pd.read_parquet(f'{S1}/legs_tk.parquet').tk.values; assert len(tk) == len(lt), 'legs_tk not aligned'
    grp = [pd.Series(np.where(tk >= 0, tk, -1 - np.arange(len(lt))), index=lt.index)]; st['legs_tk_unmapped'] += int((tk < 0).sum())
    a = pd.Series(ws_t, index=lt.index).groupby(grp, sort=False, dropna=False).transform('min').values
    b = pd.Series(e2_t, index=lt.index).groupby(grp, sort=False, dropna=False).transform('min').values
    anc = np.where(np.isfinite(a), a, b); old = np.where(np.isfinite(ws_t), ws_t, e2_t)
    st['legs'] += len(lt); st['legs_retimed'] += int((np.isfinite(anc) & np.isfinite(old) & (np.abs(anc - old) > 1e-6)).sum())
    st['legs_untimed_excluded'] += int((~np.isfinite(anc)).sum())
    return anc


def build(d, L_CXL=22.0):
    if d in SEALED:
        sys.exit(f'REFUSED: {d} is sealed')
    st = dict(rows=0, no_book=0, no_mid_after=0, legs=0, legs_retimed=0, legs_untimed_excluded=0, legs_tk_unmapped=0, book_frames_outside=0)
    END_MS = ms_(pd.Timestamp(d) + pd.Timedelta(days=1))
    S1 = f'{io.OUT}/step1/{d}'; shift = CK.load_shift(S1)
    r = pd.read_parquet(f'{io.OUT}/step5/{d}/rows_rl_d2_pk.parquet', columns=list(dict.fromkeys(['mid', 'tau', 't_rel', 'y1000'] + FE)))
    pr = pd.read_parquet(f"{io.OUT}/step5/{d}/{'preds_c3' if d >= '20261101' else 'preds_o3'}.parquet", columns=['pred_1000'])
    r['pred_1000'] = pr.pred_1000.values; del pr
    keep = ((r.t_rel < 240) & r.pf_mid.notna() & r.y1000.notna() & r.pred_1000.notna()).values & ((np.floor(r.tau.values / 50.0).astype(np.int64) % 10) == 0)
    r = r[keep].reset_index(drop=True); st['rows'] += len(r)
    bk = pd.concat([pd.read_parquet(f'{S1}/books/{f}', columns=['mid', 't_ms', 'recv_ns', 'b1_p', 'a1_p']) for f in sorted(os.listdir(f'{S1}/books'))])
    bk.sort_values(['mid', 't_ms', 'recv_ns'], inplace=True, kind='stable')
    t_ = bk.t_ms.values; okd = ~((t_ >= SEAL_MS[0]) & (t_ < SEAL_MS[1])) & (t_ < END_MS)
    st['book_frames_outside'] += int((~okd).sum()); bk = bk[okd]
    lt = pd.read_parquet(f'{S1}/legs_timed2.parquet', columns=TXK + ['side', 'ptick', 't_ms', 'how', 't_e2'])
    lt['ta'] = tx_times(lt, S1, st); lt = lt[np.isfinite(lt.ta.values)]
    LB = {k: (x.ta.values[np.argsort(x.ta.values, kind='stable')].astype(float), x.ptick.values[np.argsort(x.ta.values, kind='stable')]) for k, x in lt.groupby(['mid', 'side'], sort=False)}
    BKD = {m: g for m, g in bk.groupby('mid', sort=False)}; del bk, lt
    v = np.zeros((len(r), 2)); bad = np.zeros((len(r), 2), bool); ok = np.ones(len(r), bool)
    for mid, idx in r.groupby('mid', sort=False).indices.items():
        g = BKD.get(mid)
        if g is None:
            ok[idx] = False; st['no_book'] += len(idx); continue
        kf = np.maximum.accumulate(CK.pf_known(g.t_ms.values, g.recv_ns.values, shift)); b1 = g.b1_p.values; a1 = g.a1_p.values
        tf = g.t_ms.values.astype(float); vm = (b1 >= 0) & (a1 > b1); tv = tf[vm]; mv = ((b1 + a1) / 2.0)[vm]
        tau = r.tau.values[idx].astype(float); j = np.searchsorted(kf, tau, side='right') - 1
        good = (j >= 0); jj = np.clip(j, 0, None); pb = np.where(good, b1[jj], -1); pa = np.where(good, a1[jj], -1)
        good &= (pb >= 0) & (pa > pb); st['no_book'] += int((~good).sum()); ok[idx[~good]] = False
        for si, (s, p) in enumerate(((1.0, pb), (-1.0, pa))):
            L = LB.get((mid, si))
            if L is None:
                continue
            T, PT = L; i0 = np.searchsorted(T, tau + L_CXL, side='right'); i1 = np.searchsorted(T, tau + L_CXL + W, side='right')
            for n in np.flatnonzero(good & (i1 > i0)):
                seg = PT[i0[n]:i1[n]]; hit = np.flatnonzero(seg < p[n]) if s > 0 else np.flatnonzero(seg > p[n])
                if len(hit):
                    tsw = T[i0[n] + hit[0]]; k = np.searchsorted(tv, tsw + 1000.0, side='right') - 1
                    if k < 0:
                        st['no_mid_after'] += 1; continue
                    v[idx[n], si] = s * (mv[k] - p[n]); bad[idx[n], si] = v[idx[n], si] <= -2
    r['v_bid'] = v[:, 0]; r['v_ask'] = v[:, 1]; r['bad_bid'] = bad[:, 0]; r['bad_ask'] = bad[:, 1]
    r = r[ok].reset_index(drop=True); r['day'] = d
    io.mem_check(f'build {d}')
    return r[list(dict.fromkeys(COLS + FE))], st


def same(a, b):
    if list(a.columns) != list(b.columns) or len(a) != len(b):
        return False
    for c in a.columns:
        x, y = a[c].values, b[c].values
        if x.dtype.kind == 'f':
            if not (np.array_equal(np.isnan(x), np.isnan(y)) and np.array_equal(x[~np.isnan(x)], y[~np.isnan(y)])):
                return False
        elif not np.array_equal(x, y):
            return False
    return True


def check():
    rep = {}
    for d in ('20261031', '20261102'):
        G = pd.read_parquet(f'{GDIR}/grid.parquet', filters=[('day', '==', d)])
        R, st = build(d)
        if not same(R.reset_index(drop=True), G[list(R.columns)].reset_index(drop=True)):
            sys.exit(f'REFUSED: the per-day build of {d} does not reproduce grid.parquet ({len(R)} vs {len(G)} rows)')
        del R
        Rf, _ = build(d, L_CXL=23.0)
        if same(Rf.reset_index(drop=True), G[list(Rf.columns)].reset_index(drop=True)):
            sys.exit(f'FALSIFIER DID NOT FIRE: {d} with a 23 ms cancel latency equals grid.parquet')
        rep[d] = dict(rows=len(G), equal=True, falsifier_fired=True, statuses=st); del Rf, G
        print(d, rep[d], flush=True)
    io.write_json(f'{GDIR}/days_check.json', rep)


def main():
    if os.environ.get('PF_V2_GRID_CHECK') == '1':
        check(); return 0
    os.makedirs(f'{GDIR}/days', exist_ok=True)
    for d in sys.argv[1:]:
        R, st = build(d); R.to_parquet(f'{GDIR}/days/{d}.parquet', index=False)
        io.write_json(f'{GDIR}/days/{d}_statuses.json', st); print(d, st, flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
