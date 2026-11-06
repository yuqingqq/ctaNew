"""Guard training grid on corrected timing (2026-11-04 review; STATUS 14:50Z). Same unit, label and statuses as pf_badfill_info.build
(rows every 500 ms of the live 0-240 s window; v = s (m(t_sw + 1 s) - p) if the known best p is swept through in (tau + 22, tau + 522] ms,
else 0; bad = v <= -2 c), with two corrections:
  - features from rows_rl_d2_pk (pending increments known per frame: pf_step5_features_rl with PF_PEND_KNOWN=frame)
  - trade legs at one time per taker transaction (key mid, exec_s, first_recv_ns, taker_index, taker_quote, price_executed): the
    earliest WS-timed leg, else the earliest E2 leg - the rule of pf_step4_live PF_TXTIME=ws; legs with neither stay excluded
Features: pf_model_spec ALL only (no outside-venue columns). Output step4/badfill_info_tx/grid.parquet.
PF_TXKEY=tk (second review, 2026-11-05): the transaction is the real taker order (step1/<day>/legs_tk.parquet from pf_txmap; unmapped
legs keep their own time); output step4/badfill_info_tx_tk/grid.parquet, compared with the old-key grid too. Labels stay on the
frame stamps (what a live fit would see); the replay draws the within-frame match time.
Usage: PF_RL_DDEC=2 python3 -m predictfun.research.pf_badfill_grid_tx
"""
import os
import numpy as np, pandas as pd
from . import pf_io as io
from . import pf_clock_rl as CK
from .pf_model_spec import SETS

assert CK.D_DEC == 2, 'run with PF_RL_DDEC=2'
DAYS = ['20261023', '20261024', '20261025', '20261026', '20261027', '20261031', '20261101', '20261102']; FRESH = ('20261101', '20261102')
FE = SETS['ALL']; L_CXL = 22.0; W = 500.0
ms_ = lambda d: pd.Timestamp(d, tz='UTC').value // 1_000_000
SEAL_MS = (ms_('2026-10-28'), ms_('2026-10-31')); FWD_MS = ms_('2026-11-04')
TXKEY = os.environ.get('PF_TXKEY', ''); assert TXKEY in ('', 'tk'), TXKEY
OUTD = f"{io.OUT}/step4/badfill_info_tx{'_tk' if TXKEY else ''}"; os.makedirs(OUTD, exist_ok=True)
st = dict(rows=0, no_book=0, no_mid_after=0, legs=0, legs_retimed=0, legs_untimed_excluded=0)
TXK = ['mid', 'exec_s', 'first_recv_ns', 'taker_index', 'taker_quote', 'price_executed']


def tx_times(lt, S1):
    ws_t = np.where(lt.how.values > 0, lt.t_ms.values.astype(float), np.nan); e2_t = np.where(lt.how.values == 0, lt.t_e2.values, np.nan)
    grp = [lt[c] for c in TXK]
    if TXKEY == 'tk':
        tk = pd.read_parquet(f'{S1}/legs_tk.parquet').tk.values; assert len(tk) == len(lt), 'legs_tk not aligned'
        grp = [pd.Series(np.where(tk >= 0, tk, -1 - np.arange(len(lt))), index=lt.index)]; st['legs_tk_unmapped'] = st.get('legs_tk_unmapped', 0) + int((tk < 0).sum())
    a = pd.Series(ws_t, index=lt.index).groupby(grp, sort=False, dropna=False).transform('min').values
    b = pd.Series(e2_t, index=lt.index).groupby(grp, sort=False, dropna=False).transform('min').values
    anc = np.where(np.isfinite(a), a, b); old = np.where(np.isfinite(ws_t), ws_t, e2_t)
    st['legs'] += len(lt); st['legs_retimed'] += int((np.isfinite(anc) & np.isfinite(old) & (np.abs(anc - old) > 1e-6)).sum())
    st['legs_untimed_excluded'] += int((~np.isfinite(anc)).sum())
    return anc


def build(d):
    S1 = f'{io.OUT}/step1/{d}'; shift = CK.load_shift(S1)
    r = pd.read_parquet(f'{io.OUT}/step5/{d}/rows_rl_d2_pk.parquet', columns=list(dict.fromkeys(['mid', 'tau', 't_rel', 'y1000'] + FE)))
    pr = pd.read_parquet(f"{io.OUT}/step5/{d}/{'preds_c3' if d in FRESH else 'preds_o3'}.parquet", columns=['pred_1000'])
    r['pred_1000'] = pr.pred_1000.values; del pr
    keep = ((r.t_rel < 240) & r.pf_mid.notna() & r.y1000.notna() & r.pred_1000.notna()).values & ((np.floor(r.tau.values / 50.0).astype(np.int64) % 10) == 0)
    r = r[keep].reset_index(drop=True); st['rows'] += len(r)
    bk = pd.concat([pd.read_parquet(f'{S1}/books/{f}', columns=['mid', 't_ms', 'recv_ns', 'b1_p', 'a1_p']) for f in sorted(os.listdir(f'{S1}/books'))])
    bk.sort_values(['mid', 't_ms', 'recv_ns'], inplace=True, kind='stable')
    t_ = bk.t_ms.values; okd = ~((t_ >= SEAL_MS[0]) & (t_ < SEAL_MS[1])) & (t_ < FWD_MS)
    st['book_frames_outside_admissible_days'] = st.get('book_frames_outside_admissible_days', 0) + int((~okd).sum()); bk = bk[okd]   # 2026-11-05: no sealed / forward input
    lt = pd.read_parquet(f'{S1}/legs_timed2.parquet', columns=TXK + ['side', 'ptick', 't_ms', 'how', 't_e2'])
    lt['ta'] = tx_times(lt, S1); lt = lt[np.isfinite(lt.ta.values)]
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
    return r


R = pd.concat([build(d) for d in DAYS], ignore_index=True)
R[list(dict.fromkeys(['day', 'mid', 'tau', 't_rel', 'y1000', 'pred_1000', 'v_bid', 'v_ask', 'bad_bid', 'bad_ask'] + FE))].to_parquet(f'{OUTD}/grid.parquet', index=False)
O = pd.read_parquet(f'{io.OUT}/step4/badfill_info/grid.parquet', columns=['day', 'mid', 'tau', 'v_bid', 'v_ask', 'bad_bid', 'bad_ask'])
J = R[['day', 'mid', 'tau', 'v_bid', 'v_ask', 'bad_bid', 'bad_ask']].merge(O, on=['day', 'mid', 'tau'], how='inner', suffixes=('', '_old'))
cmp_ = {f'{c}_changed': float(np.mean(J[c].values != J[f'{c}_old'].values)) for c in ('v_bid', 'v_ask', 'bad_bid', 'bad_ask')}
cmp_.update(rows_new=len(R), rows_old=len(O), rows_joined=len(J), bad_share_new=float(np.r_[R.bad_bid, R.bad_ask].mean()), bad_share_old=float(np.r_[O.bad_bid, O.bad_ask].mean()))
print(f'statuses {st}; comparison with the per-leg grid {cmp_}', flush=True)
rep = dict(statuses=st, vs_old=cmp_)
if TXKEY:
    O2 = pd.read_parquet(f'{io.OUT}/step4/badfill_info_tx/grid.parquet', columns=['day', 'mid', 'tau', 'v_bid', 'v_ask', 'bad_bid', 'bad_ask'])
    J2 = R[['day', 'mid', 'tau', 'v_bid', 'v_ask', 'bad_bid', 'bad_ask']].merge(O2, on=['day', 'mid', 'tau'], how='inner', suffixes=('', '_old'))
    rep['vs_old_key'] = {f'{c}_changed': float(np.mean(J2[c].values != J2[f'{c}_old'].values)) for c in ('v_bid', 'v_ask', 'bad_bid', 'bad_ask')}
    rep['vs_old_key'].update(rows_joined=len(J2), bad_share_old_key=float(np.r_[O2.bad_bid, O2.bad_ask].mean()))
    print('comparison with the old-key grid', rep['vs_old_key'], flush=True)
io.write_json(f'{OUTD}/report.json', rep)
