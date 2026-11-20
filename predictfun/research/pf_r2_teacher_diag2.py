"""THE pi0_50_v1 IN-SAMPLE DIAGNOSTICS, v2 (review 8cc1dc49 P3; 2026-11-20, session aea5bfb9). This supersedes the value columns of
pf_r2_teacher.diag. Those divided the coherent rows' weighted sums by the weight of ALL rows, which is neither the coherent-conditional mean
nor the all-row mean: the excluded rows counted as zero.
This module is outside the frozen teacher closure. It verifies the freeze (pf_r2_teacher.frozen) and loads the 16 frozen models; nothing in
the freeze, the models or the policy changes.
Per side / head / horizon bin, weighted by the normalized census weights:
 - rows and their weight W; the coherent rows (the mapping's ok) and their weight W_c;
 - V_pred_coh_mean = sum_c w V_pred / W_c and V_real_coh_mean = sum_c w V_real / W_c: the coherent-conditional means;
 - V_real_all_mean = sum w V_real / W: the full population's realized mean, which includes the rows where the teacher falls back;
 - the unweighted coherent sums, as before;
 - the fill, conditional quantity / movement and severe columns, computed exactly as in v1.
 Bins are named by pandas' right-closed intervals: (0, 100] is the short bin, and the complementary statement is "> 100 ms".
Usage (run_light): pf_r2_teacher_diag2.py selftest | run"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pf_r2_teacher as T                                 # noqa: E402

C, pd = T.C, T.pd
MARK = 'R2 TEACHER DIAG2'


def value_stats(vp, vr, w, ok):
    """The explicit value estimands of one cell: W, W_c, the coherent-conditional predicted / realized means, the all-row realized mean and the
    unweighted coherent sums (NaN when W_c is 0)."""
    vp, vr, w, ok = tuple(np.asarray(x, float) for x in (vp, vr, w)) + (np.asarray(ok, bool),)
    W, Wc = float(w.sum()), float(w[ok].sum())
    return dict(W=W, W_c=Wc, n_c=int(ok.sum()), V_pred_coh_mean=float((w[ok] * vp[ok]).sum() / Wc) if Wc > 0 else np.nan,
                V_real_coh_mean=float((w[ok] * vr[ok]).sum() / Wc) if Wc > 0 else np.nan, V_real_all_mean=float((w * vr).sum() / W) if W > 0 else np.nan,
                V_pred_sum_coh=float(vp[ok].sum()), V_real_sum_coh=float(vr[ok].sum()))


def run(cfg=T.REAL, chunk=200000):
    """Every training row of the frozen models -> diag2.json + diag2_horizon.csv in out/diag2 (per side / head / horizon; the v1 fill /
    quantity / movement / severe columns and the v2 value estimands)."""
    fz = T.frozen(cfg); man = T.train_manifest(cfg); M = T.load_models(cfg, fz); out = cfg['out'] / 'diag2'; out.mkdir(exist_ok=False); rows = []
    for s, h in T.GROUPS:
        d = cfg['out'] / 'train' / f'{s}_{h}'; n = man['groups'][f'{s}_{h}']['rows']; sh = T.sh_of(s, h); r = T.Rows(d / 'X_all.f32', n)
        L = {c: T.lab(d, c, 'all', n) for c in T.LAB}; wn = T.norm_weights(L['weight']); hb = T.hbin(T.END - L['decision_ms']); nz = L['q'] > T.FILLED
        P = np.empty(n); V = np.empty(n); OK = np.empty(n, bool); S = np.empty(n); QF = np.full(n, np.nan); MQ = np.full(n, np.nan)
        for a in range(0, n, chunk):
            b = min(a + chunk, n); Xn = r.block(a, b); Xc = pd.DataFrame(Xn, columns=T.FEATS); Xc['price'] = L['price'][a:b]
            pr = T.predict_rows(M[(s, h)], Xc, s); P[a:b], V[a:b], OK[a:b], S[a:b] = pr['P'], pr['V'], pr['ok'], pr['S']; f_ = nz[a:b]
            if f_.any():
                QF[a:b][f_] = np.clip(M[(s, h)]['q'].predict(Xn[f_], num_threads=1), 0., Xc.fp_remaining.to_numpy(float)[f_])
                MQ[a:b][f_] = M[(s, h)]['qm'].predict(Xn[f_], num_threads=1)
        r.close()
        for hv in sorted(set(hb.tolist())):
            m_ = hb == hv; w_ = wn[m_]; f_ = nz[m_]; wf = w_[f_]; W = w_.sum(); Wf = wf.sum()
            vs = value_stats(V[m_], L['value'][m_], w_, OK[m_])
            rows.append(dict(side_head=sh, horizon=hv, rows=int(m_.sum()), filled_rows=int(f_.sum()), weight=float(W), pred_fill=float((w_ * P[m_]).sum() / W),
                             real_fill=float((w_ * f_).sum() / W), cond_q_pred=float((wf * QF[m_][f_]).sum() / Wf) if Wf else np.nan,
                             cond_q_real=float((wf * L['q'][m_][f_]).sum() / Wf) if Wf else np.nan,
                             cond_qm_pred=float((wf * MQ[m_][f_]).sum() / Wf) if Wf else np.nan, cond_qm_real=float((wf * L['qm'][m_][f_]).sum() / Wf) if Wf else np.nan,
                             severe_pred=float((w_ * S[m_]).sum() / W), severe_real=float((w_ * L['severe'][m_]).sum() / W), coherent_rows=vs['n_c'],
                             incoherent_rows=int(m_.sum()) - vs['n_c'], coherent_weight=vs['W_c'], V_pred_coh_mean_c=vs['V_pred_coh_mean'],
                             V_real_coh_mean_c=vs['V_real_coh_mean'], V_real_all_mean_c=vs['V_real_all_mean'], V_pred_sum_coh_c=vs['V_pred_sum_coh'],
                             V_real_sum_coh_c=vs['V_real_sum_coh']))
        tot = value_stats(V, L['value'], wn, OK)
        rows.append(dict(side_head=sh, horizon='all', rows=n, filled_rows=int(nz.sum()), weight=float(wn.sum()), pred_fill=float((wn * P).sum() / wn.sum()),
                         real_fill=float((wn * nz).sum() / wn.sum()), coherent_rows=tot['n_c'], incoherent_rows=n - tot['n_c'], coherent_weight=tot['W_c'],
                         V_pred_coh_mean_c=tot['V_pred_coh_mean'], V_real_coh_mean_c=tot['V_real_coh_mean'], V_real_all_mean_c=tot['V_real_all_mean'],
                         V_pred_sum_coh_c=tot['V_pred_sum_coh'], V_real_sum_coh_c=tot['V_real_sum_coh']))
    pd.DataFrame(rows).to_csv(out / 'diag2_horizon.csv', index=False)
    (out / 'diag2.json').write_text(json.dumps(dict(asof=C.stamp(), freeze_sha256=C.sha(cfg['out'] / 'freeze.json'), supersedes='diag (value columns; review 8cc1dc49 P3)',
                                                    rows=rows), indent=1, default=str))
    return rows


def selftest():
    """Known answers. Four rows, the last incoherent: w = (1, 1, 2, 4), V_pred = (2, 4, 1, 100), V_real = (1, 3, 0, -50).
    W = 8, W_c = 4; the coherent-conditional means are 8 / 4 = 2 (predicted) and 4 / 4 = 1 (realized); the all-row realized mean is
    (1 + 3 + 0 - 200) / 8 = -24.5; v1's mixed estimand would give 8 / 8 = 1 and 4 / 8 = 0.5, which must differ. An all-incoherent cell gives NaN means."""
    vs = value_stats([2., 4., 1., 100.], [1., 3., 0., -50.], [1., 1., 2., 4.], [True, True, True, False])
    want = dict(W=8., W_c=4., n_c=3, V_pred_coh_mean=2., V_real_coh_mean=1., V_real_all_mean=-24.5, V_pred_sum_coh=7., V_real_sum_coh=4.)
    bad = [k for k, v in want.items() if not np.isclose(vs[k], v, rtol=0, atol=1e-12)]
    v1_pred, v1_real = 8. / 8., 4. / 8.
    if np.isclose(vs['V_pred_coh_mean'], v1_pred) or np.isclose(vs['V_real_coh_mean'], v1_real):
        bad.append('the coherent mean equals the v1 mixed estimand')
    z = value_stats([1.], [1.], [1.], [False])
    if not (np.isnan(z['V_pred_coh_mean']) and np.isnan(z['V_real_coh_mean']) and z['V_real_all_mean'] == 1.):
        bad.append(f'an all-incoherent cell: {z}')
    if bad:
        sys.exit(f'{MARK} SELFTEST FAILED: {bad} ({vs})')
    print(f'{MARK} selftest passed: the coherent-conditional and all-row estimands give their known answers; v1 mixed form differs')


if __name__ == '__main__':
    a = sys.argv[1:]
    if a == ['selftest']:
        selftest()
    elif a == ['run']:
        print(json.dumps([r for r in run() if r['horizon'] == 'all'], default=str))
    else:
        sys.exit('usage: pf_r2_teacher_diag2.py selftest | run')
