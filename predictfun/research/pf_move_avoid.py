"""MOVE-AVOID (POST HOC, 2026-11-12, session ce68ebe0): is MOVE's AUC 0.87 for DN_250 avoidable harm, or moves that already happened?

MOVE (pf_move_info.py, e62e1fbd / 8d8fdd20) found that the guard's own inputs, fitted on DN_250 (the side's best moves worse than the
known best p in (t + 24, t + 274]), reach an out-of-day AUC of 0.871 (activity 0.600), and that no new source adds to it. Frames are
known ~15-60 ms after their venue stamp, so some DN_250 = 1 rows may be levels already gone before our cancel could act (at t + 24):
a pull cannot avoid those. This check, written after MOVE's result (post hoc; no reading declared), splits them:
  GONE   the side's best in the last frame stamped <= t + 24 is already worse than p (moved while unseen / before the cancel lands)
  AV     DN_250 and not GONE: the level is still there at t + 24 and goes within (t + 24, t + 274] - harm a pull could avoid
Rows are rebuilt exactly as MOVE's (same windows, every 10th grid step, same exclusions); alignment with MOVE's saved rows (m, t) is
asserted. Fits: the guard's inputs (MOVE's BASE), LightGBM pinned parameters, leave one day out: (1) on DN_250, all rows - must
reproduce MOVE's 0.8707 within 0.0005 (control); its AUC on the rows still present at t + 24 for AV; (2) fitted on AV among the rows
still present at t + 24. Also: the share of DN_250 that is GONE, and the top features by gain of a fit on all 9 days (descriptive).
Usage: research/run_light.sh move_avoid -- python3 predictfun/research/pf_move_avoid.py day <yyyymmdd> | report
"""
import bisect
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import pf_kp_s4 as K                                      # noqa: E402
import pf_move_info as MV                                 # noqa: E402
import pf_model_spec as MS                                # noqa: E402


def run_day(day):
    D = K.load_day(day); z = np.load(f'{MV.OUTM}/day_{day}.npz')
    mz, tz = z['B|m'], z['B|t']; gone = []; still = []; mm = []; tt = []
    for w in K.window_sides(D):
        tf = w['tf']; tfl = tf.tolist(); bb = w['P5'][:, 0]; side = w['side']; X = w['X']; nb = len(K.G32)
        for i in range(0, w['n'], MV.STEP):
            t = float(w['tg'][i]); p = w['best'][i]
            if p < 0 or w['jk'][i] < 0:
                continue
            v = X[i, nb]
            if not np.isfinite(v) or v == -1.0:
                continue
            j = bisect.bisect_right(tfl, t + MV.L_ACT) - 1
            b = bb[j] if j >= 0 else -1
            g = (b < p) if side == 0 else (b > p or b < 0)
            gone.append(int(g)); mm.append(w['m']); tt.append(t)
    mm = np.array(mm, float); tt = np.array(tt, float)
    if len(mm) != len(mz) or not (np.array_equal(mm, mz) and np.allclose(tt, tz)):
        sys.exit(f'MOVE-AVOID: rows not aligned with MOVE on {day} ({len(mm)} vs {len(mz)})')
    np.save(f'{MV.OUTM}/avoid_{day}.npy', np.array(gone, np.int8))
    print(f'{day}: {len(gone)} rows aligned, GONE {np.mean(gone):.3f}', flush=True)


def report():
    import lightgbm as lgb
    B = {}; gone = []
    for d in MV.DAYS9:
        z = np.load(f'{MV.OUTM}/day_{d}.npz')
        for k in MV.BASE_N + ['day', 'DN_250']:
            B.setdefault(k, []).append(z[f'B|{k}'])
        gone.append(np.load(f'{MV.OUTM}/avoid_{d}.npy'))
    B = {k: np.concatenate(v) for k, v in B.items()}; gone = np.concatenate(gone).astype(bool)
    day = B['day'].astype(int); dz = [int(d) for d in MV.DAYS9]; y = B['DN_250'].astype(int)
    Xm = np.column_stack([B[k] for k in MV.BASE_N]).astype(np.float32)
    av = (y == 1) & ~gone; still = ~gone
    p1 = MV.lgb_oof(Xm, y, day, dz, 'binary')
    a_all = MV.auc(y, p1); a_av = MV.auc(av[still], p1[still])
    ctrl = abs(a_all - 0.8707) <= 0.0005
    p2 = np.full(len(y), np.nan); p2[still] = MV.lgb_oof(Xm[still], av[still].astype(int), day[still], dz, 'binary')
    a2 = MV.auc(av[still], p2[still]); a2d = [MV.auc(av[still & (day == d)], p2[still & (day == d)]) for d in dz]
    a1d = [MV.auc(av[still & (day == d)], p1[still & (day == d)]) for d in dz]
    pr = dict(MS.PARAMS); pr['objective'] = 'binary'
    mdl = lgb.train(pr, lgb.Dataset(Xm[still], av[still].astype(int)), MS.ROUNDS)
    gain = sorted(zip(MV.BASE_N, mdl.feature_importance('gain')), key=lambda kv: -kv[1]); tot = sum(g for _, g in gain)
    L = [f'MOVE-AVOID (POST HOC) as of {time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}; rows {len(y)}; control (DN_250 refit = MOVE 0.8707): '
         f'{"passed" if ctrl else "FAILED"} ({a_all:.4f})',
         f'DN_250 rows {int(y.sum())} ({y.mean():.3f}); of them GONE before t + 24 {int((y.astype(bool) & gone).sum())} ({(y.astype(bool) & gone).sum() / max(y.sum(), 1):.1%}); '
         f'rows still present at t + 24 {int(still.sum())}, AV (goes in (t + 24, t + 274]) {int(av.sum())} ({av[still].mean():.3f} of them)',
         f'AUC on AV among rows still present at t + 24: the DN_250 fit {a_av:.4f} (days {" ".join(f"{x:.3f}" for x in a1d)}); '
         f'fitted on AV itself {a2:.4f} (days {" ".join(f"{x:.3f}" for x in a2d)})',
         'top features by gain (AV fit, all 9 days): ' + ', '.join(f'{k} {g / tot:.1%}' for k, g in gain[:10])]
    if not ctrl:
        L.append('CONTROL FAILED: the refit does not reproduce MOVE; numbers above are not to be used')
    os.makedirs(MV.OUTM, exist_ok=True)
    json.dump(dict(auc_all=a_all, auc_av_dnfit=a_av, auc_av_avfit=a2, days_dnfit=a1d, days_avfit=a2d, gone_share_of_dn=float((y.astype(bool) & gone).sum() / max(y.sum(), 1)),
                   av_rate=float(av[still].mean()), gain=[(k, float(g / tot)) for k, g in gain], control=bool(ctrl)), open(f'{MV.OUTM}/avoid.json', 'w'), indent=1)
    open(f'{MV.OUTM}/avoid.txt', 'w').write('\n'.join(L) + '\n'); print('\n'.join(L))
    if not ctrl:
        sys.exit(1)


def ops():
    """Operating points of the AV fit (added after the AV result, post hoc): for recall 50 / 70 / 90% of AV among rows still present
    at t + 24, the share of those moments flagged (how often a front order would pull) and the precision; and the AUC among the moments
    with a WS-timed trade at the level in the window (ACT_250 = 1): clearing (CLR_250) vs the level standing."""
    B = {}; gone = []
    for d in MV.DAYS9:
        z = np.load(f'{MV.OUTM}/day_{d}.npz')
        for k in MV.BASE_N + ['day', 'DN_250', 'ACT_250', 'CLR_250']:
            B.setdefault(k, []).append(z[f'B|{k}'])
        gone.append(np.load(f'{MV.OUTM}/avoid_{d}.npy'))
    B = {k: np.concatenate(v) for k, v in B.items()}; gone = np.concatenate(gone).astype(bool)
    day = B['day'].astype(int); dz = [int(d) for d in MV.DAYS9]; y = B['DN_250'].astype(int); still = ~gone; av = (y == 1) & still
    Xm = np.column_stack([B[k] for k in MV.BASE_N]).astype(np.float32)
    p2 = MV.lgb_oof(Xm[still], av[still].astype(int), day[still], dz, 'binary')
    a2 = MV.auc(av[still], p2)
    if abs(a2 - 0.8410) > 0.0005:
        sys.exit(f'MOVE-AVOID ops: refit AUC {a2:.4f} does not reproduce 0.8410')
    yy = av[still]; L = [f'MOVE-AVOID ops (POST HOC) as of {time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}; AV fit reproduced ({a2:.4f}); '
                         f'rows still present at t + 24 {int(still.sum())}, AV rate {yy.mean():.3f}']
    o = np.sort(p2[yy])[::-1]; R = {}
    for rc in (0.5, 0.7, 0.9):
        thr = o[int(np.ceil(rc * len(o))) - 1]; fl = p2 >= thr
        R[rc] = dict(flag_share=float(fl.mean()), precision=float(yy[fl].mean()), recall=float(fl[yy].mean()))
        L.append(f'  recall {rc:.0%}: flags {fl.mean():.1%} of moments, precision {yy[fl].mean():.1%} (false alarms {1 - yy[fl].mean():.1%})')
    act = B['ACT_250'][still].astype(bool); clr = B['CLR_250'][still].astype(bool)
    aa = MV.auc(clr[act], p2[act]); aad = [MV.auc(clr[act & (day[still] == d)], p2[act & (day[still] == d)]) for d in dz]
    L.append(f'  among moments with a WS-timed trade at the level in the window ({int(act.sum())}, cleared {clr[act].mean():.1%}): AUC cleared vs standing '
             f'{aa:.4f} (days {" ".join(f"{x:.3f}" for x in aad)})')
    json.dump(dict(auc_av=a2, ops={str(k): v for k, v in R.items()}, auc_clr_given_act=aa, days=aad), open(f'{MV.OUTM}/avoid_ops.json', 'w'), indent=1)
    open(f'{MV.OUTM}/avoid_ops.txt', 'w').write('\n'.join(L) + '\n'); print('\n'.join(L))


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'ops':
        ops(); sys.exit(0)
    if len(sys.argv) > 2 and sys.argv[1] == 'day':
        run_day(sys.argv[2])
    elif len(sys.argv) > 1 and sys.argv[1] == 'report':
        report()
    else:
        sys.exit(__doc__)
