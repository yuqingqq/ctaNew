"""KEEP-PLACE (2026-11-13, session ce68ebe0; after DEEP-PULL, 63e7d81e: the guard trigger on every rung (LDG) brings the ladder near break-even
(-$137 a day at M 50, -$16 at M 0) and +$781 at the front bound; what is left is queue place at the deep levels: 61-69% of our deep shares
swept against the top makers' 36%, ~75k pulls a day re-queueing the rungs. USER: "Yes go test, check the performance result, then review
our gaps to top makers"): fewer, stickier pulls so the rungs keep their place.

Simulator, sizes, rungs, signal and lines as DEEP-PULL (pf_deep_pull.py, ffc7e475): S4's rest lane, 9 consumed replay days, first 60 s,
30-sh orders kept through fills, 6 rungs per side (d = 0..5 ticks behind the best), FRONT-PULL's per-day P models and 90% line.
Arms (fill clocks M 0 / 50 / 250 and U 50, the front bound, descriptive):
  KPS       every rung pulled and kept out only while the guard trigger (v <= -0.10) AND P >= the 90% line hold (strong signal)
  KPN       entry of every rung blocked while the guard trigger holds; a resting rung is pulled on the guard trigger only while it is within
            1 tick of the known best (deeper rungs are kept: the levels ahead protect them)
  KPF       LDG with immediate re-entry: re-placed at the first step the trigger is clear (S4's 250 ms clearance set to 0 for this arm)
  LDG       DEEP-PULL's guard ladder (reference); J-v3 (5 sh, reference)
Readings DECLARED BEFORE THE FIRST RUN (day rule >= 7 of 9; $/day w/o young at 1 s; three candidate arms, no primary):
 R440 KEEP-PLACE (each of KPS, KPN, KPF against LDG at M 0 and at M 50): higher on >= 7 days at both -> HELPS; lower on >= 7 at both ->
      HURTS; else NOT-SHOWN.
 R441 POSITIVE: each arm's pooled $/day > 0 at M 0 / 50 / 250 -> 'k of 3'.
 R442 VS-V3: each arm against J-v3 at 1 s and 5 s x M 0 / 50 / 250 -> BEATS-V3 if 6 of 6, else 'k of 6'.
 R443 MECHANISM (descriptive): pulls a day; rungs 1-5's swept share and high-P share; front share.
Controls (any failure -> no output): FRONT-PULL-SIZE's self-tests (incl. the size / offset switches); C1 at every rung; P aligned with
FRONT-PULL's fit rows; age bins add up; the clearance is restored after KPF (checked); memory self-guard.
Limits: replay rule M (B1a); rungs are independent orders without inventory control; consumed days; designed after DEEP-PULL (three
tuning candidates on seen days: a pass is a candidate for a forward test only).
Usage (run_light, PF_SLOT=claude): pf_keep_place.py eval <day> | report; driver run_keep_place.sh
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import pf_kp_s4 as K                                      # noqa: E402
import pf_front_pull as FP                                # noqa: E402
import pf_front_pull_size as FS                           # noqa: E402

DAYS9 = K.DAYS9
OUTP = f'{K.IO}/keep_place_20261113'
EVAL = FS.EVAL
TERC = (0.09040728596733005, 0.4236075378593026)          # DEPTH-TIMING's terciles of P over real deep fills (depth_timing.json)
ARMS = ['J-v3', 'KPS', 'KPN', 'KPF', 'LDG']
SUB = ('all', 'r0', 'r1-5')
MEM_CAP_MIB = 2048.0


def fp_eval(day):
    import lightgbm as lgb
    FS.all_selftests()
    fit = json.load(open(f'{FP.OUTF}/fit.json'))['days'][day]; L = fit['lines']; l50, l75, l90 = L['0.50'], L['0.75'], L['0.90']
    mdl = lgb.Booster(model_file=f'{FP.OUTF}/models/av_{day}.txt')
    z = np.load(f'{FP.MOVE_DIR}/day_{day}.npz'); zm = z['B|m'].astype(np.int64); zt = z['B|t']; zs = z['B|side'].astype(int)
    pz = np.load(f'{FP.OUTF}/p_rows_{day}.npy')
    ROWS = {(int(m_), int(s_)): (zt[ix], pz[ix]) for (m_, s_), ix in pd.DataFrame(dict(m=zm, s=zs)).groupby(['m', 's']).indices.items()}
    D = K.load_day(day)
    acc = {(a, sb, q, s): FP.new_acc() for a in ARMS for sb in SUB for q, s in EVAL}
    for a_ in acc.values():
        a_.update(q_lowP=0.0, q_midP=0.0, q_highP=0.0, q_noP=0.0)
    st = D['stat']; st.update(window_sides=0, c1_differ=0, align_rows=0, align_bad=0)
    T0 = time.time(); nb = len(K.G32)
    for w in K.window_sides(D):
        st['window_sides'] += 1; n = w['n']
        Xw = np.column_stack([w['X'], np.full(n, float(w['side']))]).astype(np.float32)
        P = mdl.predict(Xw) if n else np.zeros(0)
        vk = w['X'][:, nb] if n else np.zeros(0)
        P = np.where(~np.isfinite(vk) | (vk == -1.0) | (np.asarray(w['best']) < 0) | (w['jk'] < 0), np.nan, P)
        if (w['m'], w['side']) in ROWS:
            tz_, pz_ = ROWS[(w['m'], w['side'])]
            ii = np.searchsorted(w['tg'], tz_); ok = (ii < n) & (w['tg'][np.clip(ii, 0, n - 1)] == tz_)
            st['align_rows'] += len(tz_); st['align_bad'] += int((~ok).sum()) + int((~(np.abs(P[ii[ok]] - pz_[ok]) <= 1e-9)).sum())
        vv = np.asarray(w['vkl'], float); vg = (vv <= K.GUARD_PULL).tolist()
        strong = [bool(g_ and p_ == p_ and p_ >= l90) for g_, p_ in zip(vg, P)]

        def near_dec(k, t, cur):
            f = K.own_feats(k, t, cur)
            return bool(vg[k] and f[1] == f[1] and f[1] <= 1)

        def rec(arm, d, F, O, tt, p, pos):
            for sb in ('all', 'r0' if d == 0 else 'r1-5'):
                for q_, s_ in [(q, shf)]:
                    a_ = acc[(arm, sb, q_, s_)]; FP.add(a_, w, F, O, tt, p, pos)
                    for f in F:
                        t = float(w['anc'][f[0]]); k = int(np.searchsorted(w['tg'], t - 24.0, side='right')) - 1
                        pv = P[k] if k >= 0 else np.nan
                        c = 'q_noP' if not (pv == pv) else ('q_lowP' if pv <= TERC[0] else ('q_midP' if pv <= TERC[1] else 'q_highP'))
                        a_[c] += float(f[2])
        for q, shf in EVAL:
            tt = w['anc'] - shf
            F, O, _ = K.sim_arm(w, q, shf, 'J-v3'); rec('J-v3', 0, F, O, tt, 0, list(K.LAST_POS))
            for d in FS.RUNGS:
                F, O, p = K.sim_arm(w, q, shf, 'rest', aux=vg, offset=d, size=FS.LAD_SIZE); rec('LDG', d, F, O, tt, p, list(K.LAST_POS))
                F, O, p = K.sim_arm(w, q, shf, 'rest', aux=strong, offset=d, size=FS.LAD_SIZE); rec('KPS', d, F, O, tt, p, list(K.LAST_POS))
                F, O, p = K.sim_arm(w, q, shf, 'rest', aux=vg, decide=near_dec, idle_only=1, offset=d, size=FS.LAD_SIZE)
                rec('KPN', d, F, O, tt, p, list(K.LAST_POS))
                c0 = K.REST_COOL
                try:
                    K.REST_COOL = 0.0
                    F, O, p = K.sim_arm(w, q, shf, 'rest', aux=vg, offset=d, size=FS.LAD_SIZE)
                finally:
                    K.REST_COOL = c0
                if K.REST_COOL != 250.0:
                    sys.exit('KEEP-PLACE: the clearance was not restored after KPF')
                rec('KPF', d, F, O, tt, p, list(K.LAST_POS))
                F0, O0, _ = K.sim_arm(w, q, shf, 'rest', offset=d, size=FS.LAD_SIZE)
                Fk, Ok, _ = K.sim_arm(w, q, shf, 'rest', decide=K.decide_record, offset=d, size=FS.LAD_SIZE)
                st['c1_differ'] += int(Fk != F0 or Ok != O0)
        if st['window_sides'] % 40 == 0:
            r_ = K.rss_mib(); print(f'{day}: {st["window_sides"]} window-sides, {time.time() - T0:.0f} s, rss {r_:.0f} MiB', flush=True)
            if r_ > MEM_CAP_MIB:
                sys.exit(f'REFUSED: resident memory {r_:.0f} MiB > {MEM_CAP_MIB:.0f} MiB')
    if st['c1_differ'] or st['align_bad'] or st['align_rows'] != len(zm):
        sys.exit(f'KEEP-PLACE CONTROL FAILED -> no result: C1 {st["c1_differ"]}, alignment {st["align_bad"]} of {st["align_rows"]} ({len(zm)})')
    out = dict(day=day, days=len(D['W0']) / 288.0, stat=st, arms={f'{a}|{sb}|{q}|{s:.0f}': v for (a, sb, q, s), v in acc.items()})
    os.makedirs(OUTP, exist_ok=True)
    json.dump(out, open(f'{OUTP}/eval_{day}.json', 'w'), default=float)
    print(f'{day}: evaluated in {time.time() - T0:.0f} s; {st}', flush=True)


def report():
    from scipy.stats import t as tdist
    import pf_fv3_check as FV
    E = {d: json.load(open(f'{OUTP}/eval_{d}.json')) for d in DAYS9}
    old = [nm for nm, lo, _ in FP.ABINS if lo >= 100.0]; bad = {}
    for d in DAYS9:
        for key, a in E[d]['arms'].items():
            if abs(sum(a[f'q|{nm}'] for nm, _, _ in FP.ABINS) - a['q']) > 1e-6 or abs(sum(a[f'v1|{nm}'] for nm, _, _ in FP.ABINS) - a['v1']) > 1e-6:
                bad[f'bins {d} {key}'] = 1
    if bad:
        sys.exit(f'KEEP-PLACE REPORT CONTROL FAILED -> no table: {bad}')
    usd = lambda x, dd: x / 100.0 / dd

    def cell(a, dd):
        r = dict(all1=usd(a['v1'], dd), wo1=usd(sum(a[f'v1|{nm}'] for nm in old), dd), wo5=usd(sum(a[f'v5|{nm}'] for nm in old), dd),
                 st=usd(a['vst'], dd), q=a['q'] / dd, pulls=a['pulls'] / dd)
        for nm, _, _ in FP.ABINS:
            r[f'b|{nm}'] = usd(a[f'v1|{nm}'], dd)
        for k_, wt in FV.B1A.items():
            r[f'w_{k_}'] = sum(r[f'b|{b[0]}'] * p for b, p in zip(FV.BINS, wt))
        qo = sum(a[f'q|{nm}'] for nm in old)
        r['cps_wo1'] = sum(a[f'v1|{nm}'] for nm in old) / qo if qo else float('nan')
        for c in ('q_thr', 'q_pend', 'q_front5'):
            r[c] = a[c] / a['q'] if a['q'] else float('nan')
        qp = a['q_lowP'] + a['q_midP'] + a['q_highP']
        r['highP'] = a['q_highP'] / qp if qp else float('nan'); r['lowP'] = a['q_lowP'] / qp if qp else float('nan')
        return r
    per = {d: {k_: cell(a, E[d]['days']) for k_, a in E[d]['arms'].items()} for d in DAYS9}
    alld = sum(E[d]['days'] for d in DAYS9)
    pooled = {k_: cell({c: sum(E[d]['arms'][k_][c] for d in DAYS9) for c in E[DAYS9[0]]['arms'][k_]}, alld) for k_ in E[DAYS9[0]]['arms']}
    dv = lambda arm, c, col, sb='all': np.array([per[d][f'{arm}|{sb}|{c}'][col] for d in DAYS9])
    R = {}
    for arm in ('KPS', 'KPN', 'KPF'):
        up = [int(np.sum(dv(arm, c, 'wo1') > dv('LDG', c, 'wo1'))) for c in ('M|0', 'M|50')]
        dn = [int(np.sum(dv(arm, c, 'wo1') < dv('LDG', c, 'wo1'))) for c in ('M|0', 'M|50')]
        R[f'R440 {arm}'] = 'HELPS' if min(up) >= 7 else ('HURTS' if min(dn) >= 7 else 'NOT-SHOWN'); R[f'R440 {arm} days_up_down'] = [up, dn]
        R[f'R441 {arm}'] = f"{sum(pooled[f'{arm}|all|{q}|{s:.0f}']['wo1'] > 0 for q, s in K.EVAL_CFG)} of 3"
        k6 = sum(pooled[f'{arm}|all|{q}|{s:.0f}'][u] > pooled[f'J-v3|all|{q}|{s:.0f}'][u] for q, s in K.EVAL_CFG for u in ('wo1', 'wo5'))
        R[f'R442 {arm}'] = 'BEATS-V3' if k6 == 6 else f'{k6} of 6'
    R['R443'] = {arm: dict(pulls=pooled[f'{arm}|all|M|50']['pulls'], deep_swept=pooled[f'{arm}|r1-5|M|50']['q_thr'],
                           deep_highP=pooled[f'{arm}|r1-5|M|50']['highP'], front=pooled[f'{arm}|all|M|50']['q_front5']) for arm in ('KPS', 'KPN', 'KPF', 'LDG')}
    IV = {}
    for a1, a0 in (('KPS', 'LDG'), ('KPN', 'LDG'), ('KPF', 'LDG'), ('KPS', 'J-v3'), ('KPN', 'J-v3'), ('KPF', 'J-v3')):
        for q, s in EVAL:
            c = f'{q}|{s:.0f}'; x = dv(a1, c, 'wo1') - dv(a0, c, 'wo1'); h = float(tdist.ppf(0.975, 8) * x.std(ddof=1) / 3.0)
            IV[f'{a1} - {a0} {c}'] = dict(mean=float(x.mean()), lo=float(x.mean() - h), hi=float(x.mean() + h), days_better=int((x > 0).sum()))
    asof = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    json.dump(dict(as_of=asof, readings=R, pooled=pooled, per_day=per, intervals=IV), open(f'{OUTP}/keep_place.json', 'w'), indent=1, default=float)
    T = [f'KEEP-PLACE, 9 consumed replay days, first 60 s, 30-sh rungs x 6 per side; as of {asof}; controls passed (self-tests, C1 every rung, P aligned, bins)',
         'Readings: ' + json.dumps(R, default=float), '',
         'pooled $/day at 1 s: all | < 100 | 100-500 | >= 500 ms | w/o young | B1a M0 / M50 | w/o young 5 s | settle | c/sh w/o young | sh/day | pulls/day | '
         'swept | pending | front <= 5 sh | fills at low / high P']
    for q, s in EVAL:
        for a in ARMS:
            for sb in (('all',) if a == 'J-v3' else SUB):
                y = pooled[f'{a}|{sb}|{q}|{s:.0f}']
                T.append(f"  {q}{s:<3.0f} {a:<12s} {sb:<5s} {y['all1']:+9.2f} | {y['b|< 100 ms']:+8.2f} | {y['b|100-500 ms']:+8.2f} | {y['b|>= 500 ms']:+8.2f} | "
                         f"{y['wo1']:+9.2f} | {y['w_M0']:+8.2f} / {y['w_M50']:+8.2f} | {y['wo5']:+9.2f} | {y['st']:+9.2f} | {y['cps_wo1']:+.2f} | {y['q']:7.0f} | "
                         f"{y['pulls']:6.0f} | {y['q_thr']:.0%} | {y['q_pend']:.0%} | {y['q_front5']:.0%} | {y['lowP']:.0%} / {y['highP']:.0%}")
    T += ['', 'per day, $/day w/o young at 1 s (M 50): KPS | KPN | KPF | LDG | J-v3']
    for d in DAYS9:
        T.append(f"  {d} " + ' | '.join(f"{per[d][f'{a}|all|M|50']['wo1']:+9.2f}" for a in ('KPS', 'KPN', 'KPF', 'LDG', 'J-v3')))
    T += ['', 'day-clustered differences, $/day w/o young at 1 s (mean [95% t-interval], days better):']
    for k_, y in IV.items():
        T.append(f"  {k_}: {y['mean']:+.2f} [{y['lo']:+.2f}, {y['hi']:+.2f}], {y['days_better']} / 9")
    open(f'{OUTP}/keep_place.txt', 'w').write('\n'.join(T) + '\n'); print('\n'.join(T))


if __name__ == '__main__':
    cmd = sys.argv[1] if len(sys.argv) > 1 else ''
    if cmd == 'eval' and len(sys.argv) > 2:
        fp_eval(sys.argv[2])
    elif cmd == 'report':
        report()
    else:
        sys.exit(__doc__)
