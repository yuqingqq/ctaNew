"""FRONT-PULL (2026-11-13, session ce68ebe0; USER: "How we can do front pull" / "You mean we use imbalance, level change to do the
action? We can try"): rest at the best by default, and pull on the predicted chance that the level goes, with a line set by the
order's place in the queue.

Simulator: S4's (pf_kp_s4.py: B21's rest lane on the 9 replay days, clocks, fill rules, trade times and latencies of B21 / S4; one 5-sh
order per side at the known best, held within 3 ticks, re-placed once a trigger has been clear for 250 ms). pf_kp_s4.py gains one
inert switch for this test: KP_TRIG_IDLE_ONLY (a trigger acts only while no order of ours rests: the entry line), passed by sim_arm;
off, every S4 / B21 path is unchanged (fp_selftest runs S4's and B21's self-tests and falsifies the switch).
Days: 10-23..27, 10-31, 11-01..03 (consumed; exploratory). First 60 s of each window.

Signal P = MOVE-AVOID's fit (pf_move_avoid.py ops / feeds, 3668f17c / 50227d6e): LightGBM with the pinned parameters, binary, on MOVE's
decision rows (every 500 ms) still present at t + 24; target AV = the side's best moves worse than the known best in (t + 24, t + 274]
ms (a level a cancel could still leave); inputs = the simulator's own per-step inputs (the guard's 32 features, v, our / the other
side's L1 size change over 250 ms, Binance activity, Binance imbalance change over 250 ms) and the side. One model per held-out day,
fitted on the other 8: the held-out day never enters its own model or lines. P is computed at every 50 ms step of the held-out day.
Undefined P (no known best, no known frame, or no guard value): keep and no entry block (counted).
Lines, per held-out day, from the training days only and without labels: the 50 / 75 / 90% quantiles of P over their decision rows.

Arms per held-out day, at fill clocks M 0 / 50 / 250 and U 50 (nothing ahead of us: the front bound, descriptive):
  Z-none      never pulled (S4)
  Z-guard     pull when v <= -0.10 (B21's best signal; the control)
  J-v3        the v3 quote line (lane J; the reference)
  FP10/25/50  pull when P >= the 90 / 75 / 50% line (the top 10 / 25 / 50% of moments); the same line blocks re-entry
  FPQ         the queue-priced line: entry, and orders with more than 25 sh known ahead, use the 50% line; orders with <= 25 sh known
              ahead use the 90% line (known ahead = min(the size shown at our price at activation, the smallest shown since), known
              frames only). A front order stays through more risk; a back order, reached mostly by trades that take the level, leaves
              early
  FPQ-shuf    FPQ with P shuffled in time within each window-side (falsifier)
  random      200 draws at M 0 / M 50: on each window-side as many pulls as FPQ made there, at resting steps of the never-pull path
              (S4's null)
Values per fill (S4's): sg (mid(anchor + h) - price) x shares at h = 1 / 5 / 60 s and settlement; split by our order's age at the fill
(< 100 / 100-500 / >= 500 ms); 'w/o young' = fills on orders >= 100 ms old; B1a-weighted with FV3's M 0 / M 50 weights (FINDINGS C1).
Also: shares in trades through the level (swept), shares filled while our cancel was pending (pulled too late: the part only speed
fixes), shares with <= 5 sh ahead just before the fill (front), pulls and orders per side-minute.

Readings DECLARED BEFORE THE FIRST RUN (9 consumed days; day rule >= 7 of 9; $/day w/o young at 1 s unless stated):
 R380 PULL: FPQ above Z-guard AND above the random p95 on >= 7 of 9 days, at M 0 and at M 50 -> PULL-WORKS; else PULL-NOT-SHOWN.
 R381 VS-V3: FPQ against J-v3 at 1 s and 5 s x M 0 / 50 / 250 (w/o young) -> BEATS-V3 if 6 of 6, else 'k of 6'.
 R382 FRONT-LINE: FPQ against FP50 (the same entry line; only front orders differ) at M 0 and M 50: above on >= 7 days at both ->
      FRONT-LINE-HELPS; below on >= 7 at both -> FRONT-LINE-HURTS; else NOT-SHOWN.
 R383 POSITIVE: FPQ's pooled c/sh at 1 s w/o young > 0 at M 0 / 50 / 250 -> 'k of 3'.
 R384 FALSIFIER: FPQ-shuf must read PULL-NOT-SHOWN by R380's rule; otherwise the evaluation is VOID.
 Printed, not readings: every arm by clock ($/day all / by age / w/o young / B1a M0 / M50, 5 s w/o young, settlement, c/sh, shares,
 pulls and orders per side-minute, swept / pending / front shares); day-clustered FPQ minus Z-guard / J-v3 / FP50 (t, 8 df).
Controls (any failure -> no output): S4's and B21's self-tests and mutation check; fp_selftest (the idle-only switch: a trigger at a
resting step leaves the path unchanged, at an idle step acts as in the plain lane, a callback still pulls; falsifier: without the
switch the resting-step trigger changes the path; the tier choice, with a planted swap that must fail); fit: each held-out day's AV
AUC reproduces MOVE-AVOID feeds' BASE (control) per-day AUC within 1e-6; eval: P from the simulator's inputs equals the fit's
prediction on every MOVE row of the day (|diff| <= 1e-9, every row matched); C1 (the never-pull callback path equals Z-none at every
clock); age bins add up; memory self-guard (2 GiB).
Limits: replay fill rule M not validated (B1a); one 5-sh order per side; our order not in the public book; v3's CAP10 / RO60 / MICRO /
G4 / Binance hard guards not in the rest lane; consumed days, and the signal's form was chosen after MOVE / AVOID on these days
(exploratory: a pass makes a candidate for a forward test only); MOVE's rows, not our resting steps, are the training population; the
AV fit's v comes from guard models whose fits included the held-out day (S4's indirect leak).
Usage (each through run_light, PF_SLOT=claude): pf_front_pull.py fpselftest | fpfit | fpeval <day> | fpreport; driver run_front_pull.sh
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
import pf_model_spec as MS                                # noqa: E402

INF = float('inf')
DAYS9 = K.DAYS9
OUTF = f'{K.IO}/front_pull_20261113'
MOVE_DIR = f'{K.IO}/move_20261112'
FEEDS_JSON = f'{MOVE_DIR}/avoid_feeds.json'
BASE_IN = K.G32 + ['v', 'own_d250', 'opp_d250', 'act_rel', 'd_bnimb250', 'side']   # MOVE's BASE = the simulator's inputs + side
FRONT_SH = 25.0                                           # FPQ: at most this many shares known ahead -> a front order
QL = (0.50, 0.75, 0.90)
EVAL_FP = [('M', 0.0), ('M', 50.0), ('M', 250.0), ('U', 50.0)]
NULL_FP = [('M', 0.0), ('M', 50.0)]
FP_ARMS = ['Z-none', 'Z-guard', 'J-v3', 'FP10', 'FP25', 'FP50', 'FPQ', 'FPQ-shuf']
ABINS = (('< 100 ms', -INF, 100.0), ('100-500 ms', 100.0, 500.0), ('>= 500 ms', 500.0, INF))
MUT_TIER_SWAP = 0                                         # planted error (self-test only): front and back lines swapped
MEM_CAP_MIB = 2048.0


def fp_line(ahead, l_back, l_front):
    """FPQ's line for a resting order with `ahead` shares known ahead (nan -> back)."""
    front = ahead == ahead and ahead <= FRONT_SH
    if MUT_TIER_SWAP:
        front = not front
    return l_front if front else l_back


def fp_selftest():
    """The idle-only switch on S4's synthetic book, and the tier choice."""
    bad = []
    B50 = (-100, [(50, 100)], 52); fr = [B50, (310, [(50, 50)], 52)]; tr = [(200, 49, 120), (500, 50, 5), (600, 50, 25), (800, 50, 60)]
    tf = np.array([f[0] for f in fr], float); PB = np.full((2, 5), -1); SB = np.zeros((2, 5))
    for i, f in enumerate(fr):
        for j, (p, z) in enumerate(f[1][:5]):
            PB[i, j] = p; SB[i, j] = z
    ba = np.array([f[2] for f in fr]); tau = np.arange(0.0, 1000.0, 50.0); n = len(tau); vv = [1.0] * n; st = [True] * n
    best = [50] * n; bo = [52] * n; tt = np.array([x[0] for x in tr], float); tp = [x[1] for x in tr]; tq = [float(x[2]) for x in tr]
    run = lambda aux: K.sim2('rest', 0, tf, ba, PB, SB, {}, tau, vv, st, -INF, best, bo, tt, tp, tq, 'M', 1000.0, aux=aux)
    try:
        plain = run([False] * n)
        at2 = [False] * n; at2[2] = True                  # step 2 (100 ms): our order rests (placed at 0, live at 40 ms)
        at0 = [False] * n; at0[0] = True                  # step 0: no order yet
        ref2 = run(at2); ref0 = run(at0)
        if ref2 == plain:
            bad.append('falsifier: without the switch a trigger at a resting step did not change the path')
        if ref0 == plain:
            bad.append('falsifier: a trigger at an idle step did not delay placement')
        K.KP_TRIG_IDLE_ONLY = 1
        if run(at2) != plain:
            bad.append('idle-only: a trigger at a resting step changed the path')
        if run(at0) != ref0:
            bad.append('idle-only: a trigger at an idle step acts differently from the plain lane')
        K.KP_DECIDE = lambda k, t, cur: k == 2
        if run([False] * n) != ref2:
            bad.append('idle-only: a callback pull at step 2 differs from the lane triggered at step 2')
    finally:
        K.KP_DECIDE = None; K.KP_TRIG_IDLE_ONLY = 0
    global MUT_TIER_SWAP
    cases = [(0.0, 'F'), (25.0, 'F'), (25.5, 'B'), (300.0, 'B'), (float('nan'), 'B')]
    got = lambda: ['F' if fp_line(a, 1.0, 2.0) == 2.0 else 'B' for a, _ in cases]
    if got() != [c for _, c in cases]:
        bad.append(f'tier choice {got()}')
    MUT_TIER_SWAP = 1
    try:
        if got() == [c for _, c in cases]:
            bad.append('falsifier: the planted tier swap passed the tier cases')
    finally:
        MUT_TIER_SWAP = 0
    return bad


def all_selftests():
    bad = K.selftest(); mut = K.mutation_check()
    more = K.hold_selftest() or K.rest_selftest() or K.kp_selftest() or K.s4_selftest() or K.selftest()
    fp = fp_selftest()
    if bad or not all(mut.values()) or more or fp:
        sys.exit(f'SELFTEST FAILED -> no output: {bad} {more} {fp} {mut}')


def fp_rows():
    import pf_move_info as MV
    if BASE_IN != MV.BASE_N:
        sys.exit('REFUSED: the FRONT-PULL inputs differ from MOVE BASE')
    B = {}; gone = []
    for d in DAYS9:
        z = np.load(f'{MOVE_DIR}/day_{d}.npz')
        for k in BASE_IN + ['day', 'DN_250', 'm', 't']:
            B.setdefault(k, []).append(z[f'B|{k}'])
        gone.append(np.load(f'{MOVE_DIR}/avoid_{d}.npy'))
    return {k: np.concatenate(v) for k, v in B.items()}, np.concatenate(gone).astype(bool)


def fp_fit():
    """One AV model per held-out day (the other 8 days, rows still present at t + 24), its P on the day's MOVE rows (for the
    alignment control) and its lines (quantiles of P over the training days' rows)."""
    import lightgbm as lgb
    import pf_move_info as MV
    os.makedirs(f'{OUTF}/models', exist_ok=True)
    B, gone = fp_rows(); day = B['day'].astype(int); still = ~gone
    av = ((B['DN_250'] == 1) & still).astype(int)
    Xm = np.column_stack([B[k] for k in BASE_IN]).astype(np.float32)
    Xs, ys, ds = Xm[still], av[still], day[still]
    ref = json.load(open(FEEDS_JSON))['BASE (control)']['days']
    pr = dict(MS.PARAMS); pr['objective'] = 'binary'
    fit = dict(as_of=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), rows=int(len(day)), rows_still=int(still.sum()), days={})
    bad = {}
    for i, d in enumerate(DAYS9):
        te = ds == int(d)
        mdl = lgb.train(pr, lgb.Dataset(Xs[~te], ys[~te], free_raw_data=True), MS.ROUNDS)
        mdl.save_model(f'{OUTF}/models/av_{d}.txt')
        a = MV.auc(ys[te], mdl.predict(Xs[te]))
        if abs(a - ref[i]) > 1e-6:
            bad[d] = (a, ref[i])
        np.save(f'{OUTF}/p_rows_{d}.npy', mdl.predict(Xm[day == int(d)]))
        ptr = mdl.predict(Xm[day != int(d)])
        fit['days'][d] = dict(auc_av=a, auc_ref=ref[i], lines={f'{q:.2f}': float(np.quantile(ptr, q)) for q in QL})
        print(f'fit {d}: AV AUC {a:.4f} (ref {ref[i]:.4f}); lines {fit["days"][d]["lines"]}; rss {K.rss_mib():.0f} MiB', flush=True)
        if K.rss_mib() > MEM_CAP_MIB:
            sys.exit(f'REFUSED: resident memory {K.rss_mib():.0f} MiB > {MEM_CAP_MIB:.0f} MiB')
    if bad:
        sys.exit(f'FRONT-PULL FIT CONTROL FAILED -> no models used: {bad}')
    json.dump(fit, open(f'{OUTF}/fit.json', 'w'), indent=1)


def new_acc():
    a = dict(q=0.0, v1=0.0, v5=0.0, v60=0.0, vst=0.0, fills=0, pulls=0, orders=0, q_thr=0.0, q_pend=0.0, v1_pend=0.0, q_front5=0.0)
    for nm, _, _ in ABINS:
        a[f'q|{nm}'] = 0.0; a[f'v1|{nm}'] = 0.0; a[f'v5|{nm}'] = 0.0
    return a


def add(a, w, F, O, tt, pulls, pos):
    if len(pos) != len(F):
        sys.exit(f'FRONT-PULL: queue positions ({len(pos)}) not aligned with fills ({len(F)})')
    r = K.vals(w, F, tt); q = r['q']
    age = np.array([f[4] for f in F], float); thr = np.array([bool(f[3]) for f in F], bool)
    pend = np.array([f[5] == 'pend' for f in F], bool); ahb = np.array([p_[0] for p_ in pos], float)
    a['q'] += float(q.sum()); a['fills'] += int(len(q)); a['pulls'] += int(pulls); a['orders'] += int(sum(not o[3] for o in O))
    for c, f in (('v1', 'm1'), ('v5', 'm5'), ('v60', 'm60'), ('vst', 'st')):
        a[c] += float(np.nansum(q * r[f]))
    for nm, lo, hi in ABINS:
        m_ = (age >= lo) & (age < hi)
        a[f'q|{nm}'] += float(q[m_].sum()); a[f'v1|{nm}'] += float(np.nansum(q[m_] * r['m1'][m_])); a[f'v5|{nm}'] += float(np.nansum(q[m_] * r['m5'][m_]))
    if len(q):
        a['q_thr'] += float(q[thr].sum()); a['q_pend'] += float(q[pend].sum()); a['v1_pend'] += float(np.nansum(q[pend] * r['m1'][pend]))
        a['q_front5'] += float(q[ahb <= 5.0].sum())


def run_fpq(w, q, shf, P, l_back, l_front):
    """FPQ: the entry line (idle-only trigger) is the back line; a resting order is pulled when P reaches its tier's line."""
    def dec(k, t, cur):
        f = K.own_feats(k, t, cur)                        # also updates the known minimum shown since activation
        a0, mn = f[2], cur['_kpmin']
        ah = min(a0, mn) if (a0 == a0 and mn == mn) else a0
        p = P[k]
        return bool(p == p and p >= fp_line(ah, l_back, l_front))
    return K.sim_arm(w, q, shf, 'rest', aux=(P >= l_back).tolist(), decide=dec, idle_only=1)


def fp_eval(day):
    import lightgbm as lgb
    all_selftests()
    fit = json.load(open(f'{OUTF}/fit.json'))['days'][day]; L = fit['lines']
    l50, l75, l90 = L['0.50'], L['0.75'], L['0.90']
    mdl = lgb.Booster(model_file=f'{OUTF}/models/av_{day}.txt')
    z = np.load(f'{MOVE_DIR}/day_{day}.npz'); zm = z['B|m'].astype(np.int64); zt = z['B|t']; zs = z['B|side'].astype(int)
    pz = np.load(f'{OUTF}/p_rows_{day}.npy')
    if len(pz) != len(zm):
        sys.exit(f'FRONT-PULL: p_rows ({len(pz)}) not aligned with MOVE rows ({len(zm)})')
    ROWS = {(int(m_), int(s_)): (zt[ix], pz[ix]) for (m_, s_), ix in pd.DataFrame(dict(m=zm, s=zs)).groupby(['m', 's']).indices.items()}
    D = K.load_day(day)
    acc = {(a, q, s): new_acc() for a in FP_ARMS for q, s in EVAL_FP}
    nul = {(q, s): np.zeros((K.NULL_DRAWS, 5)) for q, s in NULL_FP}   # per draw: shares, value 1 s, value 5 s, value 1 s w/o young, pulls
    st = D['stat']; st.update(window_sides=0, c1_differ=0, align_rows=0, align_bad=0, steps=0, p_undefined_steps=0)
    rng = np.random.default_rng(int(day) + 13)
    T0 = time.time(); nb = len(K.G32)
    for w in K.window_sides(D):
        st['window_sides'] += 1; n = w['n']
        Xw = np.column_stack([w['X'], np.full(n, float(w['side']))]).astype(np.float32)
        P = mdl.predict(Xw) if n else np.zeros(0)
        vk = w['X'][:, nb] if n else np.zeros(0)
        und = ~np.isfinite(vk) | (vk == -1.0) | (np.asarray(w['best']) < 0) | (w['jk'] < 0)
        P = np.where(und, np.nan, P); st['steps'] += n; st['p_undefined_steps'] += int(und.sum())
        if (w['m'], w['side']) in ROWS:
            tz_, pz_ = ROWS[(w['m'], w['side'])]
            if not n:
                sys.exit(f'FRONT-PULL CONTROL FAILED: MOVE rows on a window-side without grid steps ({w["m"]}, {w["side"]})')
            ii = np.searchsorted(w['tg'], tz_); ok = (ii < n) & (w['tg'][np.clip(ii, 0, n - 1)] == tz_)
            st['align_rows'] += len(tz_)
            st['align_bad'] += int((~ok).sum()) + int((~(np.abs(P[ii[ok]] - pz_[ok]) <= 1e-9)).sum())
        Ps = rng.permutation(P)
        for q, shf in EVAL_FP:
            tt = w['anc'] - shf
            F0, O0, _ = K.sim_arm(w, q, shf, 'Z-none'); add(acc[('Z-none', q, shf)], w, F0, O0, tt, 0, K.LAST_POS)
            Fk, Ok, _ = K.sim_arm(w, q, shf, 'rest', decide=K.decide_record); resting = list(K.CTX['resting'])
            st['c1_differ'] += int(Fk != F0 or Ok != O0)
            F, O, p = K.sim_arm(w, q, shf, 'rest', aux=(np.asarray(w['vkl']) <= K.GUARD_PULL).tolist()); add(acc[('Z-guard', q, shf)], w, F, O, tt, p, K.LAST_POS)
            F, O, _ = K.sim_arm(w, q, shf, 'J-v3'); add(acc[('J-v3', q, shf)], w, F, O, tt, 0, K.LAST_POS)
            for nm, ln in (('FP10', l90), ('FP25', l75), ('FP50', l50)):
                F, O, p = K.sim_arm(w, q, shf, 'rest', aux=(P >= ln).tolist()); add(acc[(nm, q, shf)], w, F, O, tt, p, K.LAST_POS)
            F, O, pk = run_fpq(w, q, shf, P, l50, l90); add(acc[('FPQ', q, shf)], w, F, O, tt, pk, K.LAST_POS)
            F, O, p = run_fpq(w, q, shf, Ps, l50, l90); add(acc[('FPQ-shuf', q, shf)], w, F, O, tt, p, K.LAST_POS)
            if (q, shf) in NULL_FP and resting:
                for dr in range(K.NULL_DRAWS):
                    aux = [False] * n
                    for kx in rng.choice(resting, size=min(pk, len(resting)), replace=False):
                        aux[int(kx)] = True
                    F, O, p = K.sim_arm(w, q, shf, 'rest', aux=aux, resting_only=1); r = K.vals(w, F, tt)
                    old = np.array([f[4] >= 100.0 for f in F], bool)
                    nul[(q, shf)][dr] += (float(r['q'].sum()), float(np.nansum(r['q'] * r['m1'])), float(np.nansum(r['q'] * r['m5'])),
                                          float(np.nansum(r['q'][old] * r['m1'][old])) if len(old) else 0.0, p)
        if st['window_sides'] % 40 == 0:
            r_ = K.rss_mib(); print(f'{day}: {st["window_sides"]} window-sides, {time.time() - T0:.0f} s, rss {r_:.0f} MiB', flush=True)
            if r_ > MEM_CAP_MIB:
                sys.exit(f'REFUSED: resident memory {r_:.0f} MiB > {MEM_CAP_MIB:.0f} MiB')
    if st['c1_differ'] or st['align_bad'] or st['align_rows'] != len(zm):
        sys.exit(f'FRONT-PULL CONTROL FAILED -> no result: C1 {st["c1_differ"]}, alignment {st["align_bad"]} bad of {st["align_rows"]} '
                 f'(MOVE rows {len(zm)})')
    out = dict(day=day, days=len(D['W0']) / 288.0, lines=L, stat=st, arms={f'{a}|{q}|{s:.0f}': v for (a, q, s), v in acc.items()},
               null={f'{q}|{s:.0f}': nul[(q, s)].tolist() for q, s in NULL_FP})
    os.makedirs(OUTF, exist_ok=True)
    json.dump(out, open(f'{OUTF}/eval_{day}.json', 'w'), default=float)
    print(f'{day}: evaluated in {time.time() - T0:.0f} s; {st}', flush=True)


def fp_report():
    from scipy.stats import t as tdist
    import pf_fv3_check as FV
    E = {d: json.load(open(f'{OUTF}/eval_{d}.json')) for d in DAYS9}
    bad = {}
    for d in DAYS9:
        for key, a in E[d]['arms'].items():
            if abs(sum(a[f'q|{nm}'] for nm, _, _ in ABINS) - a['q']) > 1e-6 or abs(sum(a[f'v1|{nm}'] for nm, _, _ in ABINS) - a['v1']) > 1e-6:
                bad[f'bins {d} {key}'] = 1
    if bad:
        sys.exit(f'FRONT-PULL REPORT CONTROL FAILED -> no table: {bad}')
    usd = lambda x, dd: x / 100.0 / dd
    old = [nm for nm, lo, _ in ABINS if lo >= 100.0]

    def cell(a, dd):
        r = dict(all1=usd(a['v1'], dd), all5=usd(a['v5'], dd), wo1=usd(sum(a[f'v1|{nm}'] for nm in old), dd),
                 wo5=usd(sum(a[f'v5|{nm}'] for nm in old), dd), st=usd(a['vst'], dd), q=a['q'] / dd, pulls=a['pulls'] / dd, orders=a['orders'] / dd)
        for nm, _, _ in ABINS:
            r[f'b|{nm}'] = usd(a[f'v1|{nm}'], dd)
        for k_, wt in FV.B1A.items():
            r[f'w_{k_}'] = sum(r[f'b|{b[0]}'] * p for b, p in zip(FV.BINS, wt))
        qo = sum(a[f'q|{nm}'] for nm in old)
        r['cps_wo1'] = sum(a[f'v1|{nm}'] for nm in old) / qo if qo else float('nan')
        r['cps1'] = a['v1'] / a['q'] if a['q'] else float('nan')
        for c in ('q_thr', 'q_pend', 'q_front5'):
            r[c] = a[c] / a['q'] if a['q'] else float('nan')
        r['v1_pend'] = usd(a['v1_pend'], dd)
        return r
    per = {d: {k_: cell(a, E[d]['days']) for k_, a in E[d]['arms'].items()} for d in DAYS9}
    for d in DAYS9:
        dd = E[d]['days']
        for key, M_ in E[d]['null'].items():
            M_ = np.asarray(M_); wo = M_[:, 3] / 100.0 / dd
            per[d][f'RND|{key}'] = dict(wo1_mean=float(wo.mean()), wo1_p95=float(np.quantile(wo, 0.95)), pulls=float(M_[:, 4].mean() / dd))
    alld = sum(E[d]['days'] for d in DAYS9)
    pooled = {}
    for key in E[DAYS9[0]]['arms']:
        tot = {c: sum(E[d]['arms'][key][c] for d in DAYS9) for c in E[DAYS9[0]]['arms'][key]}
        pooled[key] = cell(tot, alld)
    dv = lambda arm, c, col: np.array([per[d][f'{arm}|{c}'][col] for d in DAYS9])
    R = {}

    def pull_rule(arm):
        cond = {}
        for q, s in NULL_FP:
            c = f'{q}|{s:.0f}'
            cond[f'{c} > Z-guard'] = int(np.sum(dv(arm, c, 'wo1') > dv('Z-guard', c, 'wo1')))
            cond[f'{c} > random p95'] = int(np.sum(dv(arm, c, 'wo1') > np.array([per[d][f'RND|{c}']['wo1_p95'] for d in DAYS9])))
        return ('PULL-WORKS' if all(v >= 7 for v in cond.values()) else 'PULL-NOT-SHOWN'), cond
    R['R380'], R['R380_days'] = pull_rule('FPQ')
    k6 = sum(pooled[f'FPQ|{q}|{s:.0f}'][u] > pooled[f'J-v3|{q}|{s:.0f}'][u] for q, s in K.EVAL_CFG for u in ('wo1', 'wo5'))
    R['R381'] = 'BEATS-V3' if k6 == 6 else f'{k6} of 6'
    up = [int(np.sum(dv('FPQ', f'{q}|{s:.0f}', 'wo1') > dv('FP50', f'{q}|{s:.0f}', 'wo1'))) for q, s in NULL_FP]
    dn = [int(np.sum(dv('FPQ', f'{q}|{s:.0f}', 'wo1') < dv('FP50', f'{q}|{s:.0f}', 'wo1'))) for q, s in NULL_FP]
    R['R382'] = 'FRONT-LINE-HELPS' if min(up) >= 7 else ('FRONT-LINE-HURTS' if min(dn) >= 7 else 'NOT-SHOWN'); R['R382_days_up_down'] = [up, dn]
    k3 = sum(pooled[f'FPQ|{q}|{s:.0f}']['cps_wo1'] > 0 for q, s in K.EVAL_CFG); R['R383'] = f'{k3} of 3'
    fl, fc = pull_rule('FPQ-shuf'); R['R384'] = 'VOID' if fl == 'PULL-WORKS' else 'FALSIFIER-FAILS-AS-REQUIRED'; R['R384_days'] = fc
    IV = {}
    for other in ('Z-guard', 'J-v3', 'FP50'):
        for q, s in EVAL_FP:
            c = f'{q}|{s:.0f}'; x = dv('FPQ', c, 'wo1') - dv(other, c, 'wo1')
            h = float(tdist.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x)))
            IV[f'FPQ - {other} {c}'] = dict(mean=float(x.mean()), lo=float(x.mean() - h), hi=float(x.mean() + h), days_better=int((x > 0).sum()))
    st = {k_: sum(E[d]['stat'].get(k_, 0) for d in DAYS9) for k_ in ('window_sides', 'steps', 'p_undefined_steps', 'align_rows', 'grid_steps_without_row')}
    asof = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    res = dict(as_of=asof, days=DAYS9, readings=R, pooled=pooled, per_day=per, intervals=IV, stat=st, lines={d: E[d]['lines'] for d in DAYS9})
    json.dump(res, open(f'{OUTF}/front_pull.json', 'w'), indent=1, default=float)
    T = [f'FRONT-PULL, 9 consumed replay days, first 60 s, one 5-sh order per side; as of {asof}; controls passed (self-tests, fit '
         f'reproduces MOVE-AVOID, P aligned on {st["align_rows"]} MOVE rows, C1, bins); steps {st["steps"]}, P undefined {st["p_undefined_steps"]}',
         'Readings: ' + json.dumps(R), '',
         'pooled $/day at 1 s: all | < 100 | 100-500 | >= 500 ms | w/o young | B1a M0 / M50 | w/o young 5 s | settle | c/sh all / w/o young | '
         'sh/day | pulls/day | orders/day | swept | pending (value $/day) | front <= 5 sh']
    for q, s in EVAL_FP:
        for a in FP_ARMS:
            y = pooled[f'{a}|{q}|{s:.0f}']
            T.append(f"  {q}{s:<3.0f} {a:<9s} {y['all1']:+7.2f} | {y['b|< 100 ms']:+7.2f} | {y['b|100-500 ms']:+7.2f} | {y['b|>= 500 ms']:+7.2f} | "
                     f"{y['wo1']:+7.2f} | {y['w_M0']:+6.2f} / {y['w_M50']:+6.2f} | {y['wo5']:+7.2f} | {y['st']:+8.2f} | {y['cps1']:+.2f} / {y['cps_wo1']:+.2f} | "
                     f"{y['q']:6.0f} | {y['pulls']:6.0f} | {y['orders']:6.0f} | {y['q_thr']:.0%} | {y['q_pend']:.0%} ({y['v1_pend']:+.2f}) | {y['q_front5']:.0%}")
    T += ['', 'per day, $/day w/o young at 1 s (M 50): FPQ | FP50 | Z-guard | Z-none | J-v3 | random mean / p95 | FPQ-shuf']
    for d in DAYS9:
        p_ = per[d]; rn = p_['RND|M|50']
        T.append(f"  {d} {p_['FPQ|M|50']['wo1']:+7.2f} | {p_['FP50|M|50']['wo1']:+7.2f} | {p_['Z-guard|M|50']['wo1']:+7.2f} | {p_['Z-none|M|50']['wo1']:+8.2f} | "
                 f"{p_['J-v3|M|50']['wo1']:+6.2f} | {rn['wo1_mean']:+7.2f} / {rn['wo1_p95']:+7.2f} | {p_['FPQ-shuf|M|50']['wo1']:+7.2f}")
    T += ['', 'day-clustered FPQ minus other, $/day w/o young at 1 s (mean [95% t-interval], days better):']
    for k_, y in IV.items():
        T.append(f"  {k_}: {y['mean']:+.2f} [{y['lo']:+.2f}, {y['hi']:+.2f}], {y['days_better']} / 9")
    T += ['', 'lines (P quantiles 50 / 75 / 90 from the training days): ' + '; '.join(f"{d[4:]} {E[d]['lines']['0.50']:.3f} / {E[d]['lines']['0.75']:.3f} / "
                                                                                     f"{E[d]['lines']['0.90']:.3f}" for d in DAYS9)]
    open(f'{OUTF}/front_pull.txt', 'w').write('\n'.join(T) + '\n'); print('\n'.join(T))


if __name__ == '__main__':
    cmd = sys.argv[1] if len(sys.argv) > 1 else ''
    if cmd == 'fpselftest':
        all_selftests(); print('self-tests PASS (B21, S1, S4, FRONT-PULL)')
    elif cmd == 'fpfit':
        all_selftests(); fp_fit()
    elif cmd == 'fpeval' and len(sys.argv) > 2:
        fp_eval(sys.argv[2])
    elif cmd == 'fpreport':
        fp_report()
    else:
        sys.exit(__doc__)
