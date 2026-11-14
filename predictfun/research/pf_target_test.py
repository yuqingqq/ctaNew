"""TARGET-TEST (2026-11-14, session ce68ebe0; USER: "Yea we rebuild the model, make sure we are on the right target, proceed to test the model
and check the performance"). The keep / pull rule of FUTURE-BEST (the USER's: while the guard is on, keep a resting order unless the move is
likely to go THROUGH its price) replayed with the rebuilt models, after checking which target is the right one.

Harness: FUTURE-BEST's, unchanged (pf_future_best: the guard ladder, 30 sh, rungs d = 0..5 ticks behind the known best, S4's rest lane with
idle-only triggers and a decision callback; fb_dec: while the guard is on, pull an order d ticks behind the known best if P(move >= d + 1) >=
tau, or with a realized move if move >= d + 1; rung 0 at the best and orders ahead of the best pulled as today; a missing prediction pulls).
9 consumed replay days (10-23..27, 10-31, 11-01..03), first 60 s, fill clocks M 0 / M 50 / M 250; values as FUTURE-BEST (c/sh at 1 s / 5 s
from the trade's anchor; $/day by order age and trade timing, WS / E2).

1. THE RIGHT TARGET. The target is right if knowing it perfectly makes the most money with this rule. Oracle arms with the realized move
   (PULL-DEPTH's move: ticks against us beyond the known frame's best over the frames up to t + h) at three horizons:
     OFB1  h = 1 s (= FUTURE-BEST's OFB; STOP-DEPTH's clean 1 s target is the same move with the past-saving part kept as a status)
     OFB2  h = 2 s (INFLUENCE's target horizon)
     OFB3  h = 3 s
2. THE MODELS (out of day; each day's predictions come from models fitted without it):
     FB25    PULL-DEPTH's P (1 s, known-frame target; FUTURE-BEST's model), tau 0.25 (= FUTURE-BEST's FB25, the incumbent)
     FBN25   STOP-DEPTH's NEW P (clean 1 s target, every guard-on step), tau 0.25
     FBF25 / FBF10  INFLUENCE's FULL models (onset-trained, 2 s target) applied at EVERY guard-on step of the held-out day (features as
             INFLUENCE: STOP-DEPTH's NEW inputs + the fair-price inputs), tau 0.25 / 0.10; a step without a fair price has no prediction (pull)
     IR      the influence rule (no model; designed after INFLUENCE on these days, so post hoc in spirit): expected move X = ceil(dfv250) ticks
             when the Binance-implied fair price moved >= 0.5 c against us in the last 250 ms, else X = 1; pull an order at depth d if
             d + 1 <= X (the move goes through its price); a masked or missing fair price pulls
     LDG     every rung pulled on the guard (the reference)
Readings DECLARED BEFORE THE FIRST RUN ($/day w/o young fills (order >= 100 ms) at 1 s on WS-timed trades; day rule >= 7 of 9):
 R600 RIGHT-TARGET: OFB2 vs OFB1 higher at M 0 AND M 50 on >= 7 days -> 2S-BETTER; lower at both on >= 7 -> 1S-BETTER; else NOT-SHOWN.
      R600b the same for OFB3 vs OFB2 -> 3S-BETTER / 2S-BETTER / NOT-SHOWN. M 250 printed with whether the sign holds.
 R601 MODEL-VS-OLD: FBF25 vs FB25: higher at M 0 AND M 50 on >= 7 days -> BETTER; lower at both on >= 7 -> WORSE; else NOT-SHOWN.
 R602 MODEL-VS-LDG: FBF25 vs LDG -> HELPS / HURTS / NOT-SHOWN (the same rule).
 R603 RULE: IR vs LDG and IR vs FB25 -> HELPS / HURTS / NOT-SHOWN (the same rule).
 R604 M250 (C2): FBF25, FBN25 and IR against LDG at M 250: higher on >= 7 days -> HOLDS-AT-M250; lower on >= 7 -> FLIPS-AT-M250; else
      NOT-SHOWN.
 Printed, not readings: FBN25 vs FB25; FBF10; all-trade and E2 $/day (C12); all ages and the age split (C1); cancel-pending shares (C3);
 pulls a day; c/sh; per-day table; at the best vs behind, PRE share.
Controls (any failure -> no output): (i) FUTURE-BEST's tau = 0 control (every order pulled) equals LDG on every window-side, rung and clock;
(ii) LDG, FB25 and OFB1 equal FUTURE-BEST's saved fills (LDG, FB25, OFB) row for row at every clock; (iii) the FULL predictions recomputed at
every guard-on step equal INFLUENCE's saved onset predictions on every onset (|diff| <= 1e-9); predictions aligned by (window, side, step),
missing ones counted; (iv) synthetic cases for IR (bids and asks, no fair-price move, a 1.45 c move, missing / masked, guard off, rung 0) and
for the oracle horizons, with planted errors (IR's +1 dropped; a missing fair price kept instead of pulled; every oracle at 1 s) that must
each fail a case; FUTURE-BEST's, PULL-DEPTH's, STOP-DEPTH's and INFLUENCE's self-tests; (v) memory self-guard (2 GiB).
Limits: replay rule M (B1a; C5) and inflated volumes (C4); consumed days; the rebuilt models' inputs and IR were designed on these days
(C8: a pass is a candidate for a forward test, not a result); the FULL models are applied off their training population (onsets) at later
guard steps; the lane is the guard ladder, not the frozen pm_ladder v3 (C11); no order-level validation (C10).
Usage (run_light, PF_SLOT=claude): pf_target_test.py selftest | score <day> | eval <day> | report; driver run_target_test.sh
"""
import bisect
import json
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import pf_kp_s4 as K                                      # noqa: E402
import pf_qr_queue_race as QR                             # noqa: E402
import pf_queue_place as QP                               # noqa: E402
import pf_timing_split as TS                              # noqa: E402
import pf_pull_depth as PD                                # noqa: E402
import pf_future_best as FB                               # noqa: E402
import pf_stop_depth as SD                                # noqa: E402
import pf_influence as IN                                 # noqa: E402

DAYS9 = K.DAYS9
OUTD = f'{K.IO}/target_test_20261114'
JS = (1, 2, 3, 4, 5)
ARMS = ('LDG', 'FB25', 'FBN25', 'FBF10', 'FBF25', 'IR', 'OFB1', 'OFB2', 'OFB3')
TAUS = {'FB25': 0.25, 'FBN25': 0.25, 'FBF10': 0.10, 'FBF25': 0.25}
HOR = {'OFB1': 1000.0, 'OFB2': 2000.0, 'OFB3': 3000.0}
IR_TH = 0.5; MASK_V = -1.0
CFGS = FB.CFGS; SIZE = FB.SIZE; RUNGS = FB.RUNGS; FCOLS = FB.FCOLS; AGEB = FB.AGEB
INF = float('inf'); NAN = float('nan')
MUT_IR_SHIFT = MUT_IR_NAN_KEEP = MUT_HORIZON = 0          # planted errors (self-test only)
MEM_CAP_MIB = 2048.0


def ir_dec(vg, best, side, x250, msk, offset):
    """The influence rule (header): while the guard is on, X = ceil(dfv250) if dfv250 >= 0.5 c else 1; pull an order at depth d if d + 1 <= X;
    rung 0 at d = 0 and d < 0 pulled (as FUTURE-BEST); a masked step or a missing fair-price move pulls."""
    def dec(k, t, cur):
        if not vg[k]:
            return False
        b = best[k]
        if b < 0 or msk[k]:
            return True
        d = (b - cur['pt']) if side == 0 else (cur['pt'] - b)
        if d < 0 or (d == 0 and offset == 0):
            return True
        x = x250[k]
        if not (x == x):
            if not MUT_IR_NAN_KEEP:
                return True
            x = 0.0
        X = int(np.ceil(x)) if x >= IR_TH else 1
        return bool(d + 1 < X) if MUT_IR_SHIFT else bool(d + 1 <= X)
    return dec


def oracle_moves(b0, valid, tfl, side, jk, t):
    return {a: PD.move_ticks(b0, valid, tfl, side, jk, t, 1000.0 if MUT_HORIZON else h) for a, h in HOR.items()}


def selftest():
    bad = []
    c = lambda pt: dict(pt=pt)
    msk0 = [False]
    g = ir_dec([True], [50], 0, [0.2], msk0, 2)
    if [g(0, 0.0, c(p)) for p in (50, 49, 48)] != [True, False, False]:
        bad.append(f'IR no fair move {[g(0, 0.0, c(p)) for p in (50, 49, 48)]}')
    g = ir_dec([True], [50], 0, [1.45], msk0, 2)
    if [g(0, 0.0, c(p)) for p in (50, 49, 48)] != [True, True, False]:
        bad.append(f'IR 1.45 c {[g(0, 0.0, c(p)) for p in (50, 49, 48)]}')
    ga = ir_dec([True], [50], 1, [0.2], msk0, 2)
    if [ga(0, 0.0, c(p)) for p in (50, 51, 52)] != [True, False, False]:
        bad.append('IR asks')
    if ir_dec([True], [50], 0, [float('nan')], msk0, 2)(0, 0.0, c(48)) is not True:
        bad.append('IR missing fair price must pull')
    if ir_dec([True], [50], 0, [0.2], [True], 2)(0, 0.0, c(48)) is not True:
        bad.append('IR masked step must pull')
    if ir_dec([False], [50], 0, [3.0], msk0, 2)(0, 0.0, c(50)) is not False:
        bad.append('IR guard off must keep')
    if ir_dec([True], [50], 0, [0.2], msk0, 0)(0, 0.0, c(50)) is not True:
        bad.append('IR rung 0 at the best must be pulled')
    tfl = [0.0, 100.0, 1500.0, 2500.0]; valid = np.ones(4, bool); b0 = np.array([50, 50, 48, 47])
    om = oracle_moves(b0, valid, tfl, 0, 0, 0.0)
    if (om['OFB1'], om['OFB2'], om['OFB3']) != (0.0, 2.0, 3.0):
        bad.append(f'oracle horizons {om}')
    return bad


def mutation_check():
    global MUT_IR_SHIFT, MUT_IR_NAN_KEEP, MUT_HORIZON
    out = {}
    for nm in ('MUT_IR_SHIFT', 'MUT_IR_NAN_KEEP', 'MUT_HORIZON'):
        globals()[nm] = 1
        try:
            out[nm] = bool(selftest())
        finally:
            globals()[nm] = 0
    return out


def all_selftests():
    FB.all_selftests(); IN.all_selftests()
    b = selftest(); m = mutation_check()
    if b or not all(m.values()):
        sys.exit(f'TARGET-TEST SELFTEST FAILED -> no output: {b} planted errors caught {m}')
    print(f'target-test selftests passed; planted errors caught: {m}', flush=True)
    return m


# ---------------------------------------------------------------- FULL predictions at every guard-on step
def score_day(day):
    import lightgbm as lgb
    all_selftests()
    T0 = time.time()
    R = pd.read_parquet(f'{SD.OUTD}/rows_{day}.parquet', columns=['m', 'side', 'k', 't'] + SD.NEW_FEAT)
    P = pd.read_parquet(f'{K.IO}/step5/{day}/rows_rl_d2_pk.parquet', columns=['mid', 'tau', 'p_bn', 'tau_rem', 'sigma'])
    P = P[P.mid.isin(set(R.m.unique()))].sort_values(['mid', 'tau'], kind='stable')
    PB = {int(m): (g.tau.values.astype(float).tolist(), g.p_bn.values.astype(float), g.tau_rem.values.astype(float), g.sigma.values.astype(float))
          for m, g in P.groupby('mid', sort=False)}
    del P
    n = len(R); inf = np.full((n, len(IN.INF_FEAT)), np.nan); okp = np.zeros(n, bool)
    mv, sv, kv, tv, gv = R.m.values, R.side.values, R.k.values, R.t.values, R['x31'].values.astype(float)
    for i in range(n):
        m = int(mv[i])
        if m not in PB:
            continue
        rt, pv, trv, sgv = PB[m]; t = float(tv[i]); adv = IN.adv_of(int(sv[i]))
        pk = IN.pbn_at(rt, pv, t)
        if not np.isfinite(pk):
            continue
        jj = bisect.bisect_right(rt, t) - 1
        inf[i] = [adv * (pk - IN.pbn_at(rt, pv, t - 250.0)), adv * (pk - IN.pbn_at(rt, pv, t - 1000.0)), adv * gv[i],
                  IN.sens_of(pk, float(sgv[jj]), float(trv[jj])), abs(pk - 50.0), float(trv[jj])]
        okp[i] = True
    for j_, c_ in enumerate(IN.INF_FEAT):
        R[c_] = inf[:, j_]
    X = R[IN.FULL_FEAT].values.astype(np.float32); prev = np.ones(n); out = dict(m=mv, side=sv, k=kv, dfv250=R.dfv250.values)
    for j in JS:
        b = lgb.Booster(model_file=f'{IN.OUTD}/models/full_{day}_j{j}.txt')
        p = np.minimum(b.predict(X), prev); prev = p; out[f'FULL_P{j}'] = np.where(okp, p, np.nan)
    S = pd.DataFrame(out)
    O = pd.read_parquet(f'{IN.OUTD}/pred_{day}.parquet', columns=['m', 'side', 'k'] + [f'FULL_P{j}' for j in JS])
    Mg = O.merge(S, on=['m', 'side', 'k'], how='left', suffixes=('_on', ''), validate='one_to_one')
    miss = int(Mg.FULL_P1.isna().sum()); diff = max(float(np.nanmax(np.abs(Mg[f'FULL_P{j}_on'].values - Mg[f'FULL_P{j}'].values))) for j in JS)
    st = dict(rows=n, no_fair_price=int((~okp).sum()), onsets=len(O), onsets_unmatched=miss, max_diff_at_onsets=diff)
    if miss or not (diff <= 1e-9):
        sys.exit(f'TARGET-TEST CONTROL FAILED on {day}: FULL at onsets differs from INFLUENCE\'s -> no output: {st}')
    os.makedirs(OUTD, exist_ok=True)
    S.to_parquet(f'{OUTD}/score_{day}.parquet', index=False); json.dump(st, open(f'{OUTD}/score_{day}.json', 'w'))
    print(f'score {day}: {st} in {time.time() - T0:.0f} s', flush=True)


# ---------------------------------------------------------------- the replay (one day)
def eval_day(day):
    import pf_gap_top_makers as GP
    import pf_depth_timing as DT
    mut = all_selftests()
    T0 = time.time(); D = K.load_day(day); st = D['stat']; S1d = f'{K.IO}/step1/{day}'
    tkf = pd.read_parquet(f'{S1d}/legs_tk.parquet', columns=['tk']).tk.values
    LT = D['LT']; ix = LT.index.values
    grp = np.where(tkf[ix] >= 0, tkf[ix], -1 - ix)
    WS = pd.Series(TS.ws_groups(LT.how.values, grp), index=LT.index)
    GRP = pd.Series(grp, index=LT.index)
    TPG = pd.Series(LT.ptick.values).groupby(grp).agg(list).to_dict()
    PO = pd.read_parquet(f'{PD.OUTD}/pred_{day}.parquet'); POI = PO.groupby(['m', 'side']).indices
    PN = pd.read_parquet(f'{SD.OUTD}/pred_{day}.parquet', columns=['m', 'side', 'k'] + [f'NEW_P{j}' for j in JS]); PNI = PN.groupby(['m', 'side']).indices
    PF = pd.read_parquet(f'{OUTD}/score_{day}.parquet'); PFI = PF.groupby(['m', 'side']).indices
    st.update(window_sides=0, ctrl_differ=0, steps_guard=0, miss_old=0, miss_new=0, miss_full=0, clock_ok=1)
    fills = []

    def ttype(r_, x, p):
        return DT.trade_type(x, p, TPG[GRP.at[r_]]) if GRP.at[r_] in TPG else 'n/a'

    def fill_mat(T, I, key, cols):
        Pm = np.full((n, len(JS)), np.nan)
        if key in I:
            ii = I[key]; Pm[T.k.values[ii]] = T[cols].values[ii]
        return Pm
    for w in K.window_sides(D):
        st['window_sides'] += 1; m, side, sg, n = w['m'], w['side'], w['sg'], w['n']
        tf, mff = w['tf_mff']; tfl = tf.tolist(); P5, opp = w['P5'], w['opp']
        key = (m, side)
        Pold = fill_mat(PO, POI, key, [f'P{j}' for j in JS]); Pnew = fill_mat(PN, PNI, key, [f'NEW_P{j}' for j in JS])
        Pfull = fill_mat(PF, PFI, key, [f'FULL_P{j}' for j in JS])
        x250 = np.full(n, np.nan)
        if key in PFI:
            ii = PFI[key]; x250[PF.k.values[ii]] = PF.dfv250.values[ii]
        vv = np.asarray(w['vkl'], float); vg = (vv <= K.GUARD_PULL).tolist(); msk = (vv == MASK_V).tolist()
        b0, valid, rs, kind = QR.runs(tf, P5, opp, side)
        mvh = {a: np.full(n, np.nan) for a in HOR}
        for k in range(n):
            if not vg[k]:
                continue
            st['steps_guard'] += 1; jk = int(w['jk'][k])
            st['miss_old'] += int(not np.isfinite(Pold[k, 0])); st['miss_new'] += int(not np.isfinite(Pnew[k, 0])); st['miss_full'] += int(not np.isfinite(Pfull[k, 0]))
            if jk < 0 or not valid[jk] or w['best'][k] < 0:
                continue
            for a, x in oracle_moves(b0, valid, tfl, side, jk, float(w['tg'][k])).items():
                mvh[a][k] = x
        if (m, side) in D['LI']:
            l = D['LT'].iloc[D['LI'][(m, side)]]
            o_ = np.lexsort(((-l.ptick.values if side == 0 else l.ptick.values), l.anc.values)); lrow = l.index.values[o_]
        else:
            lrow = np.zeros(0, 'int64')
        src = WS.reindex(lrow).values.astype(bool) if len(lrow) else np.zeros(0, bool)
        val = lambda t, h, pt: sg * (mff[max(int(np.searchsorted(tf, t + h, side='right')) - 1, 0)] - pt)   # as FUTURE-BEST, bit for bit
        for q, shf in CFGS:
            tt = w['anc'] - shf; cfg = f'{q}{int(shf)}'
            for d in RUNGS:
                res = {'LDG': K.sim_arm(w, q, shf, 'rest', aux=vg, offset=d, size=SIZE)}
                Fc, Oc, _ = K.sim_arm(w, q, shf, 'rest', aux=vg, decide=FB.fb_dec(vg, w['best'], side, Pold, d, tau=0.0), idle_only=1, offset=d, size=SIZE)
                st['ctrl_differ'] += int(Fc != res['LDG'][0] or Oc != res['LDG'][1])
                for arm in ARMS[1:]:
                    if arm == 'FB25':
                        dec = FB.fb_dec(vg, w['best'], side, Pold, d, tau=TAUS[arm])
                    elif arm == 'FBN25':
                        dec = FB.fb_dec(vg, w['best'], side, Pnew, d, tau=TAUS[arm])
                    elif arm in ('FBF10', 'FBF25'):
                        dec = FB.fb_dec(vg, w['best'], side, Pfull, d, tau=TAUS[arm])
                    elif arm == 'IR':
                        dec = ir_dec(vg, w['best'], side, x250, msk, d)
                    else:
                        dec = FB.fb_dec(vg, w['best'], side, Pold, d, mv=mvh[arm])
                    res[arm] = K.sim_arm(w, q, shf, 'rest', aux=vg, decide=dec, idle_only=1, offset=d, size=SIZE)
                for arm, (F_, O_, pulls) in res.items():
                    st[f'pulls|{arm}|{cfg}'] = st.get(f'pulls|{arm}|{cfg}', 0) + int(pulls)
                    for f in F_:
                        i, pt, qq, thr, age, ph = f[0], int(f[1]), f[2], f[3], f[4], f[5]
                        t = float(w['anc'][i])
                        if not (w['ws'] <= t < w['ws'] + 61000.0):
                            continue
                        d_ = GP.depth_of(tfl, P5, opp, side, t, pt)
                        pre = 'n/a'
                        if d_ == 0:
                            ru = QP.run_at(tfl, b0, valid, rs, kind, float(tt[i]), pt)
                            if ru is not None and ru[1] != 'censored':
                                pre = 'PRE' if float(tt[i]) - age < ru[0] else 'POST'
                        ty = 'THROUGH' if thr else (ttype(lrow[i], side, pt) if src[i] else 'n/a')
                        fills.append((int(day), cfg, arm, d, 'WS' if src[i] else 'E2', -9 if d_ is None else d_, bool(thr), age < 100.0, ph == 'pend',
                                      float(qq), val(t, 1000.0, pt), val(t, 5000.0, pt), pre, ty, float(age)))
        if (K.L_PLACE, K.L_CANCEL, K.L_PENDCANCEL) != (PD.BG.CLK0['L_PLACE'], PD.BG.CLK0['L_CANCEL'], PD.BG.CLK0['L_PENDCANCEL']):
            st['clock_ok'] = 0
        w.pop('cache', None)
        if st['window_sides'] % 80 == 0:
            r2 = K.rss_mib(); print(f'{day}: {st["window_sides"]} window-sides, {time.time() - T0:.0f} s, rss {r2:.0f} MiB', flush=True)
            if r2 > MEM_CAP_MIB:
                sys.exit(f'REFUSED: resident memory {r2:.0f} MiB > {MEM_CAP_MIB:.0f} MiB')
    F = pd.DataFrame(fills, columns=FCOLS)
    R0 = pd.read_parquet(f'{FB.OUTD}/fills_{day}.parquet')
    ref_bad = {}
    for arm, arm0 in (('LDG', 'LDG'), ('FB25', 'FB25'), ('OFB1', 'OFB')):
        for cfg in ('M0', 'M50', 'M250'):
            X = F[(F.arm == arm) & (F.cfg == cfg)][FCOLS[:-1]].drop(columns=['arm']).reset_index(drop=True)
            Y = R0[(R0.arm == arm0) & (R0.cfg == cfg)][FCOLS[:-1]].drop(columns=['arm']).reset_index(drop=True)
            if not FB.same_rows(X, Y):
                ref_bad[f'{arm} {cfg}'] = (len(X), len(Y))
    st['ref_bad'] = ref_bad; st['fills'] = len(F)
    if st['ctrl_differ'] or ref_bad or not st['clock_ok']:
        sys.exit(f'TARGET-TEST CONTROL FAILED on {day} -> no result: { {k_: v_ for k_, v_ in st.items() if not isinstance(v_, (list,))} }')
    st['planted_caught'] = mut
    F.to_parquet(f'{OUTD}/fills_{day}.parquet', index=False)
    json.dump(dict(stat=st, days=len(D['W0']) / 288.0), open(f'{OUTD}/day_{day}.json', 'w'), indent=1, default=float)
    print(f'{day}: {json.dumps({k_: v_ for k_, v_ in st.items() if not isinstance(v_, dict)}, default=float)} in {time.time() - T0:.0f} s', flush=True)


# ---------------------------------------------------------------- report
def report():
    stats = {d: json.load(open(f'{OUTD}/day_{d}.json')) for d in DAYS9}
    A = pd.concat([pd.read_parquet(f'{OUTD}/fills_{d}.parquet') for d in DAYS9], ignore_index=True)
    nbad = int((~np.isfinite(A.v1)).sum()); A = A[np.isfinite(A.v1)]
    dd = {d: stats[d]['days'] for d in DAYS9}; ndays = sum(dd.values())
    cps = lambda x: float((x.q * x.v1).sum() / x.q.sum()) if x.q.sum() else NAN
    asof = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())

    def cell(arm, cfg, srcs=('WS',), d=None, h='v1', aged=True):
        x = A[(A.arm == arm) & (A.cfg == cfg) & A.src.isin(srcs)]
        if aged:
            x = x[~x.young]
        if d:
            x = x[x.day == int(d)]
        return float((x.q * x[h]).sum()) / 100.0 / (dd[d] if d else ndays)
    sc = {d: json.load(open(f'{OUTD}/score_{d}.json')) for d in DAYS9}
    miss = {k_: sum(stats[d]['stat'][k_] for d in DAYS9) for k_ in ('steps_guard', 'miss_old', 'miss_new', 'miss_full')}
    R = {}
    L = [f'TARGET-TEST, 9 consumed replay days, first 60 s, guard ladder; as of {asof}; controls passed (tau = 0 = LDG everywhere; LDG / FB25 / OFB1 = '
         f'FUTURE-BEST row for row at M 0 / 50 / 250; FULL at onsets = INFLUENCE\'s; planted errors caught); fills without a 1 s value {nbad}; guard steps '
         f'{miss["steps_guard"]}, without a prediction: OLD {miss["miss_old"]}, NEW {miss["miss_new"]}, FULL {miss["miss_full"]} (pulled); steps without a '
         f'fair price {sum(s["no_fair_price"] for s in sc.values())}', '',
         'PART 1: $/day at 1 s, w/o young (order >= 100 ms): WS-timed | E2 | all trades || WS all ages | WS 5 s | pulls a day | WS c/sh |'
         ' cancel-pending share of WS shares, its $/day']
    for cfg in ('M0', 'M50', 'M250'):
        for arm in ARMS:
            pl = sum(stats[d]['stat'].get(f'pulls|{arm}|{cfg}', 0) for d in DAYS9) / ndays
            x = A[(A.arm == arm) & (A.cfg == cfg) & (A.src == 'WS') & ~A.young]; xp = x[x.pend]
            L.append(f'  {cfg:<4} {arm:<5} {cell(arm, cfg):+8.1f} | {cell(arm, cfg, ("E2",)):+8.1f} | {cell(arm, cfg, ("WS", "E2")):+8.1f} ||'
                     f' {cell(arm, cfg, aged=False):+8.1f} | {cell(arm, cfg, h="v5"):+8.1f} | {pl:7.0f} | {cps(x):+.3f} |'
                     f' {xp.q.sum() / max(x.q.sum(), 1e-9):4.0%} {float((xp.q * xp.v1).sum()) / 100.0 / ndays:+7.1f}')
        L.append('')

    def compare(a, b, good, badl):
        up = {c: sum(cell(a, c, d=d) > cell(b, c, d=d) for d in DAYS9) for c in ('M0', 'M50', 'M250')}
        dn = {c: sum(cell(a, c, d=d) < cell(b, c, d=d) for d in DAYS9) for c in ('M0', 'M50', 'M250')}
        lab = good if min(up['M0'], up['M50']) >= 7 else (badl if min(dn['M0'], dn['M50']) >= 7 else 'NOT-SHOWN')
        diff = {c: cell(a, c) - cell(b, c) for c in ('M0', 'M50', 'M250')}
        holds = (diff['M250'] > 0) == (diff['M50'] > 0)
        L.append(f'  {a} vs {b}: {lab} (WS $/day: M 0 {diff["M0"]:+.1f}, higher {up["M0"]}/9; M 50 {diff["M50"]:+.1f}, higher {up["M50"]}/9;'
                 f' M 250 {diff["M250"]:+.1f}, higher {up["M250"]}/9, sign {"holds" if holds else "FLIPS"})')
        return dict(label=lab, up=up, down=dn, diff=diff, sign_holds_at_M250=bool(holds))
    L.append('READINGS')
    R['R600'] = compare('OFB2', 'OFB1', '2S-BETTER', '1S-BETTER')
    R['R600b'] = compare('OFB3', 'OFB2', '3S-BETTER', '2S-BETTER')
    R['R601'] = compare('FBF25', 'FB25', 'BETTER', 'WORSE')
    R['R602'] = compare('FBF25', 'LDG', 'HELPS', 'HURTS')
    R['R603_LDG'] = compare('IR', 'LDG', 'HELPS', 'HURTS')
    R['R603_FB25'] = compare('IR', 'FB25', 'HELPS', 'HURTS')
    L.append('  printed:'); R['FBN25_vs_FB25'] = compare('FBN25', 'FB25', 'BETTER', 'WORSE'); R['FBF10_vs_LDG'] = compare('FBF10', 'LDG', 'HELPS', 'HURTS')
    R['FBN25_vs_LDG'] = compare('FBN25', 'LDG', 'HELPS', 'HURTS'); R['FB25_vs_LDG'] = compare('FB25', 'LDG', 'HELPS', 'HURTS')
    R['R604'] = {}
    for a in ('FBF25', 'FBN25', 'IR'):
        hi = sum(cell(a, 'M250', d=d) > cell('LDG', 'M250', d=d) for d in DAYS9); lo = sum(cell(a, 'M250', d=d) < cell('LDG', 'M250', d=d) for d in DAYS9)
        R['R604'][a] = dict(higher=hi, lower=lo, diff=cell(a, 'M250') - cell('LDG', 'M250'),
                            label='HOLDS-AT-M250' if hi >= 7 else ('FLIPS-AT-M250' if lo >= 7 else 'NOT-SHOWN'))
        L.append(f'  R604 {a} vs LDG at M 250: {R["R604"][a]["diff"]:+.1f} $/day, higher {hi} / lower {lo} of 9 -> {R["R604"][a]["label"]}')
    L.append('')
    L.append('  per day, WS $/day w/o young, M 50: ' + ' | '.join(ARMS))
    for d in DAYS9:
        L.append(f'    {d} ' + ' | '.join(f'{cell(a_, "M50", d=d):+7.1f}' for a_ in ARMS))
    L += ['', 'C12 all trades and E2, M 50, minus LDG ($/day w/o young at 1 s; days higher): arm | all trades | E2']
    for a in ARMS[1:]:
        da = cell(a, 'M50', ('WS', 'E2')) - cell('LDG', 'M50', ('WS', 'E2')); de = cell(a, 'M50', ('E2',)) - cell('LDG', 'M50', ('E2',))
        ha = sum(cell(a, 'M50', ('WS', 'E2'), d=d) > cell('LDG', 'M50', ('WS', 'E2'), d=d) for d in DAYS9)
        L.append(f'  {a:<5} {da:+8.1f} ({ha}/9) | {de:+8.1f}')
    L += ['', 'C1 / C12: $/day at 1 s by order age, M 50: WS (< 100 | 100-500 | >= 500 ms) || E2 (same)']
    for arm in ARMS:
        x = A[(A.arm == arm) & (A.cfg == 'M50')]; parts = []
        for s_ in ('WS', 'E2'):
            y = x[x.src == s_]
            parts.append(' | '.join(f'{float((y.q * y.v1)[(y.age >= lo) & (y.age < hi)].sum()) / 100.0 / ndays:+8.1f}' for _, lo, hi in AGEB))
        L.append(f'  {arm:<5} {parts[0]} || {parts[1]}')
    L += ['', 'At the best vs behind (M 50, WS, w/o young): at-best $/day (PRE | POST) | behind $/day | at-best PRE share | PRE c/sh | PRE STOP | POST c/sh']
    for arm in ARMS:
        x = A[(A.arm == arm) & (A.cfg == 'M50') & (A.src == 'WS') & ~A.young]; ab = x[x.depth == 0]; pr = ab[ab.pre == 'PRE']; po = ab[ab.pre == 'POST']
        u = lambda y: float((y.q * y.v1).sum()) / 100.0 / ndays
        L.append(f'  {arm:<5} {u(ab):+7.1f} ({u(pr):+7.1f} | {u(po):+7.1f}) | {u(x[x.depth > 0]):+7.1f} | {pr.q.sum() / max(ab.q.sum(), 1e-9):5.1%} |'
                 f' {cps(pr):+.3f} | {pr.q[pr.type == "STOP"].sum() / max(pr.q.sum(), 1e-9):4.0%} | {cps(po):+.3f}')
    L.append('')
    lab = {k_: v_['label'] if 'label' in v_ else {a: x['label'] for a, x in v_.items()} for k_, v_ in R.items() if k_.startswith('R')}
    L.append('READINGS: ' + json.dumps(lab))
    os.makedirs(OUTD, exist_ok=True)
    json.dump(R, open(f'{OUTD}/target_test.json', 'w'), indent=1, default=float)
    open(f'{OUTD}/target_test.txt', 'w').write('\n'.join(L) + '\n'); print('\n'.join(L))


if __name__ == '__main__':
    cmd = sys.argv[1] if len(sys.argv) > 1 else ''
    if cmd == 'selftest':
        all_selftests()
    elif cmd == 'score':
        score_day(sys.argv[2])
    elif cmd == 'eval':
        eval_day(sys.argv[2])
    elif cmd == 'report':
        report()
    else:
        sys.exit('usage: pf_target_test.py selftest | score <day> | eval <day> | report')
