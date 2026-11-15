"""STOP-MODEL DIAGNOSTICS (2026-11-15, session aea5bfb9): model / data issues before the policy replay - DEVELOPMENT DAYS ONLY.

USER (2026-11-15): "Before playing model performance, can you check if our model/data has any issues? Or preprocess data and refine the model
if it has".

Every decision here uses the development days 10-23..26 only (Rule 11: the five fold test days stay for the evaluation of what v4 fits).
  F1 = fold 10-27's training days 10-23..25 and its calibration day 10-26 (v3.1's saved models and 10-26 predictions = the reference C);
  FD = a dev-only fold for choices that need a day of their own: train 10-23..24, stop / recalibrate on 10-25, evaluate on 10-26.
ISSUES SEEN BEFORE THIS RUN (fit receipts and row counts, no new data read): (a) every head trains and is scored only on rows not ALREADY_j
(D0 < j) while a policy acts on every known state (ALREADY_1 = ~4% of the training rows); (b) 60,000 of ~690,000 rows a day are drawn, so the
deep short-horizon heads have 130-232 positives (H72 j6) and overfit (AUC train / calibration 0.996 / 0.950 at H72 j2, 1.000 / 0.962 at
j3); (c) the 1 s level-1 head over-predicts 1-4 pp on the test days; (d) four fits peaked 2.06-2.18 GiB (model inputs built in float64);
(e) the short-horizon skill rests on Binance at event time + 3.5 ms (P15). Feed-GAP rows are 0.03-0.08% a day (not an issue).
CHECKS AND DECLARED RULES (evaluated in code; an option HELPS = mean relative loss gain over the 18 heads >= 0.5% AND >= 10 heads better):
 Q1 INTEGRITY (each dev day): window-side blocks contiguous with k increasing, labels >= 0 and monotone in H, no infinite input; NaN share per
    input and near-duplicate inputs (|r| > 0.999 on 10-26) printed. Any fault stops the run (fix first).
 Q2 POPULATIONS (each dev day): shares GAP, ALREADY_j, masked; label rates of the head populations C (D0 < j) and U (every row).
 CONTROL C-REPRO: the v4 helpers with target C and V60 reproduce v3.1's saved 10-26 predictions (3 heads, max |dP| <= 1e-6) or the run stops.
 Q3 TARGET POPULATION (F1, 10-26): U = v3.1's rows and seeds with every row in every head (y = D1_H >= j). U is adopted iff its log loss on
    the policy population (every usable row) is below C's on every head with an ALREADY_j share >= 0.5% on 10-26 AND no head is worse by
    more than 1% relative; else C is kept.
 Q4 TRAINING VOLUME (F1, chosen target, its population's 10-26 log loss, 200 rounds): V180 (180,000 rows a day, <= 600,000 a head) and S60
    (V60 + every row positive for a rare head - training-day rate < 2% - the others weighted by their inverse draw rate) against V60 (v3's
    draw). Choice: the HELPS option with the larger mean gain (tie -> S60, cheaper); none -> V60.
 Q5 ROUNDS (FD, chosen target and volume): early stopping on 10-25 (patience 50, <= 1,000 rounds) against the pinned 200; adopted iff it
    HELPS on 10-26 log loss.
 Q6 RECALIBRATION (FD): Platt scaling fitted on 10-25 (on the Q5 choice's predictions) against none; adopted iff it HELPS on 10-26 Brier.
 Q7 TIME IN WINDOW / MASKED (F1 chosen V60 model, 10-26, j = 1..3): AUC, mean P and rate by time bucket (0-5, 5-15, 15-30, 30-60 s) and by
    masked status; FLAGGED if |mean P - rate| > 2 pp or the AUC is > 0.05 below the head's pooled AUC. Printed (no automatic fix).
 Q8 BINANCE LATENCY (P15; F1, j = 1..3): 10-26 AUC with every Binance-derived input one step (50 ms) staler - C's models as they are, and a
    refit on stale inputs (chosen target, V60) - against fresh. Printed: how much of the skill rests on the 3.5 ms assumption.
 Q9 LEVEL MONOTONICITY (FD final predictions): share of rows with P_j > P_(j-1); under U the projection P_j = min(P_j, P_(j-1)) is adopted
    iff the summed 10-26 Brier over the 18 heads is lower with it (under C the heads condition on different rows: not applied).
OUTPUT diag_20261115.json / .txt; its 'decisions' are read by the v4 fit. Controls: pf_stop_model's selftests with planted errors (v4
helpers included), C-REPRO, prediction keys aligned with their rows, memory self-guard 2 GiB.
Usage (run_light, PF_SLOT=claude): pf_stop_model_diag.py selftest | run
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import pf_stop_model as M                                 # noqa: E402

F1_TEST = M.TESTS[0]
F1_TRAIN = list(M.train_days(F1_TEST)); EVAL = M.FOLDS[F1_TEST][1]
FD_TRAIN = ['20261023', '20261024']; FD_STOP = '20261025'
HEADS = [(h, j) for h in M.HS for j in M.JS]
HEADS3 = [(h, j) for h in M.HS for j in (1, 2, 3)]
LABC = ['D0'] + [M.DCOL[h] for h in M.HS]
TB = ((0, 100), (100, 300), (300, 600), (600, 1200))     # steps of 50 ms: 0-5, 5-15, 15-30, 30-60 s
ALREADY_MIN = 0.005; WORSE_MAX = 0.01; FLAG_CAL = 0.02; FLAG_AUC = 0.05; DUP_R = 0.999
REPRO_HEADS = [(72, 1), (250, 3), (1000, 6)]
OUT_J = f'{M.OUTD}/diag_20261115.json'; OUT_T = f'{M.OUTD}/diag_20261115.txt'


def hk(h, j):
    return f'H{h}_j{j}'


def read_day(d, cols):
    R = pd.read_parquet(f'{M.OUTD}/rows_{d}.parquet', columns=cols)
    return R[M.usable(R)].reset_index(drop=True)


def day_rates(days, target):
    """Population rate of every head over the days (usable rows; head rows by target)."""
    n = {hj: 0 for hj in HEADS}; p = {hj: 0 for hj in HEADS}
    for d in days:
        R = read_day(d, LABC + ['hgap'])
        for (h, j) in HEADS:
            r = M.head_rows(R.D0.values, j, target); n[(h, j)] += int(r.sum()); p[(h, j)] += int((R[M.DCOL[h]].values[r] >= j).sum())
    return {hj: p[hj] / max(1, n[hj]) for hj in HEADS}


def train_table(days, seed_of, vol, cut, F, scale, stale=False):
    """Training rows of the days under a volume option: X (float32), labels, weights, stratum flags."""
    mode, sub, _ = M.VOL[vol]; Xs, Ls, Ws, Zs = [], [], [], []
    for d in days:
        R = read_day(d, M.KEYS + M.RAW_NEEDED + LABC)
        lab = {c: R[c].to_numpy(np.float32) for c in LABC}
        z = M.stratum(lab, cut) if mode == 'S' else None
        ix, w = M.day_sample(len(R), sub, seed_of(d), z)
        if stale:
            X = M.model_matrix(R, scale, F)
            M.stale_shift(X, [F.index(c) for c in M.BN_F if c in F], M.stale_prev(R.m.values, R.side.values, R.k.values)); X = X[ix]
        else:
            X = M.model_matrix(R.iloc[ix].reset_index(drop=True), scale, F)
        Xs.append(X); Ls.append({c: v[ix] for c, v in lab.items()}); Ws.append(w)
        Zs.append(z[ix] if z is not None else np.zeros(len(ix), bool)); del R
        M.mem_guard(f'diag table {d}')
    X = np.concatenate(Xs); del Xs
    return X, {c: np.concatenate([x[c] for x in Ls]) for c in LABC}, np.concatenate(Ws), np.concatenate(Zs)


def train_heads(T, F, target, vol, seed_head, heads, valid=None):
    """Boosters of the heads on a training table T = (X, labels, weights, stratum): -> {(h, j): (booster, rounds, base rate, positives, rows)}."""
    X, lab, w, z = T; cap = M.VOL[vol][2]; out = {}
    for (h, j) in heads:
        ii = np.flatnonzero(M.head_rows(lab['D0'], j, target))
        ic, wc = M.head_cap(z[ii] if z.any() else None, w[ii], cap, seed_head(h, j)); ii = ii[ic]
        y = (lab[M.DCOL[h]][ii] >= j).astype(int); vv = None
        if valid is not None:
            Xv, lv = valid; rv = M.head_rows(lv['D0'], j, target); vv = (Xv[rv], (lv[M.DCOL[h]][rv] >= j).astype(int))
        b, n = M.fit_head(X[ii], y, wc, F, valid=vv)
        out[(h, j)] = (b, n, float(np.average(y, weights=wc)), int(y.sum()), int(len(y)))
        M.mem_guard(f'diag head H{h} j{j}')
    return out


def predict(B, X, heads, n_iter=None):
    return {hj: B[hj][0].predict(X, num_iteration=(n_iter or B[hj][1])).astype(np.float32) for hj in heads}


def metrics(P, lab, target_pop, heads):
    """Per head on 10-26: rows, rate, mean P, AUC, log loss, Brier; target_pop 'C' = rows not ALREADY_j, 'U' = every usable row."""
    out = {}
    for (h, j) in heads:
        r = M.head_rows(lab['D0'], j, target_pop); y = lab[M.DCOL[h]][r] >= j; p = P[(h, j)][r].astype(float)
        out[hk(h, j)] = dict(n=int(r.sum()), rate=float(y.mean()), meanP=float(p.mean()), auc=M.auc_nan(y, p)[0], ll=M.logloss(y, p),
                             brier=M.brier(y, p))
    return out


def run():
    M.all_selftests()
    T0 = time.time(); L = []; res = dict(asof=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), version='diag 2026-11-15')
    A = json.load(open(f'{M.OUTD}/audit2.json')); scale = A['scale']; F = [c for c in M.FEAT_M if c not in set(A['drop'])]
    L.append(f'STOP-MODEL DIAGNOSTICS (dev days {", ".join(M.DEV_DAYS)} only); as of {res["asof"]}; inputs {len(F)}, scale {scale}')
    # ---- Q1 integrity, Q2 populations
    q1 = {}; q2 = {}; nan_sh = {}
    for d in M.DEV_DAYS:
        R = pd.read_parquet(f'{M.OUTD}/rows_{d}.parquet', columns=M.KEYS + M.RAW_NEEDED + LABC + ['masked', 'quiet'])
        X = M.model_matrix(R, scale, F)
        q1[d] = M.integrity(R.m.values, R.side.values, R.k.values, {c: R[c].values for c in M.LAB}, X)
        nan_sh[d] = {c: float(np.isnan(X[:, i]).mean()) for i, c in enumerate(F)}
        u = M.usable(R); D0 = R.D0.values
        q2[d] = dict(rows=int(len(R)), gap=float((~u).mean()), masked=float(R.masked.values.mean()),
                     already={f'j{j}': float((D0[u] >= j).mean()) for j in M.JS},
                     rate_C={hk(h, j): float((R[M.DCOL[h]].values[u & (D0 < j)] >= j).mean()) for (h, j) in HEADS},
                     rate_U={hk(h, j): float((R[M.DCOL[h]].values[u] >= j).mean()) for (h, j) in HEADS})
        del R, X
        M.mem_guard(f'diag Q1 {d}')
    res['Q1'] = q1; res['Q2'] = q2
    L.append('Q1 INTEGRITY: ' + '; '.join(f'{d} ' + ('ok' if q1[d]['ok'] else f'FAULT {q1[d]}') for d in M.DEV_DAYS))
    hi_nan = {c: max(nan_sh[d][c] for d in M.DEV_DAYS) for c in F}
    L.append('   inputs with NaN share > 1% on some dev day: ' + ', '.join(f'{c} {v:.1%}' for c, v in sorted(hi_nan.items(), key=lambda x: -x[1]) if v > 0.01))
    if not all(q1[d]['ok'] for d in M.DEV_DAYS):
        json.dump(res, open(OUT_J, 'w'), indent=1, default=float); open(OUT_T, 'w').write('\n'.join(L) + '\n')
        sys.exit('REFUSED: integrity faults on a dev day (fix first): ' + json.dumps(q1))
    L.append('Q2 POPULATIONS (usable rows): GAP | masked | ALREADY_j j1..j6 | rate H1000 j1 C / U')
    for d in M.DEV_DAYS:
        x = q2[d]
        L.append(f'   {d}: GAP {x["gap"]:.3%} | masked {x["masked"]:.1%} | ' + ' '.join(f'{x["already"][f"j{j}"]:.2%}' for j in M.JS)
                 + f' | {x["rate_C"]["H1000_j1"]:.3f} / {x["rate_U"]["H1000_j1"]:.3f}')
    # ---- the evaluation day (10-26) and C (v3.1's saved predictions)
    E = read_day(EVAL, M.KEYS + M.RAW_NEEDED + LABC + ['masked'])
    lab = {c: E[c].to_numpy(np.float32) for c in LABC}; kk = E.k.values; msk = E.masked.values
    PCf = pd.read_parquet(f'{M.OUTD}/pred_fold{F1_TEST}_{EVAL}.parquet'); PCf = PCf[PCf.usable.values].reset_index(drop=True)
    if not (np.array_equal(PCf.m.values, E.m.values) and np.array_equal(PCf.side.values, E.side.values) and np.array_equal(PCf.k.values, kk)):
        sys.exit('REFUSED: v3.1 predictions of 10-26 not aligned with its rows')
    PC = {(h, j): PCf[M.pcol(h, j)].to_numpy(np.float32) for (h, j) in HEADS}; del PCf
    X26 = M.model_matrix(E, scale, F); keys26 = (E.m.values.copy(), E.side.values.copy(), kk.copy()); del E
    M.mem_guard('diag eval day')
    s1 = lambda d: int(F1_TEST) * 31 + int(d)                                    # noqa: E731  (v3.1's seeds)
    h1 = lambda h, j: int(F1_TEST) * 7 + j * 1009 + h                             # noqa: E731
    T60 = train_table(F1_TRAIN, s1, 'V60', None, F, scale)
    # ---- CONTROL C-REPRO
    BR = train_heads(T60, F, 'C', 'V60', h1, REPRO_HEADS); PR = predict(BR, X26, REPRO_HEADS)
    rep = {hk(*hj): float(np.abs(PR[hj] - PC[hj]).max()) for hj in REPRO_HEADS}; res['C_REPRO'] = rep
    L.append(f'CONTROL C-REPRO (v4 helpers, target C, V60 vs v3.1 saved 10-26 predictions): max |dP| {rep}')
    if max(rep.values()) > 1e-6:
        json.dump(res, open(OUT_J, 'w'), indent=1, default=float); open(OUT_T, 'w').write('\n'.join(L) + '\n')
        sys.exit(f'REFUSED: the v4 helpers do not reproduce v3.1 {rep}')
    del BR, PR
    # ---- Q3 target population
    BU = train_heads(T60, F, 'U', 'V60', h1, HEADS); PU = predict(BU, X26, HEADS)
    mC_U = metrics(PC, lab, 'U', HEADS); mU_U = metrics(PU, lab, 'U', HEADS); mC_C = metrics(PC, lab, 'C', HEADS); mU_C = metrics(PU, lab, 'C', HEADS)
    alr = {j: float((lab['D0'] >= j).mean()) for j in M.JS}
    need = [hk(h, j) for (h, j) in HEADS if alr[j] >= ALREADY_MIN]
    rel = {k: (mU_U[k]['ll'] - mC_U[k]['ll']) / mC_U[k]['ll'] for k in mC_U}
    q3_ok = bool(need) and all(mU_U[k]['ll'] < mC_U[k]['ll'] for k in need) and max(rel.values()) <= WORSE_MAX
    target = 'U' if q3_ok else 'C'
    res['Q3'] = dict(already=alr, need=need, C_on_U=mC_U, U_on_U=mU_U, C_on_C=mC_C, U_on_C=mU_C, rel_ll_U_minus_C=rel, adopt_U=q3_ok,
                     ALREADY_P_C={f'j{j}': dict(meanP=float(PC[(1000, j)][lab['D0'] >= j].mean()), rate=float((lab['D1'][lab['D0'] >= j] >= j).mean()))
                                  for j in (1, 2, 3)})
    L.append(f'Q3 TARGET POPULATION (10-26; ALREADY_j share ' + ' '.join(f'j{j} {alr[j]:.2%}' for j in M.JS) + ')')
    L.append('   head      | policy population (every usable row): rate, mean P C / U, AUC C / U, log loss C / U | rows not ALREADY_j: AUC C / U, log loss C / U')
    for (h, j) in HEADS:
        k = hk(h, j); a, b, c_, d_ = mC_U[k], mU_U[k], mC_C[k], mU_C[k]
        L.append(f'   {k:9s} | {a["rate"]:.4f} {a["meanP"]:.4f}/{b["meanP"]:.4f} {a["auc"]:.3f}/{b["auc"]:.3f} {a["ll"]:.5f}/{b["ll"]:.5f} ({rel[k]:+.2%})'
                 f' | {c_["auc"]:.3f}/{d_["auc"]:.3f} {c_["ll"]:.5f}/{d_["ll"]:.5f}')
    L.append('   C on ALREADY rows (H1000): ' + ', '.join(f'j{j} mean P {v["meanP"]:.3f} vs rate {v["rate"]:.3f}' for j, v in
                                                  zip((1, 2, 3), res['Q3']['ALREADY_P_C'].values())))
    L.append(f'   -> target {target} (U adopted: {q3_ok}; heads needing it {len(need)}, worst relative change {max(rel.values()):+.2%})')
    P60 = PU if target == 'U' else PC; m60 = mU_U if target == 'U' else mC_C
    del BU
    M.mem_guard('diag Q3')
    # ---- Q4 training volume
    r_f1 = day_rates(F1_TRAIN, target); cut1 = M.rare_levels(r_f1)
    q4 = dict(rare_cut={str(h): v for h, v in cut1.items()}, options={})
    for vol in ('S60', 'V180'):
        Tv = train_table(F1_TRAIN, s1, vol, cut1, F, scale)
        Bv = train_heads(Tv, F, target, vol, h1, HEADS); Pv = predict(Bv, X26, HEADS); mv = metrics(Pv, lab, target, HEADS)
        mr, nb, ok = M.helps([m60[hk(*hj)]['ll'] for hj in HEADS], [mv[hk(*hj)]['ll'] for hj in HEADS])
        q4['options'][vol] = dict(metrics=mv, mean_gain=mr, heads_better=nb, helps=ok, rows=int(len(Tv[1]['D0'])),
                                  positives={hk(*hj): Bv[hj][3] for hj in HEADS})
        del Tv, Bv, Pv
        M.mem_guard(f'diag Q4 {vol}')
    hl = [v for v in ('S60', 'V180') if q4['options'][v]['helps']]
    vol = 'V60' if not hl else (max(hl, key=lambda v: (q4['options'][v]['mean_gain'], v == 'S60')))
    q4['choice'] = vol; res['Q4'] = q4
    L.append(f'Q4 TRAINING VOLUME (F1, target {target}, 10-26 log loss vs V60; rare levels {cut1}): '
             + '; '.join(f'{v} rows {q4["options"][v]["rows"]} gain {q4["options"][v]["mean_gain"]:+.2%} better {q4["options"][v]["heads_better"]}/18 '
                         f'{"HELPS" if q4["options"][v]["helps"] else "no"}' for v in ('S60', 'V180')) + f' -> {vol}')
    L.append('   per head log loss V60 / S60 / V180: ' + ' '.join(f'{hk(*hj)} {m60[hk(*hj)]["ll"]:.5f}/{q4["options"]["S60"]["metrics"][hk(*hj)]["ll"]:.5f}/'
                                                           f'{q4["options"]["V180"]["metrics"][hk(*hj)]["ll"]:.5f}' for hj in HEADS))
    # ---- Q5 rounds, Q6 recalibration, Q9 monotonicity (FD)
    sD = lambda d: int(EVAL) * 31 + int(d)                                        # noqa: E731
    hD = lambda h, j: int(EVAL) * 7 + j * 1009 + h                                 # noqa: E731
    cutD = M.rare_levels(day_rates(FD_TRAIN, target))
    TD = train_table(FD_TRAIN, sD, vol, cutD, F, scale)
    V = read_day(FD_STOP, M.KEYS + M.RAW_NEEDED + LABC)
    iv = np.sort(np.random.default_rng(int(FD_STOP)).choice(len(V), min(len(V), M.VAL_N), replace=False))
    V = V.iloc[iv].reset_index(drop=True); XV = M.model_matrix(V, scale, F); labV = {c: V[c].to_numpy(np.float32) for c in LABC}; del V
    BF = train_heads(TD, F, target, vol, hD, HEADS); BE = train_heads(TD, F, target, vol, hD, HEADS, valid=(XV, labV)); del TD
    PF = predict(BF, X26, HEADS); PE = predict(BE, X26, HEADS)
    mF = metrics(PF, lab, target, HEADS); mE = metrics(PE, lab, target, HEADS)
    mr5, nb5, ok5 = M.helps([mF[hk(*hj)]['ll'] for hj in HEADS], [mE[hk(*hj)]['ll'] for hj in HEADS])
    res['Q5'] = dict(fixed=mF, early=mE, rounds={hk(*hj): BE[hj][1] for hj in HEADS}, mean_gain=mr5, heads_better=nb5, adopt=ok5)
    L.append(f'Q5 ROUNDS (FD {"+".join(FD_TRAIN)} -> stop {FD_STOP} -> 10-26, target {target}, {vol}): early stopping gain {mr5:+.2%}, better '
             f'{nb5}/18 -> {"ADOPT" if ok5 else "keep 200"}; rounds ' + ' '.join(f'{hk(*hj)} {BE[hj][1]}' for hj in HEADS))
    L.append('   per head log loss 200 / early: ' + ' '.join(f'{hk(*hj)} {mF[hk(*hj)]["ll"]:.5f}/{mE[hk(*hj)]["ll"]:.5f}' for hj in HEADS))
    BQ = BE if ok5 else BF; PQ = PE if ok5 else PF; mQ = mE if ok5 else mF
    PVq = predict(BQ, XV, HEADS); ab = {}; PP = {}
    for (h, j) in HEADS:
        rv = M.head_rows(labV['D0'], j, target); ab[(h, j)] = M.platt_fit(PVq[(h, j)][rv], labV[M.DCOL[h]][rv] >= j)
        PP[(h, j)] = M.platt_apply(PQ[(h, j)], ab[(h, j)]).astype(np.float32)
    mP = metrics(PP, lab, target, HEADS)
    mr6, nb6, ok6 = M.helps([mQ[hk(*hj)]['brier'] for hj in HEADS], [mP[hk(*hj)]['brier'] for hj in HEADS])
    res['Q6'] = dict(raw=mQ, platt=mP, ab={hk(*hj): v for hj, v in ab.items()}, mean_gain=mr6, heads_better=nb6, adopt=ok6)
    L.append(f'Q6 RECALIBRATION (Platt on {FD_STOP} -> 10-26 Brier): gain {mr6:+.2%}, better {nb6}/18 -> {"ADOPT" if ok6 else "none"}')
    L.append('   per head mean P raw / Platt vs rate: ' + ' '.join(f'{hk(*hj)} {mQ[hk(*hj)]["meanP"]:.4f}/{mP[hk(*hj)]["meanP"]:.4f}/{mQ[hk(*hj)]["rate"]:.4f}'
                                                            for hj in HEADS))
    PZ = PP if ok6 else PQ
    mono = {f'H{h}': {f'j{j}': float((PZ[(h, j)] > PZ[(h, j - 1)] + 1e-7).mean()) for j in M.JS[1:]} for h in M.HS}
    cm = False; bs0 = bs1 = None
    if target == 'U':
        PM = {}
        for h in M.HS:
            Z = np.minimum.accumulate(np.stack([PZ[(h, j)] for j in M.JS], axis=1), axis=1)
            for i, j in enumerate(M.JS):
                PM[(h, j)] = Z[:, i]
        mM = metrics(PM, lab, target, HEADS); mZ = metrics(PZ, lab, target, HEADS)
        bs0 = sum(mZ[hk(*hj)]['brier'] for hj in HEADS); bs1 = sum(mM[hk(*hj)]['brier'] for hj in HEADS); cm = bool(bs1 < bs0)
    res['Q9'] = dict(nonmonotone=mono, brier_sum=bs0, brier_sum_cummin=bs1, adopt=cm)
    L.append('Q9 LEVEL MONOTONICITY (share P_j > P_(j-1)): ' + '; '.join(f'H{h} ' + ' '.join(f'{v:.1%}' for v in mono[f'H{h}'].values()) for h in M.HS)
             + (f'; summed Brier {bs0:.6f} -> cummin {bs1:.6f} -> {"ADOPT" if cm else "none"}' if target == 'U' else '; (target C: not applied)'))
    del BF, BE, PF, PE, PVq, XV
    M.mem_guard('diag FD')
    # ---- Q7 time in window / masked (F1 chosen-target V60 model)
    q7 = {}; fl = []
    for (h, j) in HEADS3:
        r = M.head_rows(lab['D0'], j, target); y = lab[M.DCOL[h]] >= j; p = P60[(h, j)]; ap = m60[hk(h, j)]['auc']; row = {}
        for nm, sel in [(f'{a * 50 / 1000:.0f}-{b * 50 / 1000:.0f}s', (kk >= a) & (kk < b)) for a, b in TB] + [('masked', msk), ('unmasked', ~msk)]:
            s = r & sel
            if s.sum() < 100:
                continue
            x = dict(n=int(s.sum()), rate=float(y[s].mean()), meanP=float(p[s].mean()), auc=M.auc_nan(y[s], p[s])[0])
            x['flag'] = bool(abs(x['meanP'] - x['rate']) > FLAG_CAL or (x['auc'] == x['auc'] and x['auc'] < ap - FLAG_AUC)); row[nm] = x
            if x['flag']:
                fl.append(f'{hk(h, j)} {nm}')
        q7[hk(h, j)] = row
    res['Q7'] = dict(buckets=q7, flagged=fl)
    L.append(f'Q7 TIME IN WINDOW / MASKED (F1 {target} V60, 10-26): flagged {fl if fl else "none"}')
    for (h, j) in HEADS3:
        L.append(f'   {hk(h, j):9s} ' + ' | '.join(f'{nm} {x["rate"]:.3f}/{x["meanP"]:.3f} AUC {x["auc"]:.3f}' for nm, x in q7[hk(h, j)].items()))
    # ---- Q8 Binance latency (stale inputs) - last: X26 is shifted in place
    import lightgbm as lgb
    fresh_C = {hk(*hj): mC_C[hk(*hj)]['auc'] for hj in HEADS3}; fresh_T = {hk(*hj): m60[hk(*hj)]['auc'] for hj in HEADS3}
    bi = [F.index(c) for c in M.BN_F if c in F]
    M.stale_shift(X26, bi, M.stale_prev(*keys26))
    BC = {hj: (lgb.Booster(model_file=f'{M.OUTD}/models/fold{F1_TEST}_H{hj[0]}_j{hj[1]}.txt'), M.PD.N_ROUNDS) for hj in HEADS3}
    stale_C = {hk(*hj): v['auc'] for hj, v in zip(HEADS3, metrics(predict(BC, X26, HEADS3), lab, 'C', HEADS3).values())}
    TS_ = train_table(F1_TRAIN, s1, 'V60', None, F, scale, stale=True)
    BS = train_heads(TS_, F, target, 'V60', h1, HEADS3); del TS_
    stale_T = {k: v['auc'] for k, v in metrics(predict(BS, X26, HEADS3), lab, target, HEADS3).items()}
    res['Q8'] = dict(binance_inputs=[F[i] for i in bi], fresh_C=fresh_C, stale_C_asis=stale_C, fresh_T=fresh_T, stale_T_refit=stale_T)
    L.append(f'Q8 BINANCE LATENCY ({len(bi)} Binance-derived inputs one step = 50 ms staler; 10-26 AUC): head fresh C | stale C as is | fresh '
             f'{target} | stale {target} refit')
    for (h, j) in HEADS3:
        k = hk(h, j)
        L.append(f'   {k:9s} {fresh_C[k]:.3f} | {stale_C[k]:.3f} | {fresh_T[k]:.3f} | {stale_T[k]:.3f}')
    # ---- decisions
    dec = dict(target=target, volume=vol, early_stop=ok5, platt=ok6, cummin=cm, rule=f'HELPS: mean gain >= {M.HELP_MEAN:.1%} and >= {M.HELP_HEADS}/18 heads')
    res['decisions'] = dec; res['seconds'] = round(time.time() - T0, 1)
    L.append(f'DECISIONS (dev days only, declared rules): {json.dumps(dec)}; {res["seconds"]} s')
    json.dump(res, open(OUT_J, 'w'), indent=1, default=float); open(OUT_T, 'w').write('\n'.join(L) + '\n')
    print('\n'.join(L), flush=True)


if __name__ == '__main__':
    cmd = sys.argv[1] if len(sys.argv) > 1 else 'selftest'
    if cmd == 'selftest':
        M.all_selftests()
    elif cmd == 'run':
        run()
    else:
        sys.exit(f'unknown command {cmd}')
