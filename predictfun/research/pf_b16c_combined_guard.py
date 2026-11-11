"""B16c (2026-11-11, USER: "I don't think we can just switch 1s to 60s, maybe a combined version of multiple windows make more sense"):
B16 stage 1 with the guard trained on a COMBINED target - the mean of our resting order's value at 1 s, 5 s and 60 s (qlm, from
pf_b16_queue_label.py with an output path: step4/badfill_info_tx_tk/grid_ql2.parquet). B16 found that a 1 s target with the pressure
features (qb) buys 1 s fills at the expense of 60 s and settlement; B12g found the top makers' fills stay good at 1 s, 60 s and at
settlement. Arms: 'qc' = the current 32 features, 'qd' = + B13a's 13 pressure features. Everything else is B16's pipeline, imported
(pf_b16_queue_guard, which imports B14's): the live guard's recipe, LODO on the 8 read days, 11-03 from the fit on all 8, the frozen
line rule and map, the frozen v3 engine at J and X; benchmark REF = the live guard (B14's b14rJ / b14rX, out-of-fold base_* values).
This target was chosen after B16's results were seen: the 9 days are in-sample for that choice (exploratory; a winner needs >= 5
untouched days).

Readings DECLARED BEFORE THE FIRST RUN:
 R190 decision (PRIMARY: B18 found the replay scores pressure fills like any other, so the replay cannot reward pulling there; this
      referee is the tape): on the grid rows each guard lets quote (its out-of-fold v above its own per-minute 90th percentile on the
      grid rows, bid and ask pooled), the mean COMBINED label, arm minus REF per day, G = 8, t(7): IMPROVES if the interval is above 0
      and >= 6 of 8 days are positive; WORSE if below 0; else NO-CHANGE. VOID (both arms) if the falsifier (arm qd refit on combined
      labels permuted within each day) also reads IMPROVES. Beside: the same at 1 s, 5 s and 60 s separately.
 R191 settle, R192 +1 s, R193 +60 s: the v3 replay, arm minus REF per day, B14's rule (BETTER if the 9-day sum > 0 at J and at X
      and >= 6 of 9 days positive at each; WORSE if < 0 at both and <= 3 of 9 positive at each; else MIXED), per arm.
Controls (any failure -> no table): the label report's K1 (all sweep rows equal the old label) on the new file; grid rows aligned;
the falsifier keeps each day's label multiset; B16 / B14's map controls (C2 / C3 with the planted p90 + 0.001); C5 alignment; every
replay's env as declared; the reading rules (B14 / B16 selftests).
Limits: as B16 (rule M credits every shrink ahead of us; every grid row an order at the back; the replay's fill model is not
validated and misses pressure adverse selection, B18) and the 60 s value carries more of the window's drift (noisier labels).
Usage: predictfun/research/run_b16c.sh (each step through run_light.sh):
  python3 -m predictfun.research.pf_b16c_combined_guard build qc | build qd | map qc | map qd | report
"""
import json
import os
import sys

import numpy as np
import pandas as pd

from . import pf_io as io
from . import pf_b14_pressure_guard as B
from . import pf_b16_queue_guard as G

READ, FRESH, DAYS, S4, S5 = B.READ, B.FRESH, B.DAYS, B.S4, B.S5
QL2 = f'{S4}/badfill_info_tx_tk/grid_ql2.parquet'
QL2_REPORT = f'{S4}/badfill_info_tx_tk/grid_ql2_report.json'
OUTD = f'{io.OUT}/b16c_20261111'
ARMS = {'qc': ('preds_qc60', B.FE), 'qd': ('preds_qd60', B.EMPH)}
TAGS = {'qc': 'b16c', 'qd': 'b16d'}
G.ARMS.update(ARMS)                       # B16's map step reads the arm's stem from its ARMS
LABEL = 'qlm'
SEED = 20261112


def load_inputs():
    rq = json.load(open(QL2_REPORT))
    if rq.get('k1_rows', 0) < 1000 or rq.get('k1_equal') != rq.get('k1_rows') or 'mean_qlm' not in rq:
        sys.exit('REFUSED: the combined label report fails K1 or lacks the combined label')
    Gd, miss = B.load_grid()
    Q = pd.read_parquet(QL2, columns=['day', 'mid', 'tau', f'{LABEL}_bid', f'{LABEL}_ask'])
    if len(Q) != len(Gd) or not ((Q.day.values == Gd.day.values).all() and (Q.mid.values == Gd.mid.values).all() and (Q.tau.values == Gd.tau.values).all()):
        sys.exit('REFUSED: labels not aligned with the grid rows')
    return Gd, Q, miss


def step_build(arm):
    stem, feats = ARMS[arm]
    for d in DAYS:
        for st_ in (stem, stem + 'k', stem + 'km'):
            if os.path.exists(f'{S5}/{d}/{st_}.parquet'):
                sys.exit(f'REFUSED: {S5}/{d}/{st_}.parquet exists')
    os.makedirs(OUTD, exist_ok=True)
    Gd, Q, miss = load_inputs(); io.mem_check('inputs')
    nan_lab = {s: int(np.isnan(Q[f'{LABEL}_{s}'].values).sum()) for s in ('bid', 'ask')}
    Y = {s: np.nan_to_num(Q[f'{LABEL}_{s}'].values, nan=0.0).clip(-10, 10) for s in ('bid', 'ask')}
    del Q
    X = np.concatenate([Gd[B.FE].values.astype(np.float32)] + ([Gd[B.EXTRA].values.astype(np.float32)] if feats is B.EMPH else []), axis=1)
    if X.shape[1] != len(feats):
        sys.exit(f'REFUSED: training matrix {X.shape[1]} wide, expected {len(feats)}')
    day = Gd.day.values
    oof = pd.DataFrame(dict(day=Gd.day.values, mid=Gd.mid.values, tau=Gd.tau.values, t_rel=Gd.t_rel.values))
    del Gd
    for c in (f'{arm}_bid', f'{arm}_ask', f'{arm}perm_bid', f'{arm}perm_ask'):
        oof[c] = np.float32(np.nan)
    rep = dict(arm=arm, stem=stem, label=LABEL, features=len(feats), grid_rows=int(len(X)), label_nan_set_to_0=nan_lab,
               grid_rows_without_any_engineered_value=miss, thresholds={}, falsifier=None)
    YP = None
    if arm == 'qd':
        rng = np.random.default_rng(SEED); YP = {s: Y[s].copy() for s in Y}
        for d in READ:
            ix = np.nonzero(day == d)[0]; pm = rng.permutation(ix)
            for s in YP:
                YP[s][ix] = Y[s][pm]
        same = all(np.array_equal(np.sort(YP[s][day == d]), np.sort(Y[s][day == d])) for s in Y for d in READ)
        rep['falsifier'] = dict(seed=SEED, multiset_kept=bool(same))
        if not same:
            sys.exit('REFUSED: the falsifier permutation changed a day\'s labels')
    for d in READ:
        tr = day != d; te = day == d
        ms, cs = B.fit(X[tr], {s: Y[s][tr] for s in Y}); rep['thresholds'][d] = cs
        for s in ('bid', 'ask'):
            oof.loc[te, f'{arm}_{s}'] = (ms[s].predict(X[te], num_threads=1) - cs[s]).astype(np.float32)
        rows, live, XD = B.day_matrix(d, feats)
        G.write_preds(stem, d, B.apply(ms, cs, live, XD), rows); del ms, XD
        if YP is not None:
            mp, cp = B.fit(X[tr], {s: YP[s][tr] for s in YP})
            for s in ('bid', 'ask'):
                oof.loc[te, f'{arm}perm_{s}'] = (mp[s].predict(X[te], num_threads=1) - cp[s]).astype(np.float32)
            del mp
        io.mem_check(f'{arm} fold {d}'); print(arm, 'fold', d, cs, flush=True)
    ms, cs = B.fit(X, Y); rep['thresholds'][FRESH] = cs
    rows, live, XD = B.day_matrix(FRESH, feats)
    G.write_preds(stem, FRESH, B.apply(ms, cs, live, XD), rows)
    for s in ('bid', 'ask'):
        ms[s].save_model(f'{OUTD}/guard_{arm}_{s}.txt')
    oof.to_parquet(f'{OUTD}/oof_{arm}.parquet', index=False)
    io.write_json(f'{OUTD}/build_{arm}_report.json', rep)
    print(json.dumps({k: rep[k] for k in ('arm', 'grid_rows', 'label_nan_set_to_0', 'falsifier')}), flush=True)


def step_map(arm):
    G.OUTD = OUTD                          # B16's map writes its report under OUTD
    G.step_map(arm)


def summary(tag, d):
    from .pf_inv_report import name
    from .pf_ladder_v3_build import ARM
    stem = {'b14r': 'preds_t60k', 'b16c': 'preds_qc60km', 'b16d': 'preds_qd60km'}[tag[:4]]
    s = json.load(open(f'{S4}/ladder_tokyoreal_custom_e2_{stem}_{tag}/{d}/summary.json'))
    return s, s['arms'][name(ARM)], stem


def step_report():
    from .pf_ladder_v3_build import ARM
    rep = dict(arm=ARM, label=LABEL, readings={})
    for arm, tag in TAGS.items():
        for tm in ('J', 'X'):
            for d in DAYS:
                s, _, stem = summary(f'{tag}{tm}', d); e = s['env']
                want = dict(PF_PREDS=stem, PF_ARMS=ARM, PF_TIMING='e2', PF_TXTIME='ws', PF_TXKEY='tk', PF_RL_LPLACE='40', PF_RL_LCANCEL='22',
                            PF_RL_LPENDCANCEL='95', PF_RL_LFILL='100', PF_RL_DDEC='2')
                want.update(dict(PF_TXJIT='frame') if tm == 'J' else dict(PF_TXSHIFT='-50', PF_QACT='max'))
                bad = {k: (e.get(k), v) for k, v in want.items() if e.get(k) != v}
                if bad or e.get('PF_R60_FILE'):
                    sys.exit(f'REFUSED: {tag}{tm} {d} env differs: {bad}')
    J = pd.read_parquet(f'{B.OUTD}/oof_grid.parquet')                      # B14: old labels, base_* (REF out of fold)
    Q = pd.read_parquet(QL2).rename(columns={'qa_bid': 'ahead_bid', 'qa_ask': 'ahead_ask'})
    J = J.merge(Q, on=['day', 'mid', 'tau'], how='left', validate='one_to_one')
    for arm in TAGS:
        A = pd.read_parquet(f'{OUTD}/oof_{arm}.parquet', columns=['day', 'mid', 'tau', f'{arm}_bid', f'{arm}_ask', f'{arm}perm_bid', f'{arm}perm_ask'])
        J = J.merge(A, on=['day', 'mid', 'tau'], how='left', validate='one_to_one')
    if J[['qc_bid', 'qd_bid', 'base_bid', 'qlm_bid']].isna().all(axis=0).any():
        sys.exit('REFUSED: a guard or label column is empty after the joins')
    labs = {'combined': 'qlm', '1 s': 'ql', '5 s': 'ql5', '60 s': 'ql60'}
    ref = {k: G.quoted_means(J, 'base', v) for k, v in labs.items()}
    fal = G.quoted_means(J, 'qdperm', 'qlm'); dF = [fal[d]['mean'] - ref['combined'][d]['mean'] for d in READ]; lab_f = B.label_r120(dF)
    rep['falsifier'] = dict(delta=dF, interval=B.interval(dF, B.T7), reading=lab_f)
    for arm, tag in TAGS.items():
        rr = {}
        for k, v in labs.items():
            m = G.quoted_means(J, arm, v); dd = [m[d]['mean'] - ref[k][d]['mean'] for d in READ]
            rr[k] = dict(delta=dd, interval=B.interval(dd, B.T7), days_positive=int(sum(x > 0 for x in dd)), reading=B.label_r120(dd))
        rep[f'r190_{arm}'] = rr
        rep['readings'][f'R190_{arm}'] = 'VOID' if lab_f == 'IMPROVES' else rr['combined']['reading']
        rows = []
        for tm in ('J', 'X'):
            for d in DAYS:
                _, a, _ = summary(f'b14r{tm}', d); _, b, _ = summary(f'{tag}{tm}', d)
                rows.append(dict(timing=tm, day=d, **{f'ref_{k}': float(a[k]) for k in ('gross_usd', 'mo1_usd', 'mo60_usd', 'mtm_stop_usd', 'fills', 'shares')},
                                 **{f'arm_{k}': float(b[k]) for k in ('gross_usd', 'mo1_usd', 'mo60_usd', 'mtm_stop_usd', 'fills', 'shares')}))
        T = pd.DataFrame(rows); T.to_csv(f'{OUTD}/replay_days_{arm}.csv', index=False)
        dd = {(k, tm): (T[T.timing == tm][f'arm_{k}'] - T[T.timing == tm][f'ref_{k}']).values for k in ('gross_usd', 'mo1_usd', 'mo60_usd', 'mtm_stop_usd') for tm in ('J', 'X')}
        rep['readings'][f'R191_{arm}'] = B.label_r12x(dd[('gross_usd', 'J')], dd[('gross_usd', 'X')])
        rep['readings'][f'R192_{arm}'] = B.label_r12x(dd[('mo1_usd', 'J')], dd[('mo1_usd', 'X')])
        rep['readings'][f'R193_{arm}'] = B.label_r12x(dd[('mo60_usd', 'J')], dd[('mo60_usd', 'X')])
        rep[f'replay_{arm}'] = {f'{k}_{tm}': dict(delta_sum=float(v.sum()), days_positive=int((v > 0).sum()), interval=B.interval(v, B.T8),
                                                  fresh_1103=float(v[DAYS.index(FRESH)]), per_day=[float(x) for x in v]) for (k, tm), v in dd.items()}
        rep[f'replay_{arm}']['fills'] = {tm: dict(ref=float(T[T.timing == tm].ref_fills.sum()), arm=float(T[T.timing == tm].arm_fills.sum())) for tm in ('J', 'X')}
    io.write_json(f'{OUTD}/report.json', rep)
    L = [f"B16c combined target (see report.json). Readings: {rep['readings']}  falsifier: {lab_f} ({rep['falsifier']['interval'][0]:+.4f})"]
    for arm in TAGS:
        for k in labs:
            x = rep[f'r190_{arm}'][k]
            L.append(f"{arm} R190 {k:8s}: arm - REF mean {x['interval'][0]:+.4f} [{x['interval'][1]:+.4f}, {x['interval'][2]:+.4f}], days > 0 {x['days_positive']}/8 -> {x['reading']}")
        for key in ('gross_usd_J', 'gross_usd_X', 'mo1_usd_J', 'mo1_usd_X', 'mo60_usd_J', 'mo60_usd_X', 'mtm_stop_usd_J', 'mtm_stop_usd_X'):
            y = rep[f'replay_{arm}'][key]
            L.append(f"{arm} {key}: sum {y['delta_sum']:+.2f}, days > 0 {y['days_positive']}/9, 11-03 {y['fresh_1103']:+.2f}")
        L.append(f"{arm} fills: {rep[f'replay_{arm}']['fills']}")
    open(f'{OUTD}/report.txt', 'w').write('\n'.join(L) + '\n'); print('\n'.join(L))


if __name__ == '__main__':
    if not B.selftest():
        sys.exit('REFUSED: B14 selftest failed')
    mode = sys.argv[1]
    if mode in ('build', 'map'):
        (step_build if mode == 'build' else step_map)(sys.argv[2])
    elif mode == 'report':
        step_report()
    else:
        sys.exit(f'unknown mode {mode}')
