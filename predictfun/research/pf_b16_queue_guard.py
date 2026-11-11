"""B16 stage 1 (2026-11-11, USER: "the model may not predict the right action to take ... we haven't built the strong model"): the
guard trained on the right target - what OUR order earns at the back of the queue (step4/badfill_info_tx_tk/grid_ql.parquet from
pf_b16_queue_label.py) - instead of sweeps only. Two arms: 'qa' = the current 32 features (the target alone), 'qb' = + B13a's 13
pressure features (target and features). Everything else is B14's verified pipeline, imported unchanged (pf_b14_pressure_guard: fit,
apply, row matrices, the line rule and the per-minute map onto the frozen lines): the live guard's recipe (pinned LightGBM, threshold =
the 60% quantile of in-sample predictions, LODO on the 8 read days, 11-03 from the fit on all 8), then pf_preds_startknown, then the
frozen v3 engine at J and X on the 9 consumed days. The benchmark REF is the live guard: B14's replays b14rJ / b14rX (which reproduced
the stored v3 replays exactly, C6) and its out-of-fold values on the grid rows (B14's oof_grid.parquet, base_*).

Readings DECLARED BEFORE THE FIRST RUN (R160 / R163: G = 8 read days, t(7); R161 / R162: 9 days):
 R160 decision, per arm against REF: on the grid rows each guard lets quote (its out-of-fold v above its own per-minute 90th
      percentile on the grid rows, bid and ask pooled), the mean NEW label (our order's 1 s value at the back of the queue), arm minus
      REF per day: IMPROVES if the interval is above 0 and >= 6 of 8 days are positive; WORSE if below 0; else NO-CHANGE. VOID (for
      both arms) if the falsifier (arm qb refit on new labels permuted within each day) also reads IMPROVES.
 R163 the same with the OLD label (sweeps only) as the referee, per arm: KEEPS if the interval's low end > -0.005 c/sh; LOSES if its
      high end < 0 and >= 6 of 8 days are negative; else UNCLEAR.
 R161 settle / R162 +1 s, per arm: the v3 replay, arm minus REF per day, B14's rule (BETTER / WORSE / MIXED).
Beside: R160 on the 5 s label; shares filled and the sweep share on the quoted rows; per-day tables; 11-03 alone; fills and shares.
Controls (any failure -> no table): the label's own K1-K3 (its report must exist with K1 all equal); every grid row aligned to its
label and features; the falsifier's permutation keeps each day's label multiset; C2 / C3 of the map (the frozen line rule reproduced;
each arm's map cuts exactly at its own lines; a planted p90 + 0.001 is caught); C5 row alignment; every replay's env as declared (no
PF_R60_FILE); the reading rules on synthetic inputs (B14's selftest).
Label (pf_b16_queue_label.py, decided after its one-day checks on 11-02 and before any guard was fit): a 5-share order RESTING at the
back of the queue at the known best from tau + 22 to tau + 522 ms, filled by rule M (the v3 engine's: trades at its price use up the
size ahead, which every frame caps at the size shown, so a thinning level moves it up; a trade through fills the rest), valued at the
1 s markout; on the 11-02 check it fills 14.6% of rows (11.9% by a sweep), its mean -0.290 against the old label's -0.292 c/sh.
Limits: as B14 (9 consumed days, read before; in-sample for the 13 features; replay fill model not validated, so only differences
under one model are read) and the label's: rule M credits every shrink of the level ahead of us (an upper bound on how fast we move
up; rule L, no credit, is kept beside in the label file); every grid row is an order at the back (an order that rests longer is
nearer the front than this label gives it); the old label's trade times.
Usage: predictfun/research/run_b16.sh (each step through run_light.sh):
  python3 -m predictfun.research.pf_b16_queue_guard build qa | build qb | map qa | map qb | report
"""
import json
import os
import sys

import numpy as np
import pandas as pd

from . import pf_io as io
from . import pf_b14_pressure_guard as B

READ, FRESH, DAYS, S4, S5 = B.READ, B.FRESH, B.DAYS, B.S4, B.S5
QL = f'{S4}/badfill_info_tx_tk/grid_ql.parquet'
QL_REPORT = f'{S4}/badfill_info_tx_tk/grid_ql_report.json'
OUTD = f'{io.OUT}/b16_20261111'
ARMS = {'qa': ('preds_qa60', B.FE), 'qb': ('preds_qb60', B.EMPH)}
TAGS = {'qa': 'b16a', 'qb': 'b16b'}
SEED = 20261111


def load_inputs():
    rq = json.load(open(QL_REPORT))
    if rq.get('k1_rows', 0) < 1000 or rq.get('k1_equal') != rq.get('k1_rows'):
        sys.exit('REFUSED: the label report fails K1')
    G, miss = B.load_grid()
    Q = pd.read_parquet(QL)
    if len(Q) != len(G) or not ((Q.day.values == G.day.values).all() and (Q.mid.values == G.mid.values).all() and (Q.tau.values == G.tau.values).all()):
        sys.exit('REFUSED: labels not aligned with the grid rows')
    return G, Q, miss


def write_preds(stem, d, o, rows):
    src = pd.read_parquet(f'{S5}/{d}/preds_t60.parquet', columns=['mid', 'tau', 'pred_1000', 'pred_250'])
    if not ((src.mid.values == rows.mid.values).all() and (src.tau.values == rows.tau.values).all()):
        sys.exit(f'REFUSED: C5 {d} rows differ from preds_t60')
    src['v_bid'] = o['bid']; src['v_ask'] = o['ask']
    src.to_parquet(f'{S5}/{d}/{stem}.parquet', index=False)


def step_build(arm):
    stem, feats = ARMS[arm]
    for d in DAYS:
        for st_ in (stem, stem + 'k', stem + 'km'):
            if os.path.exists(f'{S5}/{d}/{st_}.parquet'):
                sys.exit(f'REFUSED: {S5}/{d}/{st_}.parquet exists')
    os.makedirs(OUTD, exist_ok=True)
    G, Q, miss = load_inputs(); io.mem_check('inputs')
    nan_lab = {s: int(np.isnan(Q[f'ql_{s}'].values).sum()) for s in ('bid', 'ask')}
    Y = {s: np.nan_to_num(Q[f'ql_{s}'].values, nan=0.0).clip(-10, 10) for s in ('bid', 'ask')}
    del Q
    X = np.concatenate([G[B.FE].values.astype(np.float32)] + ([G[B.EXTRA].values.astype(np.float32)] if feats is B.EMPH else []), axis=1)
    if X.shape[1] != len(feats):
        sys.exit(f'REFUSED: training matrix {X.shape[1]} wide, expected {len(feats)}')
    day = G.day.values
    oof = pd.DataFrame(dict(day=G.day.values, mid=G.mid.values, tau=G.tau.values, t_rel=G.t_rel.values))
    del G
    for c in (f'{arm}_bid', f'{arm}_ask', f'{arm}perm_bid', f'{arm}perm_ask'):
        oof[c] = np.float32(np.nan)
    rep = dict(arm=arm, stem=stem, features=len(feats), grid_rows=int(len(X)), label_nan_set_to_0=nan_lab, grid_rows_without_any_engineered_value=miss,
               thresholds={}, falsifier=None)
    YP = None
    if arm == 'qb':
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
        write_preds(stem, d, B.apply(ms, cs, live, XD), rows); del ms, XD
        if YP is not None:
            mp, cp = B.fit(X[tr], {s: YP[s][tr] for s in YP})
            for s in ('bid', 'ask'):
                oof.loc[te, f'{arm}perm_{s}'] = (mp[s].predict(X[te], num_threads=1) - cp[s]).astype(np.float32)
            del mp
        io.mem_check(f'{arm} fold {d}'); print(arm, 'fold', d, cs, flush=True)
    ms, cs = B.fit(X, Y); rep['thresholds'][FRESH] = cs
    rows, live, XD = B.day_matrix(FRESH, feats)
    write_preds(stem, FRESH, B.apply(ms, cs, live, XD), rows)
    for s in ('bid', 'ask'):
        ms[s].save_model(f'{OUTD}/guard_{arm}_{s}.txt')
    oof.to_parquet(f'{OUTD}/oof_{arm}.parquet', index=False)
    io.write_json(f'{OUTD}/build_{arm}_report.json', rep)
    print(json.dumps({k: rep[k] for k in ('arm', 'grid_rows', 'label_nan_set_to_0', 'falsifier')}), flush=True)


def step_map(arm):
    stem = ARMS[arm][0]
    L = json.load(open(B.LINES_FILE))['lines']; L90 = np.array(L['90'], float); L95 = np.array(L['95'], float)
    rep = dict(arm=arm, controls={})
    t90, t95 = B.per_minute_q('preds_t60k', READ)
    eq = bool(np.allclose(np.round(t90, 4), L90, atol=1e-12) and np.allclose(np.round(t95, 4), L95, atol=1e-12))
    rep['controls']['C2_lines'] = dict(equal=eq)
    if not eq:
        sys.exit('REFUSED: C2 lines differ')
    q90, q95 = B.per_minute_q(stem + 'k', READ)
    rep['q90'] = [float(x) for x in q90]; rep['q95'] = [float(x) for x in q95]
    if not np.all(q95 > q90):
        sys.exit('REFUSED: p95 not above p90 in some minute')
    c3 = {}
    for d in DAYS:
        out = f'{S5}/{d}/{stem}km.parquet'
        if os.path.exists(out):
            sys.exit(f'REFUSED: {out} exists')
        pk = pd.read_parquet(f'{S5}/{d}/{stem}k.parquet'); p0 = pd.read_parquet(f'{S5}/{d}/{stem}.parquet', columns=['v_bid', 'v_ask'])
        mn = B.minute_of(B.rel_of(d, pk.mid.values, pk.tau.values)); ok_all = True; fal_all = True
        for s in ('bid', 'ask'):
            v = pk[f'v_{s}'].values.astype(np.float32)
            keep = (v == np.float32(-1.0)) & (p0[f'v_{s}'].values != np.float32(-1.0))
            w = B.map_values(v, mn, keep, q90, q95, L90, L95)
            ok_all = ok_all and B.mapping_ok(v, w, mn, keep, q90, q95, L90, L95)
            fal_all = fal_all and not B.mapping_ok(v, B.map_values(v, mn, keep, q90 + 0.001, q95, L90, L95), mn, keep, q90, q95, L90, L95)
            pk[f'v_{s}'] = w
        c3[d] = dict(ok=bool(ok_all), falsifier_breaks=bool(fal_all))
        if not (ok_all and fal_all):
            rep['controls']['C3_mapping'] = c3; io.write_json(f'{OUTD}/map_{arm}_report.json', rep); sys.exit(f'REFUSED: C3 {d}')
        pk.to_parquet(out, index=False)
    rep['controls']['C3_mapping'] = c3
    io.write_json(f'{OUTD}/map_{arm}_report.json', rep); print(json.dumps(rep), flush=True)


def summary(tag, d):
    from .pf_inv_report import name
    from .pf_ladder_v3_build import ARM
    stem = {'b14r': 'preds_t60k', 'b16a': 'preds_qa60km', 'b16b': 'preds_qb60km'}[tag[:4]]
    s = json.load(open(f'{S4}/ladder_tokyoreal_custom_e2_{stem}_{tag}/{d}/summary.json'))
    return s, s['arms'][name(ARM)], stem


def quoted_means(J, guard, label):
    """Per read day: mean label on the grid rows the guard lets quote (bid + ask pooled), with filled shares and sweep share."""
    mn = B.minute_of(J.t_rel.values); q = B.grid_q90(J[f'{guard}_bid'].values, J[f'{guard}_ask'].values, mn); out = {}
    for d in READ:
        s_ = (J.day == d).values; ys, fs, ws = [], [], []
        for side in ('bid', 'ask'):
            on = B.gate_on(J[f'{guard}_{side}'].values[s_], mn[s_], q)
            ys.append(J[f'{label}_{side}'].values[s_][on]); fs.append(J[f'qf_{side}'].values[s_][on]); ws.append(J[f'qs_{side}'].values[s_][on])
        y = np.concatenate(ys); y = y[np.isfinite(y)]; f = np.concatenate(fs); w = np.concatenate(ws)
        out[d] = dict(mean=float(y.mean()) if len(y) else float('nan'), n=int(len(y)), filled_share=float(np.nanmean(f > 0)), sweep_share=float(np.mean(w)))
    return out


def label_r163(d):
    d = np.asarray(d, float)
    if len(d) != 8 or not np.all(np.isfinite(d)):
        return 'n/a'
    _, lo, hi = B.interval(d, B.T7)
    if lo > -0.005:
        return 'KEEPS'
    if hi < 0 and int((d < 0).sum()) >= 6:
        return 'LOSES'
    return 'UNCLEAR'


def step_report():
    from .pf_ladder_v3_build import ARM
    if not (label_r163([0.01] * 8) == 'KEEPS' and label_r163([-0.05] * 8) == 'LOSES' and label_r163([0.02, -0.05] * 4) == 'UNCLEAR'):
        sys.exit('REFUSED: R163 rule on synthetic inputs')
    rep = dict(arm=ARM, readings={}, controls={})
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
    G = pd.read_parquet(f'{B.OUTD}/oof_grid.parquet')                      # B14: labels (old), base_* (REF out of fold)
    Q = pd.read_parquet(QL).rename(columns={'qa_bid': 'ahead_bid', 'qa_ask': 'ahead_ask'})   # the label's size-ahead columns; arm 'qa' owns qa_*
    J = G.merge(Q, on=['day', 'mid', 'tau'], how='left', validate='one_to_one')
    for arm in TAGS:
        A = pd.read_parquet(f'{OUTD}/oof_{arm}.parquet', columns=['day', 'mid', 'tau', f'{arm}_bid', f'{arm}_ask', f'{arm}perm_bid', f'{arm}perm_ask'])
        J = J.merge(A, on=['day', 'mid', 'tau'], how='left', validate='one_to_one')
    if J[['qa_bid', 'qb_bid', 'base_bid', 'ql_bid']].isna().all(axis=0).any():
        sys.exit('REFUSED: a guard or label column is empty after the joins')
    ref_new = quoted_means(J, 'base', 'ql'); ref_old = quoted_means(J, 'base', 'label'); ref_new5 = quoted_means(J, 'base', 'ql5')
    fal = quoted_means(J, 'qbperm', 'ql'); dF = [fal[d]['mean'] - ref_new[d]['mean'] for d in READ]; lab_f = B.label_r120(dF)
    rep['falsifier'] = dict(delta=dF, interval=B.interval(dF, B.T7), reading=lab_f)
    rep['ref'] = dict(new=ref_new, old=ref_old, new5=ref_new5)
    for arm, tag in TAGS.items():
        n_ = quoted_means(J, arm, 'ql'); o_ = quoted_means(J, arm, 'label'); n5 = quoted_means(J, arm, 'ql5')
        dN = [n_[d]['mean'] - ref_new[d]['mean'] for d in READ]; dO = [o_[d]['mean'] - ref_old[d]['mean'] for d in READ]
        dN5 = [n5[d]['mean'] - ref_new5[d]['mean'] for d in READ]
        r160 = 'VOID' if lab_f == 'IMPROVES' else B.label_r120(dN)
        rep['readings'][f'R160_{arm}'] = r160; rep['readings'][f'R163_{arm}'] = label_r163(dO)
        rep[f'r160_{arm}'] = dict(delta_new=dN, interval_new=B.interval(dN, B.T7), delta_old=dO, interval_old=B.interval(dO, B.T7),
                                  delta_new5=dN5, interval_new5=B.interval(dN5, B.T7), per_day_new=n_, per_day_old=o_)
        rows = []
        for tm in ('J', 'X'):
            for d in DAYS:
                _, a, _ = summary(f'b14r{tm}', d); _, b, _ = summary(f'{tag}{tm}', d)
                rows.append(dict(timing=tm, day=d, **{f'ref_{k}': float(a[k]) for k in ('gross_usd', 'mo1_usd', 'mo60_usd', 'fills', 'shares')},
                                 **{f'arm_{k}': float(b[k]) for k in ('gross_usd', 'mo1_usd', 'mo60_usd', 'fills', 'shares')}))
        T = pd.DataFrame(rows); T.to_csv(f'{OUTD}/replay_days_{arm}.csv', index=False)
        dd = {(k, tm): (T[T.timing == tm][f'arm_{k}'] - T[T.timing == tm][f'ref_{k}']).values for k in ('gross_usd', 'mo1_usd', 'mo60_usd') for tm in ('J', 'X')}
        rep['readings'][f'R161_{arm}'] = B.label_r12x(dd[('gross_usd', 'J')], dd[('gross_usd', 'X')])
        rep['readings'][f'R162_{arm}'] = B.label_r12x(dd[('mo1_usd', 'J')], dd[('mo1_usd', 'X')])
        rep[f'replay_{arm}'] = {f'{k}_{tm}': dict(delta_sum=float(v.sum()), days_positive=int((v > 0).sum()), interval=B.interval(v, B.T8),
                                                  fresh_1103=float(v[DAYS.index(FRESH)]), per_day=[float(x) for x in v]) for (k, tm), v in dd.items()}
        rep[f'replay_{arm}']['fills'] = {tm: dict(ref=float(T[T.timing == tm].ref_fills.sum()), arm=float(T[T.timing == tm].arm_fills.sum()),
                                                  ref_shares=float(T[T.timing == tm].ref_shares.sum()), arm_shares=float(T[T.timing == tm].arm_shares.sum())) for tm in ('J', 'X')}
    io.write_json(f'{OUTD}/report.json', rep)
    L = [f"B16 stage 1 (see report.json). Readings: {rep['readings']}  falsifier: {lab_f}"]
    for arm in TAGS:
        x = rep[f'r160_{arm}']
        L.append(f"{arm} R160 new label, arm - REF by day: " + ', '.join(f'{d[4:]} {v:+.4f}' for d, v in zip(READ, x['delta_new'])) +
                 f" | mean {x['interval_new'][0]:+.4f} [{x['interval_new'][1]:+.4f}, {x['interval_new'][2]:+.4f}]")
        L.append(f"{arm} R163 old label: mean {x['interval_old'][0]:+.4f} [{x['interval_old'][1]:+.4f}, {x['interval_old'][2]:+.4f}]; "
                 f"5 s new label: mean {x['interval_new5'][0]:+.4f} [{x['interval_new5'][1]:+.4f}, {x['interval_new5'][2]:+.4f}]")
        for key in ('gross_usd_J', 'gross_usd_X', 'mo1_usd_J', 'mo1_usd_X', 'mo60_usd_J', 'mo60_usd_X'):
            y = rep[f'replay_{arm}'][key]
            L.append(f"{arm} {key}: sum {y['delta_sum']:+.2f}, days > 0 {y['days_positive']}/9, 11-03 {y['fresh_1103']:+.2f}")
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
