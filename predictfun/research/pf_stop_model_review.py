"""STOP-MODEL INPUT REVIEW (2026-11-15, session aea5bfb9): inputs, target, data coverage and the inputs' influence on the target - BEFORE the
build.

USER (2026-11-15): "Before model play, can we first review the model inputs, training target, data coverage, etc, then check the model inputs
influence on our target, check their influence first, then proceed to build the model".

Decisions use the development days 10-23..26 only (Rule 11). Coverage is printed for all nine days as counts (no input or target statistic
of a test day). The code-level catalogue of the inputs (definitions, clocks, sign conventions) is a separate independent code review.
TARGET reviewed = v4's (diag 2026-11-15, target U): every usable row, y^H_j = [D1_H >= j] - the side's best moves >= j ticks against the
side beyond the known best in the state at t + 22 ms or a frame in (t + 22, t + H], H = 72 / 250 / 1000 ms, levels j = 1..6.
PARTS (declared before the run):
 A COVERAGE (9 days): registered / used windows, rows, statuses (invalid book, GAP, masked, boundary windows), window-sides below 1,200 rows,
   UTC hours without rows, untimed legs in the first 60 s.
 B TARGET (dev days): rates per (H, j) per day and side; by time in window; the share of rows whose move inside the cancel latency reverted
   (D0 >= j but D1_H < j).
 C INPUTS (dev days): group, missing share per day, constant flag, p1 / p50 / p99, the largest daily-median drift in pooled-sd units;
   near-duplicate pairs (|Pearson r| > 0.99 on 200,000 rows of 10-26).
 D UNIVARIATE INFLUENCE (dev days; heads H72 j1, H250 j1, H1000 j1, H1000 j3): per input the pooled AUC and its sign, the AUC per day, and
   the shape dispersion across days (rate of y by pooled input decile minus the day's base rate, sd across days, pp).
 E MODEL INFLUENCE (v4's fold-10-27 models: trained on 10-23..25, target U, 180k rows a day, 200 rounds; evaluated on 10-26 = a dev day;
   heads j = 1..3 at every H): TreeSHAP on 30,000 rows - mean |SHAP| per input and group, its direction (Spearman of the input against its
   SHAP value) and the mean SHAP by input decile for the top inputs; PERMUTATION importance (log loss increase when the input / the whole
   group is shuffled across the 30,000 rows); GROUP ABLATION (refit without the group, 60k rows a day, 200 rounds, the 9 heads) -> 10-26
   log loss change on every usable row.
 RULES (input changes the build applies): DROP an input iff (i) |r| > 0.999 with an earlier input of FEAT_M, or (ii) its per-dev-day AUC
   flips sign (> 0.52 on one day and < 0.48 on another at the same head) AND its permutation importance is <= 0 at every one of the 9 heads;
   DROP a group iff its ablation lowers the 10-26 log loss on >= 7 of the 9 heads with a mean gain >= 0.1%. FLAGS (printed, no automatic
   change): an input whose SHAP direction contradicts its EXPECTED sign at >= 2 of the 3 j1..j3 heads of some H (EXPECT: adverse Binance /
   Predict / fair-price moves raise the risk; a thicker own best level and more own-side size lower it; sigma raises it).
OUTPUT review_20261115.json / .txt; 'drop_inputs' is read by the v4 build. Controls: pf_stop_model's selftests (planted errors); group
membership covers every input exactly once; the SHAP rows sum to the raw score (TreeSHAP additivity, max error <= 1e-4).
Usage (run_light, PF_SLOT=claude): pf_stop_model_review.py selftest | run | reg (the regularization review, after run) | s5check (input-pipeline checks) | dup (within-side copies)
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
import pf_stop_model_diag as DG                           # noqa: E402

F1_TEST = M.TESTS[0]; F1_TRAIN = list(M.train_days(F1_TEST)); EVAL = M.FOLDS[F1_TEST][1]
UHEADS = [(72, 1), (250, 1), (1000, 1), (1000, 3)]
MHEADS = [(h, j) for h in M.HS for j in (1, 2, 3)]
LABC = ['D0'] + [M.DCOL[h] for h in M.HS]
N_SHAP = 30000; N_COR = 200000; TOP_N = 25; DUP_R = 0.99; DUP_DROP = 0.999; FLIP_HI, FLIP_LO = 0.52, 0.48
ABL_HEADS = 7; ABL_GAIN = 0.001
OUT_J = f'{M.OUTD}/review_20261115.json'; OUT_T = f'{M.OUTD}/review_20261115.txt'
FAIR = ('dfv250', 'dfv1000', 'gap_adv', 'sens', 'pbn_dist')
TIMEV = ('t_rel', 'tau_rem', 'sigma', 'fside')
PFDYN = ('pf_age', 'pf_nfr1000', 'pf_dm250', 'pf_dm1000', 'pf_dm5000', 'adv_pf_dm250', 'adv_pf_dm1000', 'adv_pf_dm5000', 'own_d250', 'opp_d250',
         'own_mv250', 'own_mv1000', 'opp_mv250', 'opp_mv1000', 'own_lvl_age', 'opp_lvl_age', 'act_rel')
EXPECT = {**{f'adv_bn_r{x}': 1 for x in (25, 50, 100, 250, 500, 1000, 3000)}, **{f'adv_pf_dm{x}': 1 for x in (250, 1000, 5000)},
          'adv_bn_flow250': 1, 'adv_bn_flow1000': 1, 'adv_d_bnimb250': 1, 'dfv250': 1, 'dfv1000': 1, 'gap_adv': 1, 'sigma': 1,
          'r_s1': -1, 'r_own_l1': -1}


def group_of(c):
    if c.startswith(('bn_', 'adv_bn_')) or c in ('d_bnimb250', 'adv_d_bnimb250'):
        return 'BINANCE'
    if c in FAIR:
        return 'FAIR_PRICE'
    if c.startswith('r_'):
        return 'BOOK_SIZES'
    if c in TIMEV:
        return 'TIME_VOL_SIDE'
    if c in PFDYN:
        return 'PREDICT_DYNAMICS'
    return 'PREDICT_BOOK_STATE'


def spearman(a, b):
    f = np.isfinite(a) & np.isfinite(b)
    if f.sum() < 100:
        return M.NAN
    from scipy.stats import spearmanr
    return float(spearmanr(a[f], b[f])[0])


def selftest():
    bad = []
    G = {c: group_of(c) for c in M.FEAT_M}
    if len(G) != len(M.FEAT_M) or sorted(set(G.values())) != sorted(['BINANCE', 'FAIR_PRICE', 'BOOK_SIZES', 'TIME_VOL_SIDE', 'PREDICT_DYNAMICS',
                                                                         'PREDICT_BOOK_STATE']):
        bad.append(f'groups {sorted(set(G.values()))}')
    if not all(c in M.FEAT_M for c in list(EXPECT) + list(FAIR) + list(TIMEV) + list(PFDYN)):
        bad.append('a named input is not a model input')
    rg = np.random.default_rng(3); x = rg.normal(0, 1, 5000)
    if not (spearman(x, 2 * x) > 0.99 and spearman(x, -x) < -0.99):
        bad.append('spearman')
    return bad


def run():
    M.all_selftests()
    b = selftest()
    if b:
        sys.exit(f'REVIEW SELFTEST FAILED: {b}')
    import lightgbm as lgb
    T0 = time.time(); asof = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()); L = []; res = dict(asof=asof, version='review 2026-11-15')
    A = json.load(open(f'{M.OUTD}/audit2.json')); scale = A['scale']; F = [c for c in M.FEAT_M if c not in set(A['drop'])]
    G = {c: group_of(c) for c in F}; groups = sorted(set(G.values()))
    L.append(f'STOP-MODEL INPUT REVIEW (decisions on dev days {", ".join(M.DEV_DAYS)} only); as of {asof}; {len(F)} inputs, scale {scale}')
    L.append('INPUT GROUPS: ' + '; '.join(f'{g} ({sum(1 for c in F if G[c] == g)}): ' + ', '.join(c for c in F if G[c] == g) for g in groups))
    # ---- A coverage (counts, every day)
    cov = {}
    for d in M.DAYS9:
        rj = json.load(open(f'{M.OUTD}/rows_{d}.json')); ls = rj['load_stat']
        kc = M.day_cols(d, ['m', 'side', 't']); per = pd.Series(1, index=pd.MultiIndex.from_arrays([kc['m'], kc['side']])).groupby(level=[0, 1]).sum()
        hrs = np.bincount(((kc['t'] // 3600000) % 24).astype(int), minlength=24)
        cov[d] = dict(registry=ls['registry'], windows=ls['windows'], window_sides=rj['window_sides'], steps=rj['steps'], rows=rj['rows'],
                      invalid_book=rj['invalid_book'], gap_rows=rj['gap_rows'], masked=rj['masked'], boundary_windows=rj['boundary_windows'],
                      sides_short=int((per < 1200).sum()), min_rows_side=int(per.min()), hours_without_rows=[int(h) for h in np.flatnonzero(hrs == 0)],
                      untimed_legs_0_60=ls['untimed_legs_0_60'])
    res['A_coverage'] = cov
    L.append('A COVERAGE (counts): day | windows used / registered | window-sides | rows | invalid book | GAP | masked | boundary windows | '
             'sides < 1,200 rows (min) | UTC hours without rows | untimed legs 0-60 s')
    for d, x in cov.items():
        L.append(f'   {d}: {x["windows"]}/{x["registry"]} | {x["window_sides"]} | {x["rows"]} | {x["invalid_book"]} | {x["gap_rows"]} | {x["masked"]} | '
                 f'{x["boundary_windows"]} | {x["sides_short"]} ({x["min_rows_side"]}) | {x["hours_without_rows"] or "none"} | {x["untimed_legs_0_60"]}')
    # ---- B target (dev days) + labels kept for D
    LBD = {}; tgt = {}
    for d in M.DEV_DAYS:
        sel = M.day_usable(d); lb = M.day_cols(d, LABC + ['side', 'k'], sel); LBD[d] = lb
        t_ = {}
        for h in M.HS:
            for j in M.JS:
                y = lb[M.DCOL[h]] >= j
                t_[f'H{h}_j{j}'] = dict(rate=float(y.mean()), bid=float(y[lb['side'] == 0].mean()), ask=float(y[lb['side'] == 1].mean()),
                                         reverted=float(((lb['D0'] >= j) & ~y).mean()),
                                         by_time=[float(y[(lb['k'] >= a) & (lb['k'] < b_)].mean()) for a, b_ in DG.TB])
        tgt[d] = t_
    res['B_target'] = tgt
    L.append('B TARGET (v4 target U, usable rows): rate of y^H_j per dev day (bid / ask on 10-26) | rate by time in window 0-5 / 5-15 / 15-30 / 30-60 s '
             '(10-26) | reverted share (D0 >= j, D1_H < j; 10-26)')
    for h in M.HS:
        for j in M.JS:
            k_ = f'H{h}_j{j}'; e = tgt[EVAL][k_]
            L.append(f'   {k_:9s} ' + ' '.join(f'{tgt[d][k_]["rate"]:.4f}' for d in M.DEV_DAYS) + f' ({e["bid"]:.4f} / {e["ask"]:.4f}) | '
                     + ' / '.join(f'{x:.4f}' for x in e['by_time']) + f' | {e["reverted"]:.4%}')
    # ---- C inputs and D univariate influence (one input at a time over the four dev days)
    sels = {d: M.day_usable(d) for d in M.DEV_DAYS}; inp = {}; uni = {}
    for c in F:
        xs = {d: M.day_matrix(d, scale, [c], sels[d])[:, 0] for d in M.DEV_DAYS}
        allx = np.concatenate(list(xs.values())); fin = np.isfinite(allx)
        q = np.quantile(allx[fin], [0.01, 0.5, 0.99]) if fin.any() else [M.NAN] * 3; sd = float(np.nanstd(allx)) if fin.any() else M.NAN
        med = {d: float(np.nanmedian(x)) if np.isfinite(x).any() else M.NAN for d, x in xs.items()}
        inp[c] = dict(group=G[c], missing={d: float((~np.isfinite(x)).mean()) for d, x in xs.items()}, constant=bool(fin.any() and np.nanmin(allx) == np.nanmax(allx)),
                      p01=float(q[0]), p50=float(q[1]), p99=float(q[2]),
                      median_drift_sd=float(max(abs(m_ - float(q[1])) for m_ in med.values()) / sd) if sd and sd > 0 else M.NAN)
        u = {}
        for (h, j) in UHEADS:
            ys = {d: LBD[d][M.DCOL[h]] >= j for d in M.DEV_DAYS}
            ap = M.auc_nan(np.concatenate(list(ys.values())), allx)[0]
            u[f'H{h}_j{j}'] = dict(auc=ap, auc_days={d: M.auc_nan(ys[d], xs[d])[0] for d in M.DEV_DAYS},
                                   shape_pp=100.0 * M.shape_disp([xs[d] for d in M.DEV_DAYS], [ys[d] for d in M.DEV_DAYS]))
        uni[c] = u; del xs, allx
        M.mem_guard(f'review input {c}')
    res['C_inputs'] = inp; res['D_univariate'] = uni
    flips = {c: [hk for hk, v in uni[c].items() if max(v['auc_days'].values()) > FLIP_HI and min(v['auc_days'].values()) < FLIP_LO] for c in F}
    flips = {c: v for c, v in flips.items() if v}
    # near-duplicates on 10-26
    rg = np.random.default_rng(int(EVAL)); s26 = sels[EVAL]; isub = np.sort(rg.choice(len(s26), min(len(s26), N_COR), replace=False))
    Xc = M.day_matrix(EVAL, scale, F, s26[isub]); C_ = pd.DataFrame(Xc, columns=F).corr(min_periods=1000).values; del Xc
    dups = [(F[a], F[b_], float(C_[a, b_])) for a in range(len(F)) for b_ in range(a + 1, len(F)) if np.isfinite(C_[a, b_]) and abs(C_[a, b_]) > DUP_R]
    res['C_near_duplicates'] = dups
    L.append('C INPUTS (dev days): input | group | missing % per day | p1 / p50 / p99 | max daily-median drift (sd) || D UNIVARIATE: AUC pooled '
             '(per day) and shape dispersion pp for H72 j1 | H250 j1 | H1000 j1 | H1000 j3')
    for c in F:
        x = inp[c]
        L.append(f'   {c:16s} {G[c][:12]:12s} ' + '/'.join(f'{100 * v:.1f}' for v in x['missing'].values()) + f' | {x["p01"]:.4g} / {x["p50"]:.4g} / {x["p99"]:.4g} | '
                 f'{x["median_drift_sd"]:.2f}' + (' CONST' if x['constant'] else '') + ' || '
                 + ' | '.join(f'{v["auc"]:.3f} (' + ' '.join(f'{a:.2f}' for a in v['auc_days'].values()) + f') {v["shape_pp"]:.1f}' for v in uni[c].values()))
    L.append(f'   near-duplicate pairs (|r| > {DUP_R}, 10-26): ' + ('; '.join(f'{a} ~ {b_} {r:+.4f}' for a, b_, r in dups) or 'none'))
    L.append(f'   per-day AUC sign flips (> {FLIP_HI} and < {FLIP_LO} at one head): ' + ('; '.join(f'{c} {v}' for c, v in flips.items()) or 'none'))
    del LBD
    M.mem_guard('review C/D')
    # ---- E model influence: v4 fold-10-27 models on 10-26
    fj = json.load(open(f'{M.OUTD}/fit_{F1_TEST}_v4.json'))
    if fj['version'] != '4' or fj['features'] != F or fj['decisions']['target'] != 'U':
        sys.exit('REFUSED: the v4 fold-10-27 fit is not the reviewed model (version / inputs / target)')
    ish = np.sort(rg.choice(len(s26), N_SHAP, replace=False)); XS = M.day_matrix(EVAL, scale, F, s26[ish])
    lbS = {c: v.astype(np.float32) for c, v in M.day_cols(EVAL, LABC, s26[ish]).items()}
    shap = {}; perm = {}; perm_g = {}; sdec = {}; add_err = 0.0
    for (h, j) in MHEADS:
        bst = lgb.Booster(model_file=f'{M.OUTD}/models/fold{F1_TEST}_H{h}_j{j}_v4.txt'); y = lbS[M.DCOL[h]] >= j
        S = bst.predict(XS, pred_contrib=True); raw = bst.predict(XS, raw_score=True); add_err = max(add_err, float(np.abs(S.sum(1) - raw).max()))
        p0 = bst.predict(XS); ll0 = M.logloss(y, p0); ma = np.abs(S[:, :-1]).mean(0)
        hk = f'H{h}_j{j}'
        shap[hk] = {c: dict(mean_abs=float(ma[i]), share=float(ma[i] / ma.sum()), direction=spearman(XS[:, i].astype(float), S[:, i])) for i, c in enumerate(F)}
        top = [F[i] for i in np.argsort(-ma)[:TOP_N]]; cand = list(dict.fromkeys(top + list(flips)))
        rp = np.random.default_rng(7 + h + j)
        pr = {}
        for c in cand:
            Xp = XS.copy(); i = F.index(c); Xp[:, i] = Xp[rp.permutation(N_SHAP), i]; pr[c] = M.logloss(y, bst.predict(Xp)) - ll0; del Xp
        perm[hk] = pr; pg = {}
        for g in groups:
            Xp = XS.copy(); ii = [F.index(c) for c in F if G[c] == g]; Xp[:, ii] = Xp[rp.permutation(N_SHAP)][:, ii]
            pg[g] = M.logloss(y, bst.predict(Xp)) - ll0; del Xp
        perm_g[hk] = pg
        if j == 1:
            dd = {}
            for c in top[:8]:
                i = F.index(c); xv = XS[:, i].astype(float); f = np.isfinite(xv)
                e = np.unique(np.quantile(xv[f], np.linspace(0.1, 0.9, 9))) if f.sum() > 100 else []
                bb = np.searchsorted(e, xv[f], side='right')
                dd[c] = [float(S[f, i][bb == q_].mean()) if (bb == q_).any() else M.NAN for q_ in range(len(e) + 1)] + ([float(S[~f, i].mean())] if (~f).any() else [])
            sdec[hk] = dd
        del S
        M.mem_guard(f'review shap {hk}')
    if add_err > 1e-4:
        sys.exit(f'REFUSED: TreeSHAP additivity error {add_err}')
    res['E_shap'] = shap; res['E_perm'] = perm; res['E_perm_group'] = perm_g; res['E_shap_deciles'] = sdec; res['E_additivity_err'] = add_err
    del XS
    # group ablation refits (V60, the 9 heads)
    s1 = lambda d: int(F1_TEST) * 31 + int(d)                                    # noqa: E731
    h1 = lambda h, j: int(F1_TEST) * 7 + j * 1009 + h                             # noqa: E731
    T60 = DG.train_table(F1_TRAIN, s1, 'V60', None, F, scale)
    X26 = M.day_matrix(EVAL, scale, F, s26); lab26 = {c: v.astype(np.float32) for c, v in M.day_cols(EVAL, LABC, s26).items()}
    abl = {}
    for g in ['ALL'] + groups:
        keep = [i for i, c in enumerate(F) if g == 'ALL' or G[c] != g]; Fk = [F[i] for i in keep]
        Tk = (np.ascontiguousarray(T60[0][:, keep]), T60[1], T60[2], T60[3])
        Bk = DG.train_heads(Tk, Fk, 'U', 'V60', h1, MHEADS); Xk = np.ascontiguousarray(X26[:, keep]) if g != 'ALL' else X26
        Pk = DG.predict(Bk, Xk, MHEADS); abl[g] = DG.metrics(Pk, lab26, 'U', MHEADS); del Tk, Bk, Pk
        if g != 'ALL':
            del Xk
        M.mem_guard(f'review ablation {g}')
    res['E_ablation'] = abl
    base = abl['ALL']; drop_g = []; abl_s = {}
    for g in groups:
        rel = [(base[f'H{h}_j{j}']['ll'] - abl[g][f'H{h}_j{j}']['ll']) / base[f'H{h}_j{j}']['ll'] for (h, j) in MHEADS]   # > 0: better without it
        nb = int(sum(r > 0 for r in rel)); abl_s[g] = dict(mean_gain_without=float(np.mean(rel)), heads_better_without=nb)
        if nb >= ABL_HEADS and np.mean(rel) >= ABL_GAIN:
            drop_g.append(g)
    # ---- rules
    drop_i = []
    for a, b_, r in dups:
        if abs(r) > DUP_DROP and b_ not in drop_i and a not in drop_i:
            drop_i.append(b_)
    for c in flips:
        if all(perm[hk].get(c, 0.0) <= 0.0 for hk in perm) and c not in drop_i:
            drop_i.append(c)
    flags = []
    for c, sgn in EXPECT.items():
        for h in M.HS:
            n_bad = sum(1 for j in (1, 2, 3) if np.isfinite(shap[f'H{h}_j{j}'][c]['direction']) and np.sign(shap[f'H{h}_j{j}'][c]['direction']) == -sgn
                        and abs(shap[f'H{h}_j{j}'][c]['direction']) > 0.05)
            if n_bad >= 2:
                flags.append(f'{c} at H{h} (expected {"+" if sgn > 0 else "-"})')
    res.update(drop_inputs=drop_i, drop_groups=drop_g, ablation_summary=abl_s, flags=flags, sign_flips=flips, seconds=round(time.time() - T0, 1))
    L.append(f'E MODEL INFLUENCE (v4 fold-10-27 models, 10-26, {N_SHAP} rows; TreeSHAP additivity max error {add_err:.2e})')
    L.append('   group | SHAP share H72 j1 / H250 j1 / H1000 j1 | permutation log-loss increase (x1e3) H72 j1 / H250 j1 / H1000 j1 | ablation (refit without): '
             'mean log-loss change, heads better without')
    for g in groups:
        sh = [sum(v['share'] for c, v in shap[f'H{h}_j1'].items() if G[c] == g) for h in M.HS]
        L.append(f'   {g:18s} ' + ' / '.join(f'{x:.1%}' for x in sh) + ' | ' + ' / '.join(f'{1e3 * perm_g[f"H{h}_j1"][g]:+.2f}' for h in M.HS)
                 + f' | {-abl_s[g]["mean_gain_without"]:+.2%} worse without, {abl_s[g]["heads_better_without"]}/9 better without')
    for h in M.HS:
        hk = f'H{h}_j1'; top = sorted(shap[hk], key=lambda c: -shap[hk][c]['mean_abs'])[:15]
        L.append(f'   top inputs {hk} (share, direction, permutation x1e3): ' + '; '.join(f'{c} {shap[hk][c]["share"]:.1%} {shap[hk][c]["direction"]:+.2f} '
                                                                               f'{1e3 * perm[hk].get(c, M.NAN):+.2f}' for c in top))
        L.append(f'   mean SHAP by input decile {hk} (low -> high; last = NaN rows): ' + '; '.join(f'{c}: ' + ' '.join(f'{v:+.3f}' for v in vv)
                                                                                                  for c, vv in sdec[hk].items()))
    L.append(f'FLAGS (SHAP direction contradicts the expected sign at >= 2 of j1..j3): {flags or "none"}')
    L.append(f'DECISIONS (declared rules): drop inputs {drop_i or "none"}; drop groups {drop_g or "none"}; {res["seconds"]} s')
    json.dump(res, open(OUT_J, 'w'), indent=1, default=float); open(OUT_T, 'w').write('\n'.join(L) + '\n')
    print('\n'.join(L), flush=True)


NORM_DRIFT = 0.5; NORM_CUT = 0.25; NORM_HEADS = 3
REG_VARIANTS = {'MDL1000': {'min_data_in_leaf': 1000}, 'L2_10': {'lambda_l2': 10.0}, 'LEAF15': {'num_leaves': 15}}
OUT_RJ = f'{M.OUTD}/reg_20261115.json'; OUT_RT = f'{M.OUTD}/reg_20261115.txt'


def reg():
    """REGULARIZATION REVIEW (USER 2026-11-15: "And check if any inputs need regularization etc"), after 'run' (its decisions), dev days only.
    R1 INPUT CONDITIONING: candidates = inputs the audit v2 flagged (PSI >= 0.10 on >= 2 dev days) or whose daily median drifts > 0.5 pooled sd
       (review C), except the book sizes (already relative to W30), time / side inputs and dropped inputs. For each, the causal normalization
       N1 = x / its trailing day-level scale (M.trail_scale: median of the previous <= 12 windows' median |x|, >= 3 windows) against raw: the
       shape dispersion across the dev days (rate of y by pooled decile minus the day's base rate) at H72 j1, H250 j1, H1000 j1, H1000 j3.
       SELECTED iff N1 lowers it by >= 25% at >= 3 of the 4 heads; the selected set is ADOPTED iff a fold-10-27 refit (60k rows a day, target U,
       j = 1..3 at every H) with them normalized is not worse on 10-26 log loss on average than the review's raw refit (same rows and seeds).
    R2 TAILS (10-26, 200,000 rows, printed): per input the share of finite values beyond 10 IQR from the median, (p99.9 - p50) / IQR. Trees
       are invariant to monotone transforms: tails matter only through drift (R1) - no clipping rule.
    R3 MODEL REGULARIZATION (fold 10-27: 180k rows a day, target U, every head, 200 rounds; 10-26 log loss on every usable row) against v4's
       fold-10-27 models (pinned parameters): min_data_in_leaf 1000; lambda_l2 10; num_leaves 15. ADOPTED: the HELPS variant (mean relative
       gain >= 0.5% AND >= 10 of 18 heads better) with the largest mean gain; none -> the pinned parameters stay (a change is flagged: the
       LightGBM parameters were pinned since PULL-DEPTH).
    Output reg_20261115.json / .txt ('norm_inputs', 'lgb_params' are read by the v4 build)."""
    M.all_selftests()
    T0 = time.time(); asof = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()); L = []; res = dict(asof=asof, version='reg 2026-11-15')
    A = json.load(open(f'{M.OUTD}/audit2.json')); scale = A['scale']; RV = json.load(open(OUT_J))
    F0 = [c for c in M.FEAT_M if c not in set(A['drop'])]
    rdrop = set(RV['drop_inputs']) | {c for c, x in RV['C_inputs'].items() if x['group'] in RV['drop_groups']}
    F = [c for c in F0 if c not in rdrop]
    cands = sorted(({c for c in A['flag']} | {c for c, x in RV['C_inputs'].items() if x['median_drift_sd'] == x['median_drift_sd'] and x['median_drift_sd'] > NORM_DRIFT})
                   & set(F) - set(TIMEV) - {c for c in F if c.startswith('r_')})
    L.append(f'REGULARIZATION REVIEW (dev days only); as of {asof}; review drops applied: {sorted(rdrop) or "none"}; {len(F)} inputs')
    LB = {}; KK = {}; SEL = {}
    for d in M.DEV_DAYS:
        SEL[d] = M.day_usable(d); LB[d] = M.day_cols(d, LABC, SEL[d]); KK[d] = M.day_cols(d, ['m', 'side'])
    r1 = {}; selc = []
    for c in cands:
        xr = {}; xn = {}
        for d in M.DEV_DAYS:
            x = M.day_matrix(d, scale, [c])[:, 0].astype(np.float64); sc = M.trail_scale(x, KK[d]['m'], KK[d]['side'])
            xr[d] = x[SEL[d]]; xn[d] = (x / sc)[SEL[d]]
        z = {}
        for (h, j) in UHEADS:
            ys = [LB[d][M.DCOL[h]] >= j for d in M.DEV_DAYS]
            dr = 100.0 * M.shape_disp([xr[d] for d in M.DEV_DAYS], ys); dn = 100.0 * M.shape_disp([xn[d] for d in M.DEV_DAYS], ys)
            z[f'H{h}_j{j}'] = dict(disp_raw=dr, disp_norm=dn, auc_raw=M.auc_nan(np.concatenate(ys), np.concatenate([xr[d] for d in M.DEV_DAYS]))[0],
                                   auc_norm=M.auc_nan(np.concatenate(ys), np.concatenate([xn[d] for d in M.DEV_DAYS]))[0])
        nb = sum(1 for v in z.values() if v['disp_raw'] == v['disp_raw'] and v['disp_norm'] <= (1.0 - NORM_CUT) * v['disp_raw'])
        r1[c] = dict(heads=z, heads_better=nb, nan_norm=float(np.mean([np.isnan(xn[d]).mean() for d in M.DEV_DAYS])))
        if nb >= NORM_HEADS:
            selc.append(c)
        M.mem_guard(f'reg R1 {c}')
    s1 = lambda d: int(F1_TEST) * 31 + int(d)                                    # noqa: E731
    h1 = lambda h, j: int(F1_TEST) * 7 + j * 1009 + h                             # noqa: E731
    s26 = M.day_usable(EVAL); lab26 = {c: v.astype(np.float32) for c, v in M.day_cols(EVAL, LABC, s26).items()}
    adopt_norm = False; mnorm = None
    if selc:
        Xs, Ls, Ws = [], [], []
        for d in F1_TRAIN:
            ix, w = M.day_sample(len(SEL.get(d, M.day_usable(d))), M.VOL['V60'][1], s1(d))
            pos = SEL.get(d, M.day_usable(d))[ix]
            Xs.append(M.apply_norm(M.day_matrix(d, scale, F, pos), F, d, pos, selc, scale)); Ws.append(w)
            Ls.append({c: v.astype(np.float32) for c, v in M.day_cols(d, LABC, pos).items()})
        Tn = (np.concatenate(Xs), {c: np.concatenate([x[c] for x in Ls]) for c in LABC}, np.concatenate(Ws), np.zeros(sum(len(w) for w in Ws), bool))
        del Xs
        Bn = DG.train_heads(Tn, F, 'U', 'V60', h1, MHEADS); del Tn
        Xn = M.apply_norm(M.day_matrix(EVAL, scale, F, s26), F, EVAL, s26, selc, scale)
        mnorm = DG.metrics(DG.predict(Bn, Xn, MHEADS), lab26, 'U', MHEADS); del Xn, Bn
        base = RV['E_ablation']['ALL']
        if F == F0:                                                    # the review's raw refit used the same inputs, rows and seeds
            gain = [(base[k]['ll'] - mnorm[k]['ll']) / base[k]['ll'] for k in mnorm]; adopt_norm = bool(np.mean(gain) >= 0.0)
            res['R1_refit'] = dict(gain=dict(zip(mnorm, gain)), mean_gain=float(np.mean(gain)), heads_better=int(sum(g > 0 for g in gain)))
        else:
            res['R1_refit'] = 'not compared: the review dropped inputs (its raw refit had other inputs)'
        M.mem_guard('reg R1 refit')
    res.update(R1=r1, R1_candidates=cands, R1_selected=selc, norm_inputs=selc if adopt_norm else [])
    L.append(f'R1 INPUT CONDITIONING: candidates {cands}')
    L.append('   input | per head (H72 j1, H250 j1, H1000 j1, H1000 j3): shape dispersion pp raw -> normalized, AUC raw / normalized | heads better | NaN after N1')
    for c in cands:
        z = r1[c]
        L.append(f'   {c:15s} ' + ' | '.join(f'{v["disp_raw"]:.2f}->{v["disp_norm"]:.2f} {v["auc_raw"]:.3f}/{v["auc_norm"]:.3f}' for v in z['heads'].values())
                 + f' | {z["heads_better"]}/4 | {z["nan_norm"]:.1%}')
    L.append(f'   selected {selc or "none"}; refit {res.get("R1_refit", "not run (none selected)")}; -> normalize {res["norm_inputs"] or "none"}')
    # ---- R2 tails
    rg = np.random.default_rng(int(EVAL) + 1); it = np.sort(rg.choice(len(s26), 200000, replace=False)); XT = M.day_matrix(EVAL, scale, F, s26[it])
    tails = {}
    for i, c in enumerate(F):
        x = XT[:, i].astype(float); x = x[np.isfinite(x)]
        if len(x) < 1000:
            continue
        q1, q50, q3, q999 = np.quantile(x, [0.25, 0.5, 0.75, 0.999]); iqr = q3 - q1
        tails[c] = dict(iqr=float(iqr), beyond_10iqr=float((np.abs(x - q50) > 10 * iqr).mean()) if iqr > 0 else M.NAN,
                        p999_ratio=float((q999 - q50) / iqr) if iqr > 0 else M.NAN)
    del XT
    res['R2_tails'] = tails
    heavy = sorted([c for c, v in tails.items() if v['beyond_10iqr'] == v['beyond_10iqr'] and v['beyond_10iqr'] > 0.001], key=lambda c: -tails[c]['beyond_10iqr'])
    zero_iqr = [c for c, v in tails.items() if not v['iqr'] > 0]
    L.append(f'R2 TAILS (10-26, printed): beyond 10 IQR > 0.1% of rows: ' + (', '.join(f'{c} {tails[c]["beyond_10iqr"]:.2%}' for c in heavy) or 'none')
             + f'; IQR = 0 (mostly one value): {zero_iqr or "none"} (trees: no clipping rule)')
    # ---- R3 model regularization
    fj = json.load(open(f'{M.OUTD}/fit_{F1_TEST}_v4.json'))
    if fj['features'] != F0 or fj['decisions']['target'] != 'U' or fj['volume']['rows_a_day'] != M.VOL['V180'][1]:
        sys.exit('REFUSED: the v4 fold-10-27 fit is not the R3 base')
    PB = M.day_cols(f'{M.OUTD}/pred_fold{F1_TEST}_{EVAL}_v4.parquet', ['m', 'side', 'k', 'usable'] + [f'R{h}_{j}' for (h, j) in DG.HEADS])
    uu = PB['usable'].astype(bool)
    kk26 = M.day_cols(EVAL, ['m', 'side', 'k'], s26)
    if not (np.array_equal(PB['m'][uu], kk26['m']) and np.array_equal(PB['k'][uu], kk26['k'])):
        sys.exit('REFUSED: v4 fold-10-27 predictions not aligned with 10-26 rows')
    mbase = DG.metrics({hj: PB[f'R{hj[0]}_{hj[1]}'][uu] for hj in DG.HEADS}, lab26, 'U', DG.HEADS); del PB
    T180 = DG.train_table(F1_TRAIN, s1, 'V180', None, F0, scale); X26 = M.day_matrix(EVAL, scale, F0, s26); r3 = {}
    for nm, pv in REG_VARIANTS.items():
        X_, lab_, w_, _ = T180; P = {}
        for (h, j) in DG.HEADS:
            ii = np.flatnonzero(M.head_rows(lab_['D0'], j, 'U')); ic, wc = M.head_cap(None, w_[ii], M.VOL['V180'][2], h1(h, j)); ii = ii[ic]
            y = (lab_[M.DCOL[h]][ii] >= j).astype(int); b_, n_ = M.fit_head(X_[ii], y, wc, F0, params=pv)
            P[(h, j)] = b_.predict(X26).astype(np.float32); del b_
            M.mem_guard(f'reg R3 {nm} H{h} j{j}')
        mv = DG.metrics(P, lab26, 'U', DG.HEADS); del P
        mr, nb, ok = M.helps([mbase[hk(*hj)]['ll'] for hj in DG.HEADS], [mv[hk(*hj)]['ll'] for hj in DG.HEADS])
        r3[nm] = dict(params=pv, metrics=mv, mean_gain=mr, heads_better=nb, helps=ok)
    del T180, X26
    hl = [nm for nm in r3 if r3[nm]['helps']]; best = max(hl, key=lambda nm: r3[nm]['mean_gain']) if hl else None
    res.update(R3=r3, R3_base=mbase, lgb_params=(REG_VARIANTS[best] if best else {}), lgb_choice=best or 'pinned', seconds=round(time.time() - T0, 1))
    L.append('R3 MODEL REGULARIZATION (fold 10-27, V180, U, 18 heads, 10-26 log loss vs the pinned parameters): '
             + '; '.join(f'{nm} {r3[nm]["params"]} gain {r3[nm]["mean_gain"]:+.2%} better {r3[nm]["heads_better"]}/18 {"HELPS" if r3[nm]["helps"] else "no"}' for nm in r3)
             + f' -> {best or "keep the pinned parameters"}')
    L.append('   per head log loss pinned / ' + ' / '.join(r3) + ': ' + ' '.join(f'{hk(*hj)} {mbase[hk(*hj)]["ll"]:.5f}/' + '/'.join(f'{r3[nm]["metrics"][hk(*hj)]["ll"]:.5f}' for nm in r3)
                                                                       for hj in DG.HEADS))
    L.append(f'DECISIONS: normalize {res["norm_inputs"] or "none"}; LightGBM {res["lgb_choice"]} {res["lgb_params"]}; {res["seconds"]} s')
    json.dump(res, open(OUT_RJ, 'w'), indent=1, default=float); open(OUT_RT, 'w').write('\n'.join(L) + '\n')
    print('\n'.join(L), flush=True)


def hk(h, j):
    return f'H{h}_j{j}'


OUT_DJ = f'{M.OUTD}/dup_20261115.json'; OUT_DT = f'{M.OUTD}/dup_20261115.txt'


def dup():
    """WITHIN-SIDE COPIES (the code review D3; declared before the run, dev days): the 18 raw directional inputs with an adv_ copy (adv_x = -x
    for bids, +x for asks) and pf_pend_bid1000 / pf_pend_ask1000 (= own_pend / opp_pend by side) carry nothing the copy and fside do not;
    their pooled correlation is ~0, so the review's |r| > 0.999 rule cannot see them. Test: fold 10-27 refits (60k rows a day, target U, the 9
    heads j = 1..3 at every H, v3.1 seeds) with the review's drops applied, with and without the 20 copies; the copies are DROPPED iff the
    10-26 log loss without them is not worse on average (mean relative change >= 0). Output dup_20261115.json / .txt ('drop_inputs')."""
    M.all_selftests()
    T0 = time.time(); asof = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()); res = dict(asof=asof)
    A = json.load(open(f'{M.OUTD}/audit2.json')); scale = A['scale']; RV = json.load(open(OUT_J))
    rdrop = set(RV['drop_inputs']) | {c for c, x in RV['C_inputs'].items() if x['group'] in RV['drop_groups']}
    F = [c for c in M.FEAT_M if c not in set(A['drop']) | rdrop]
    cop = [c for c in M.SD.ADV if c in F and f'adv_{c}' in F] + [c for c in ('pf_pend_bid1000', 'pf_pend_ask1000') if c in F and {'own_pend', 'opp_pend'} <= set(F)]
    Fd = [c for c in F if c not in cop]
    s1 = lambda d: int(F1_TEST) * 31 + int(d)                                    # noqa: E731
    h1 = lambda h, j: int(F1_TEST) * 7 + j * 1009 + h                             # noqa: E731
    T60 = DG.train_table(F1_TRAIN, s1, 'V60', None, F, scale); s26 = M.day_usable(EVAL)
    X26 = M.day_matrix(EVAL, scale, F, s26); lab26 = {c: v.astype(np.float32) for c, v in M.day_cols(EVAL, LABC, s26).items()}
    out = {}
    for nm, FF in (('with', F), ('without', Fd)):
        keep = [F.index(c) for c in FF]; Tk = (np.ascontiguousarray(T60[0][:, keep]), T60[1], T60[2], T60[3])
        B = DG.train_heads(Tk, FF, 'U', 'V60', h1, MHEADS); out[nm] = DG.metrics(DG.predict(B, np.ascontiguousarray(X26[:, keep]), MHEADS), lab26, 'U', MHEADS)
        del Tk, B
        M.mem_guard(f'dup {nm}')
    rel = {k: (out['with'][k]['ll'] - out['without'][k]['ll']) / out['with'][k]['ll'] for k in out['with']}
    mg = float(np.mean(list(rel.values()))); drop = cop if mg >= 0.0 else []
    # R1's refit comparison (declared 2026-11-15 04:1xZ, before reg's output was read): reg compares against the review's raw refit only
    # when the review dropped nothing; it dropped 4 duplicates, so the comparison runs here on the FINAL input set: the inputs reg's
    # dispersion rule selected, normalized, against the same set raw (same rows and seeds); adopted iff the mean log-loss change >= 0
    RG = json.load(open(OUT_RJ)); Ff = [c for c in F if c not in drop]; sel = [c for c in RG['R1_selected'] if c in Ff]; norm = []
    if sel:
        keep = [F.index(c) for c in Ff]; Xs, Ls, Ws = [], [], []
        for d in F1_TRAIN:
            sd_ = M.day_usable(d); ix, w = M.day_sample(len(sd_), M.VOL['V60'][1], s1(d)); pos = sd_[ix]
            Xs.append(M.apply_norm(M.day_matrix(d, scale, Ff, pos), Ff, d, pos, sel, scale)); Ws.append(w)
            Ls.append({c: v.astype(np.float32) for c, v in M.day_cols(d, LABC, pos).items()})
        Tn = (np.concatenate(Xs), {c: np.concatenate([x[c] for x in Ls]) for c in LABC}, np.concatenate(Ws), np.zeros(sum(len(w) for w in Ws), bool)); del Xs
        B = DG.train_heads(Tn, Ff, 'U', 'V60', h1, MHEADS); del Tn
        Xn = M.apply_norm(np.ascontiguousarray(X26[:, keep]), Ff, EVAL, s26, sel, scale)
        mn = DG.metrics(DG.predict(B, Xn, MHEADS), lab26, 'U', MHEADS); del B, Xn
        bse = out['without'] if drop else out['with']
        g2 = {k: (bse[k]['ll'] - mn[k]['ll']) / bse[k]['ll'] for k in mn}; norm = sel if np.mean(list(g2.values())) >= 0.0 else []
        res.update(R1_final=dict(selected=sel, metrics=mn, rel_gain=g2, mean_gain=float(np.mean(list(g2.values()))), heads_better=int(sum(v > 0 for v in g2.values()))))
    res['norm_inputs'] = norm
    res.update(copies=cop, metrics=out, rel_gain_without=rel, mean_gain_without=mg, heads_better_without=int(sum(v > 0 for v in rel.values())), drop_inputs=drop,
               inputs_with=len(F), inputs_without=len(Fd), seconds=round(time.time() - T0, 1))
    L = [f'WITHIN-SIDE COPIES (fold 10-27 refits, V60, U, 9 heads; 10-26 log loss); as of {asof}',
         f'   copies ({len(cop)}): {cop}',
         '   head: log loss with / without (relative gain without): ' + '; '.join(f'{k} {out["with"][k]["ll"]:.5f} / {out["without"][k]["ll"]:.5f} ({rel[k]:+.2%})' for k in rel),
         f'   mean gain without {mg:+.3%}, better without {res["heads_better_without"]}/9 -> drop {"the copies" if drop else "none"} ({len(F)} -> {len(F) - len(drop)} inputs)',
         (f'   R1 final (reg selected {sel}): normalized vs raw on the final inputs, mean gain {res["R1_final"]["mean_gain"]:+.3%}, better {res["R1_final"]["heads_better"]}/9 -> '
          f'normalize {norm or "none"}' if sel else '   R1 final: reg selected no input (nothing to normalize)') + f'; {res["seconds"]} s']
    json.dump(res, open(OUT_DJ, 'w'), indent=1, default=float); open(OUT_DT, 'w').write('\n'.join(L) + '\n')
    print('\n'.join(L), flush=True)


S5_BN = ['bn_r25', 'bn_r50', 'bn_r100', 'bn_r250', 'bn_r500', 'bn_r1000', 'bn_r3000', 'bn_imb', 'bn_age', 'bn_flow250', 'bn_flow1000', 'bn_flow5000',
         'bn_ntr1000']
S5_PF = ['pf_spread', 'pf_mid', 'pf_micro_dev', 'pf_imb1', 'pf_imb3', 'pf_age', 'pf_nfr1000', 'pf_dm250', 'pf_dm1000', 'pf_dm5000']
K1_DAYS = ('20261024', '20261026'); K1_HOURS = 3; K1_WIN = 2; K1_STEPS = 50; K1_OK = 0.99
OUT_KJ = f'{M.OUTD}/s5check_20261115.json'; OUT_KT = f'{M.OUTD}/s5check_20261115.txt'


_BN_CACHE = {}


def _bn_hour(h, b_in):
    """One raw Binance hour (read once, cached by hour): bookTicker (known time E + b_in, mid, imbalance) and trades (known time, signed qty),
    with step 5's filters."""
    if h not in _BN_CACHE:
        _BN_CACHE[h] = _bn_hour_raw(h)
    bt, tr = _BN_CACHE[h]
    return ((bt[0] + b_in, bt[1], bt[2]) if bt is not None else None), ((tr[0] + b_in, tr[1]) if tr is not None else None)


def _bn_hour_raw(h):
    import pf_io as io
    b_in = 0.0; bt = tr = None
    p = f'{io.BN_DIR}/{h}.csv.gz'
    if os.path.exists(p):
        df = pd.read_csv(p, header=None, names=['recv', 'E', 'T', 'u', 'bid', 'bq', 'ask', 'aq'], usecols=['E', 'u', 'bid', 'bq', 'ask', 'aq'])
        df = df.apply(pd.to_numeric, errors='coerce').dropna(); df = df[(df.bid > 0) & (df.ask > df.bid)].drop_duplicates('u', keep='first')
        bt = (df.E.values.astype(np.float64) + b_in, (df.bid.values + df.ask.values) / 2, (df.bq.values - df.aq.values) / (df.bq.values + df.aq.values))
    tp = f'{io.ROOT}/data/mm_hf/raw/trade/BTCUSDT/{h}.csv.gz'
    if os.path.exists(tp):
        td = pd.read_csv(tp, header=None, names=['recv', 'E', 'T', 'id', 'px', 'qty', 'bm'], usecols=['E', 'qty', 'bm']).apply(pd.to_numeric, errors='coerce').dropna()
        tr = (td.E.values.astype(np.float64) + b_in, np.where(td.bm.values == 0, 1.0, -1.0) * td.qty.values)
    return bt, tr


def _recompute(t, BT, TR, FR, shift, pf_min=13.2, d_dec=2.0):
    """Step-5 inputs at decision times t from events known by t only: BT / TR = globally sorted Binance arrays, FR = one market's frames."""
    out = {c: np.full(len(t), np.nan) for c in S5_BN + S5_PF}
    k, m, imb = BT; i = np.searchsorted(k, t, side='right') - 1; ok = i >= 0; ic = np.clip(i, 0, None)
    out['bn_imb'] = np.where(ok, imb[ic], np.nan); out['bn_age'] = np.where(ok, t - k[ic], np.nan)
    for hh in (25, 50, 100, 250, 500, 1000, 3000):
        j = np.searchsorted(k, t - hh, side='right') - 1
        out[f'bn_r{hh}'] = np.where(ok & (j >= 0), 1e4 * np.log(m[ic] / m[np.clip(j, 0, None)]), np.nan)
    tk, sq = TR; csq = np.cumsum(sq); it = np.searchsorted(tk, t, side='right') - 1
    for hh in (250, 1000, 5000):
        jt = np.searchsorted(tk, t - hh, side='right') - 1
        out[f'bn_flow{hh}'] = np.where(it >= 0, csq[np.clip(it, 0, None)], 0.0) - np.where(jt >= 0, csq[np.clip(jt, 0, None)], 0.0)
    out['bn_ntr1000'] = (it - (np.searchsorted(tk, t - 1000, side='right') - 1)).astype(float)
    g = FR.sort_values(['t_ms', 'recv_ns'], kind='stable')
    kn = np.maximum.accumulate(np.maximum(g.t_ms.values.astype(np.float64) + pf_min, g.recv_ns.values.astype(np.float64) / 1e6 - shift) + d_dec)
    b1, a1, bs1, as1 = g.b1_p.values, g.a1_p.values, g.b1_s.values, g.a1_s.values; valid = (b1 >= 0) & (a1 > b1)
    midv = np.where(valid, (a1 + b1) / 2.0, np.nan); mf = pd.Series(midv).ffill().values
    i = np.searchsorted(kn, t, side='right') - 1; ok = i >= 0; ic = np.clip(i, 0, None); vv = ok & valid[ic]
    out['pf_spread'] = np.where(vv, (a1 - b1)[ic], np.nan); out['pf_mid'] = np.where(vv, midv[ic], np.nan)
    micro = (b1 * as1 + a1 * bs1) / np.maximum(bs1 + as1, 1e-9); out['pf_micro_dev'] = np.where(vv, (micro - midv)[ic], np.nan)
    out['pf_imb1'] = np.where(vv, ((bs1 - as1) / np.maximum(bs1 + as1, 1e-9))[ic], np.nan)
    sb3 = sum(np.where(g[f'b{j}_p'].values >= 0, g[f'b{j}_s'].values, 0) for j in (1, 2, 3))
    sa3 = sum(np.where(g[f'a{j}_p'].values >= 0, g[f'a{j}_s'].values, 0) for j in (1, 2, 3))
    out['pf_imb3'] = np.where(vv, ((sb3 - sa3) / np.maximum(sb3 + sa3, 1e-9))[ic], np.nan)
    out['pf_age'] = np.where(ok, t - kn[ic], np.nan); out['pf_nfr1000'] = (i - (np.searchsorted(kn, t - 1000, side='right') - 1)).astype(float)
    cur = np.where(ok, mf[ic], np.nan)
    for hh in (250, 1000, 5000):
        j = np.searchsorted(kn, t - hh, side='right') - 1; out[f'pf_dm{hh}'] = cur - np.where(j >= 0, mf[np.clip(j, 0, None)], np.nan)
    return out


def _agree(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float); fa, fb = np.isfinite(a), np.isfinite(b); both = fa & fb
    close = np.abs(a[both] - b[both]) <= np.maximum(1e-4 * np.abs(b[both]), 1e-3)
    return dict(n=int(both.sum()), match=float(close.mean()) if both.any() else M.NAN, nan_agree=float((fa == fb).mean()))


def s5check():
    """INPUT-PIPELINE CHECKS (the independent code review 2026-11-15: D1 / D2 / D9 / D12 / D13); declared before the run.
    K1 STEP-5 RECOMPUTATION (D1: the future-perturbation falsifier covers only the 25 model-side inputs): on 10-24 and 10-26, 2 windows in
       each of 3 random hours, both sides, 50 random steps per window-side; 23 step-5 inputs (S5_BN + S5_PF) recomputed independently from
       the raw Binance hour files and the day's Predict frames with ONLY events known by t (Binance E + 1.5 + 2 ms; a Predict frame at the
       cumulative max of max(update + 13.2, receipt - shift) + 2 ms; the day's shift from clock_rl_d2.json); PASS iff each input matches the
       stored row (|d| <= max(1e-4 relative, 1e-3)) on >= 99% of the steps where both are finite and NaN agrees on >= 99%. POSITIVE CONTROLS
       (must FAIL the same criterion): Binance known at E - 50 ms; Predict frames known at their update time (no feed delay).
    K2 STALE / FUTURE BINANCE (D2 / D13; nine days, counts): steps with bn_age < 0 (an event after t), > 1 s, > 60 s; missing Binance
       bookTicker / trade hour files for the day's 24 hours and the hour before (sealed days are not listed).
    K3 TWO CLOCKS (D9; nine days, counts): rows where step 5's pf_spread differs from the model side's spread, and expm1(own_l1) from s1.
    Predict part files are read only for the day's own hours and the previous day's last hour (D12)."""
    M.all_selftests()
    T0 = time.time(); asof = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()); L = []; res = dict(asof=asof)
    L.append(f'INPUT-PIPELINE CHECKS; as of {asof}')
    k1 = {}; pos = {'bn_future': [], 'pf_nodelay': []}; allok = True
    for d in K1_DAYS:
        ck = json.load(open(f'{M.K.IO}/step1/{d}/clock_rl_d2.json')); shift = float(ck['shift_ms']); b_in = float(ck['bn_in']) + float(ck['d_dec'])
        R = M.day_cols(d, ['m', 'side', 'k', 't'] + S5_BN + S5_PF); rg = np.random.default_rng(int(d))
        hrs = (R['t'] // 3600000).astype(np.int64); cand_h = np.unique(hrs[(R['t'] % 3600000) > 120000])
        pick_h = rg.choice(cand_h, K1_HOURS, replace=False); rows = []
        for hh in pick_h:
            ms = np.unique(R['m'][hrs == hh]); ms = rg.choice(ms, min(K1_WIN, len(ms)), replace=False)
            for m_ in ms:
                for s_ in (0, 1):
                    ii = np.flatnonzero((R['m'] == m_) & (R['side'] == s_))
                    rows.extend(rg.choice(ii, min(K1_STEPS, len(ii)), replace=False).tolist())
        rows = np.sort(np.array(rows)); t = R['t'][rows].astype(np.float64); mm = R['m'][rows]
        hours = sorted({int(x) for x in (t // 3600000)} | {int(x) - 1 for x in (t // 3600000)})
        hname = [pd.Timestamp(h * 3600000, unit='ms', tz='UTC').strftime('%Y%m%d_%H') for h in hours]
        books = [f for f in sorted(os.listdir(f'{M.K.IO}/step1/{d}/books')) if f[5:13] == d or (f[5:13] < d and f[14:16] == '23')]
        mids = set(int(x) for x in np.unique(mm))
        FR = pd.concat([pd.read_parquet(f'{M.K.IO}/step1/{d}/books/{f}', columns=['mid', 't_ms', 'recv_ns'] + [f'{x}{i}_{y}' for x in 'ba' for i in (1, 2, 3) for y in 'ps'],
                                        filters=[('mid', 'in', sorted(mids))]) for f in books], ignore_index=True)
        for tag, bi, pdel in (('real', b_in, None), ('bn_future', -50.0, None), ('pf_nodelay', b_in, 'nodelay')):
            BTs, TRs = [], []
            for h in hname:
                bt, tr = _bn_hour(h, bi)
                if bt is not None:
                    BTs.append(bt)
                if tr is not None:
                    TRs.append(tr)
            k_ = np.concatenate([x[0] for x in BTs]); o = np.argsort(k_, kind='stable'); BT = (k_[o], np.concatenate([x[1] for x in BTs])[o], np.concatenate([x[2] for x in BTs])[o])
            tk_ = np.concatenate([x[0] for x in TRs]); o = np.argsort(tk_, kind='stable'); TR = (tk_[o], np.concatenate([x[1] for x in TRs])[o])
            rec = {c: np.full(len(rows), np.nan) for c in S5_BN + S5_PF}
            for m_ in mids:
                sl = np.flatnonzero(mm == m_)
                fr = FR[FR.mid.values == m_]
                if pdel == 'nodelay':
                    fr = fr.assign(recv_ns=fr.t_ms.values.astype(np.float64) * 1e6)
                o_ = _recompute(t[sl], BT, TR, fr, shift if pdel is None else 0.0, pf_min=(13.2 if pdel is None else 0.0), d_dec=(2.0 if pdel is None else 0.0))
                for c in rec:
                    rec[c][sl] = o_[c]
            ag = {c: _agree(rec[c], R[c][rows]) for c in S5_BN + S5_PF}
            ok_ = all((x['match'] >= K1_OK or x['n'] == 0) and x['nan_agree'] >= K1_OK for x in ag.values())
            if tag == 'real':
                k1[d] = dict(rows=int(len(rows)), hours=hname, inputs=ag, ok=ok_); allok &= ok_
            else:
                pos[tag].append(dict(day=d, fails=not ok_, worst={c: x['match'] for c, x in sorted(ag.items(), key=lambda z: z[1]['match'] if z[1]['match'] == z[1]['match'] else 1)[:4]}))
            del BTs, TRs, BT, TR
            M.mem_guard(f's5check {d} {tag}')
        del FR, R; _BN_CACHE.clear()
    pos_ok = all(x['fails'] for v in pos.values() for x in v)
    res.update(K1=k1, K1_positive=pos, K1_pass=bool(allok), K1_controls_fire=bool(pos_ok))
    L.append(f'K1 STEP-5 RECOMPUTATION (raw data known by t vs stored rows): {"PASS" if allok else "FAIL"}; positive controls fire: {pos_ok}')
    for d, x in k1.items():
        bad = {c: v for c, v in x['inputs'].items() if not ((v['match'] >= K1_OK or v['n'] == 0) and v['nan_agree'] >= K1_OK)}
        L.append(f'   {d}: {x["rows"]} steps, hours {x["hours"]}; min match {min(v["match"] for v in x["inputs"].values() if v["n"]):.4f}; '
                 f'min NaN agreement {min(v["nan_agree"] for v in x["inputs"].values()):.4f}; failing {bad or "none"}')
    for tag, v in pos.items():
        L.append(f'   control {tag}: ' + '; '.join(f'{x["day"]} fails {x["fails"]} worst {x["worst"]}' for x in v))
    # K2 / K3 counts on every day
    import pf_io as io
    k23 = {}
    for d in M.DAYS9:
        R = M.day_cols(d, ['bn_age', 'pf_spread', 'spread', 'own_l1', 's1'])
        ba = R['bn_age'].astype(float); fs = np.isfinite(R['pf_spread']) & np.isfinite(R['spread']); fo = np.isfinite(R['own_l1'])
        prev = (pd.Timestamp(f'{d[:4]}-{d[4:6]}-{d[6:]}', tz='UTC') - pd.Timedelta(hours=1)).strftime('%Y%m%d_%H')
        hrs = ([prev] if prev[:8] not in ('20261028', '20261029', '20261030') else []) + [f'{d}_{h:02d}' for h in range(24)]
        k23[d] = dict(rows=int(len(ba)), bn_age_neg=int((ba < 0).sum()), bn_age_gt1s=int((ba > 1000).sum()), bn_age_gt60s=int((ba > 60000).sum()),
                      bn_age_nan=int((~np.isfinite(ba)).sum()), missing_bookticker=[h for h in hrs if not os.path.exists(f'{io.BN_DIR}/{h}.csv.gz')],
                      missing_trades=[h for h in hrs if not os.path.exists(f'{io.ROOT}/data/mm_hf/raw/trade/BTCUSDT/{h}.csv.gz')],
                      spread_differs=float((R['pf_spread'][fs] != R['spread'][fs]).mean()) if fs.any() else M.NAN,
                      l1_differs=float((np.abs(np.expm1(R['own_l1'][fo].astype(float)) - R['s1'][fo]) > 1e-3).mean()) if fo.any() else M.NAN)
        del R
    res['K2_K3'] = k23; res['seconds'] = round(time.time() - T0, 1)
    L.append('K2 / K3 (counts): day | rows | bn_age < 0 / > 1 s / > 60 s / NaN | missing Binance bookTicker / trade hours | pf_spread != spread | '
             'expm1(own_l1) != s1')
    for d, x in k23.items():
        L.append(f'   {d}: {x["rows"]} | {x["bn_age_neg"]} / {x["bn_age_gt1s"]} / {x["bn_age_gt60s"]} / {x["bn_age_nan"]} | {x["missing_bookticker"] or "none"} / '
                 f'{x["missing_trades"] or "none"} | {x["spread_differs"]:.3%} | {x["l1_differs"]:.3%}')
    json.dump(res, open(OUT_KJ, 'w'), indent=1, default=float); open(OUT_KT, 'w').write('\n'.join(L) + '\n')
    print('\n'.join(L), flush=True)


if __name__ == '__main__':
    cmd = sys.argv[1] if len(sys.argv) > 1 else 'selftest'
    if cmd == 'selftest':
        M.all_selftests(); b_ = selftest(); print('review selftest', b_ or 'passed')
    elif cmd == 'run':
        run()
    elif cmd == 'reg':
        reg()
    elif cmd == 's5check':
        s5check()
    elif cmd == 'dup':
        dup()
    else:
        sys.exit(f'unknown command {cmd}')
