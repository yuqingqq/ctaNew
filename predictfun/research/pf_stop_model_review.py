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
Usage (run_light, PF_SLOT=claude): pf_stop_model_review.py selftest | run
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


if __name__ == '__main__':
    cmd = sys.argv[1] if len(sys.argv) > 1 else 'selftest'
    if cmd == 'selftest':
        M.all_selftests(); b_ = selftest(); print('review selftest', b_ or 'passed')
    elif cmd == 'run':
        run()
    else:
        sys.exit(f'unknown command {cmd}')
