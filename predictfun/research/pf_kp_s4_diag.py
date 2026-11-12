"""Keep-or-pull S4 diagnostic (2026-11-12, USER: "current model does not capture so many true positives, it still fires a lot on false
[positives] ... can you check the raw input to find out any reasons"). Post hoc: no readings, nothing here is validated on these days.

On S4's 1.10 M never-pull rows (9 replay days, held-out predictions of the S4 models, M 50, y1 = Delta at H 5 s / 1 s, c/sh):
  A  input sanity    per input: NaN share, share at its most common value, p1 / p50 / p99, infinities, day drift (max |day mean -
                     pooled mean| / pooled sd); our order's invariants: behind an integer, age >= 0, a0k >= 0, shown_min_rel <= 1
                     and <= shown_now_rel; a planted copy (bad values written into 100 rows) must be flagged (falsifier); the
                     distribution of behind (< 0 = the known book's best is worse than our price: we would be alone at the best)
  B  label anatomy   Delta = Q_keep - Q_pull split by what made it: keep path's fill lost / won, pull path's re-entry fill won / lost
  C  parity          the saved models re-predict their held-out rows = oof.parquet (alignment by day, side and y1)
  D  contributions   LightGBM per-input contributions on held-out rows (a sample: up to 400 rows per group and day-side), mean per decision group: TP (pred < 0, y < 0), FP+ (pred < 0,
                     y > 0), FP0 (pred < 0, y = 0), FN (pred >= 0, y < 0); the inputs that push FP rows below 0
  E  raw separation  per input (and the model): AUC for "pulling right" (y < 0 vs rest), for the sign among rows with a fill
                     (y < 0 vs y > 0), and for activity (y != 0 vs 0); days on which the sign AUC has the pooled direction
Output: diag/ in the S4 output directory. Usage (through run_light, PF_SLOT=claude): pf_kp_s4_diag.py
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import rankdata

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
OUT4 = '/home/yuqing/ctaNew/data/predict_fun/research/kp_s4_20261112'
OUTD = f'{OUT4}/diag'
DAYS9 = ['20261023', '20261024', '20261025', '20261026', '20261027', '20261031', '20261101', '20261102', '20261103']
OWN = ['age', 'behind', 'a0k', 'shown_now', 'shown_now_rel', 'shown_min_rel']
NONFEAT = {'day', 'mid', 'side', 'k', 'y1', 'y5', 'qk1', 'qp1'}


def auc(x, pos):
    """AUC of x for the boolean pos (NaN x dropped); 0.5 = no separation, > 0.5 = pos has larger x."""
    ok = np.isfinite(x); x = x[ok]; pos = pos[ok]; n1 = int(pos.sum()); n0 = len(x) - n1
    if n1 == 0 or n0 == 0:
        return float('nan')
    r = rankdata(x)
    return float((r[pos].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def own_invariants(A):
    """Counts of rows breaking our order's invariants (NaN is not a violation; its count is reported in A)."""
    b = A.behind.values; f = lambda m: int(np.nansum(m))
    with np.errstate(invalid='ignore'):
        return dict(behind_not_integer=f(np.isfinite(b) & (b != np.round(b))), age_negative=f(A.age.values < 0),
                    a0k_negative=f(A.a0k.values < 0), shown_min_rel_above_1=f(A.shown_min_rel.values > 1 + 1e-6),
                    shown_min_above_now=f(A.shown_min_rel.values > A.shown_now_rel.values + 1e-6))


def main():
    import lightgbm as lgb
    os.makedirs(OUTD, exist_ok=True)
    A = pd.concat([pd.read_parquet(f'{OUT4}/rows_{d}.parquet') for d in DAYS9], ignore_index=True)
    FEATS = [c for c in A.columns if c not in NONFEAT]
    if len(FEATS) != 43 or len(set(FEATS)) != 43 or FEATS[-6:] != OWN:
        sys.exit(f'REFUSED: unexpected input columns {FEATS}')
    res = dict(rows=int(len(A)))
    # A: sanity
    san = {}
    for c in FEATS:
        x = A[c].values.astype(float); fin = x[np.isfinite(x)]
        vc = pd.Series(fin).value_counts()
        dm = A.groupby('day')[c].mean().values; sd = float(np.nanstd(x))
        san[c] = dict(nan=float(np.isnan(x).mean()), inf=int(np.isinf(x).sum()), mode_share=float(vc.iloc[0] / len(x)) if len(vc) else 1.0,
                      p1=float(np.quantile(fin, 0.01)) if len(fin) else float('nan'), p50=float(np.quantile(fin, 0.5)) if len(fin) else float('nan'),
                      p99=float(np.quantile(fin, 0.99)) if len(fin) else float('nan'),
                      drift=float(np.nanmax(np.abs(dm - np.nanmean(x))) / sd) if sd > 0 else float('nan'))
    inv = own_invariants(A)
    P = A.loc[:99, OWN].copy(); P['behind'] = 0.5; P['age'] = -1.0; P['a0k'] = -2.0; P['shown_min_rel'] = 2.0; P['shown_now_rel'] = 1.0
    planted = own_invariants(P)
    if any(v != 100 for v in planted.values()):
        sys.exit(f'FALSIFIER FAILED: the invariant check did not flag every planted row {planted}')
    res['sanity'] = san; res['own_invariants'] = inv; res['own_invariants_planted'] = planted
    bh = A.behind.values; res['behind'] = {k_: float(m_.mean()) for k_, m_ in (('nan', np.isnan(bh)), ('<0', bh < 0), ('0', bh == 0), ('1', bh == 1), ('2', bh == 2), ('3', bh == 3), ('>3', bh > 3))}
    # B: label anatomy
    y = A.y1.values.astype(float); qk = A.qk1.values.astype(float); qp = A.qp1.values.astype(float)
    neg = y < 0; pos = y > 0
    anat = dict(
        y_neg=float(neg.mean()), y_zero=float((y == 0).mean()), y_pos=float(pos.mean()),
        neg_keep_loses_pull_nothing=float((neg & (qk < 0) & (qp == 0)).sum() / neg.sum()),
        neg_keep_nothing_pull_wins=float((neg & (qk == 0) & (qp > 0)).sum() / neg.sum()),
        neg_both_filled=float((neg & (qk != 0) & (qp != 0)).sum() / neg.sum()),
        pos_keep_wins_pull_nothing=float((pos & (qk > 0) & (qp == 0)).sum() / pos.sum()),
        pos_keep_nothing_pull_loses=float((pos & (qk == 0) & (qp < 0)).sum() / pos.sum()),
        pos_both_filled=float((pos & (qk != 0) & (qp != 0)).sum() / pos.sum()),
        mean_y_neg=float(y[neg].mean()), mean_y_pos=float(y[pos].mean()),
        sum_neg_from_keep_losses=float(np.minimum(qk[neg], 0).sum() / y[neg].sum()),
        keep_filled=float((qk != 0).mean()), pull_filled=float((qp != 0).mean()), keep_mean=float(qk.mean()), pull_mean=float(qp.mean()),
        keep_neg_given_filled=float((qk[qk != 0] < 0).mean()), pull_neg_given_filled=float((qp[qp != 0] < 0).mean()))
    res['label'] = anat
    # C: parity + D: contributions
    O = pd.read_parquet(f'{OUT4}/oof.parquet')
    import time
    T0 = time.time(); pk = np.full(len(A), np.nan); maxdiff = 0.0; BST = {}
    for d in DAYS9:
        for s_ in (0, 1):
            ia = np.where((A.day.values == int(d)) & (A.side.values == s_))[0]
            o = O[(O.day.values == int(d)) & (O.side.values == s_)]
            if len(o) != len(ia) or not np.array_equal(o.y1.values, A.y1.values[ia]):
                sys.exit(f'REFUSED: oof rows of {d} side {s_} do not align with the built rows')
            BST[(d, s_)] = bst = lgb.Booster(model_file=f'{OUT4}/models/KP_{d}_{s_}.txt')
            p = bst.predict(A.loc[ia, FEATS].values.astype(np.float32)); maxdiff = max(maxdiff, float(np.max(np.abs(p - o.p_kp.values))))
            pk[ia] = p
    if maxdiff > 1e-6:
        sys.exit(f'PARITY FAILED: re-predicted held-out rows differ from oof.parquet by {maxdiff}')
    res['parity_max_abs_diff'] = maxdiff
    print(f'parity done, {time.time() - T0:.0f} s', flush=True)
    pull = pk < 0
    groups = dict(TP=pull & neg, FPpos=pull & pos, FP0=pull & (y == 0), FN=~pull & neg, TNpos=~pull & pos, TN0=~pull & (y == 0))
    res['groups'] = {g: dict(share=float(m.mean()), mean_y=float(y[m].mean()), mean_pred=float(pk[m].mean())) for g, m in groups.items()}
    # contributions on a sample (exact TreeSHAP on all rows is hours on one core): up to CPG rows per group and day-side
    CPG = 400; rng = np.random.default_rng(20261112); gid = np.full(len(A), -1)
    for i_, m in enumerate(groups.values()):
        gid[m] = i_
    C = np.full((len(A), len(FEATS) + 1), np.nan, np.float32); smp = np.zeros(len(A), bool)
    for (d, s_), bst in BST.items():
        ia = np.where((A.day.values == int(d)) & (A.side.values == s_))[0]
        pick = np.concatenate([rng.choice(ia[gid[ia] == g], size=min(CPG, int((gid[ia] == g).sum())), replace=False) for g in range(len(groups))])
        C[pick] = bst.predict(A.loc[pick, FEATS].values.astype(np.float32), pred_contrib=True).astype(np.float32); smp[pick] = True
        print(f'contributions {d} side {s_}: {len(pick)} rows, {time.time() - T0:.0f} s', flush=True)
    res['contrib_rows'] = {g: int((m & smp).sum()) for g, m in groups.items()}
    res['contrib_base'] = float(np.nanmean(C[smp, -1]))
    con = {g: {c: float(C[m & smp, i].mean()) for i, c in enumerate(FEATS)} for g, m in groups.items()}
    res['contrib'] = con
    mu = A[FEATS].mean(); sd = A[FEATS].std()
    res['raw_z'] = {g: {c: float((A.loc[m, c].mean() - mu[c]) / sd[c]) if sd[c] > 0 else float('nan') for c in FEATS} for g, m in groups.items()}
    # E: raw separation
    print(f'contributions done, {time.time() - T0:.0f} s', flush=True)
    fill = y != 0; sep = {}
    for c in FEATS + ['pred_KP']:
        x = pk if c == 'pred_KP' else A[c].values.astype(float)
        a_pull = auc(x, neg); a_sign = auc(x[fill], neg[fill]); a_act = auc(x, fill)
        dsgn = []
        for d in DAYS9:
            md = (A.day.values == int(d)) & fill
            dsgn.append(auc(x[md], neg[md]))
        dsgn = np.array(dsgn); dirn = np.sign(a_sign - 0.5)
        sep[c] = dict(auc_pull=a_pull, auc_sign=a_sign, auc_act=a_act, sign_days_same=int(np.sum(np.sign(dsgn - 0.5) == dirn)),
                      auc_sign_days_min=float(np.nanmin(dsgn)), auc_sign_days_max=float(np.nanmax(dsgn)))
    res['separation'] = sep
    print(f'separation done, {time.time() - T0:.0f} s', flush=True)
    json.dump(res, open(f'{OUTD}/kp_s4_diag.json', 'w'), indent=1, default=float)
    # text
    L = [f'Keep-or-pull S4 diagnostic (post hoc). {len(A)} never-pull rows, 9 days, held-out predictions, M 50, y1 c/sh.', '',
         'A input sanity: inputs with NaN > 5%, a single value > 50% of rows, or day drift > 0.5 sd:']
    for c, s in san.items():
        if s['nan'] > 0.05 or s['mode_share'] > 0.5 or s['drift'] > 0.5 or s['inf']:
            L.append(f'  {c:16s} NaN {s["nan"]:.3f}  mode share {s["mode_share"]:.3f}  p1 / p50 / p99 {s["p1"]:+.4g} / {s["p50"]:+.4g} / {s["p99"]:+.4g}'
                     f'  drift {s["drift"]:.2f}  inf {s["inf"]}')
    L.append(f'  our order invariants violated (rows): {inv}; planted 100 bad rows flagged: {planted}')
    L.append(f'  behind (ticks, share of rows): {json.dumps({k_: round(v_, 4) for k_, v_ in res["behind"].items()})}')
    L += ['', 'B label anatomy (y1 = Q_keep - Q_pull):',
          f'  y < 0 {anat["y_neg"]:.3f}, y = 0 {anat["y_zero"]:.3f}, y > 0 {anat["y_pos"]:.3f}; mean y given < 0 {anat["mean_y_neg"]:+.2f}, given > 0 {anat["mean_y_pos"]:+.2f}',
          f'  pulling right (y < 0): keep path loses and pull path no fill {anat["neg_keep_loses_pull_nothing"]:.3f}; keep no fill and the re-entered '
          f'order wins {anat["neg_keep_nothing_pull_wins"]:.3f}; both filled {anat["neg_both_filled"]:.3f}; share of the negative sum from keep losses '
          f'{anat["sum_neg_from_keep_losses"]:.3f}',
          f'  keeping right (y > 0): keep wins and pull no fill {anat["pos_keep_wins_pull_nothing"]:.3f}; keep no fill and the re-entered order loses '
          f'{anat["pos_keep_nothing_pull_loses"]:.3f}; both filled {anat["pos_both_filled"]:.3f}',
          f'  keep path filled in {anat["keep_filled"]:.3f} of rows (mean {anat["keep_mean"]:+.3f}, losing given a fill {anat["keep_neg_given_filled"]:.3f}); '
          f'pull path filled in {anat["pull_filled"]:.3f} (mean {anat["pull_mean"]:+.3f}, losing given a fill {anat["pull_neg_given_filled"]:.3f})',
          '', f'C parity: re-predicted held-out rows vs oof.parquet, max |diff| {maxdiff:.2e}', '',
          'D decision groups (KP, pull when pred < 0): share of rows, mean y, mean prediction']
    for g, s in res['groups'].items():
        L.append(f'  {g:6s} {s["share"]:.3f}  mean y {s["mean_y"]:+.3f}  mean pred {s["mean_pred"]:+.3f}')
    L.append(f'  mean contribution (base) {res["contrib_base"]:+.3f}; inputs by mean contribution in FP0 + FP+ rows (what pushes false pulls below 0),'
             f' with TP and the raw input z (FP0 / FP+ / TP):')
    fpm = groups['FP0'] | groups['FPpos']
    cfp = {c: float(C[fpm & smp, i].mean()) for i, c in enumerate(FEATS)}
    for c in sorted(FEATS, key=lambda c: cfp[c])[:12]:
        L.append(f'  {c:16s} FP {cfp[c]:+.3f}  TP {con["TP"][c]:+.3f}  FN {con["FN"][c]:+.3f}  | z FP0 {res["raw_z"]["FP0"][c]:+.2f}  FP+ {res["raw_z"]["FPpos"][c]:+.2f}'
                 f'  TP {res["raw_z"]["TP"][c]:+.2f}  FN {res["raw_z"]["FN"][c]:+.2f}')
    L.append('  inputs that separate TP from FP the most (mean contribution TP - FP):')
    for c in sorted(FEATS, key=lambda c: con['TP'][c] - cfp[c])[:8]:
        L.append(f'  {c:16s} TP - FP {con["TP"][c] - cfp[c]:+.3f}')
    L += ['', 'E raw separation, AUC (0.5 = none): pulling right (y < 0 vs rest) | sign among rows with a fill (y < 0 vs y > 0) | activity (y != 0)',
          '  | days (of 9) with the pooled sign direction, sign AUC range over days; sorted by |sign AUC - 0.5|']
    for c in sorted(sep, key=lambda c: -abs(sep[c]['auc_sign'] - 0.5) if sep[c]['auc_sign'] == sep[c]['auc_sign'] else 0):
        s = sep[c]
        L.append(f'  {c:16s} pull {s["auc_pull"]:.3f} | sign {s["auc_sign"]:.3f} | act {s["auc_act"]:.3f} | {s["sign_days_same"]} days,'
                 f' {s["auc_sign_days_min"]:.3f}..{s["auc_sign_days_max"]:.3f}')
    open(f'{OUTD}/kp_s4_diag.txt', 'w').write('\n'.join(L) + '\n')
    print('\n'.join(L))


if __name__ == '__main__':
    main()
