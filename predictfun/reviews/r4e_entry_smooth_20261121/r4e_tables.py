"""R4E ADV2-ENTRY-SMOOTH-v1 receipt tables: markdown copies of the round's outputs, so that every number in the receipt is copied by code. A
formatter: it computes no predicate. Each number is a value read from an output, a sum, difference or ratio of such values (a mean is a sum
over a count; pooled means are weight-pooled with each bin's w_sum), or a count of rows.
Usage: r4e_tables.py [data dir] -> markdown on stdout"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('/home/yuqing/ctaNew/data/predict_fun/research/adv2_entry_smooth_v1_20261121')
EV, RP = OUT / 'eval', OUT / 'replay_v1'
R3_PRED = Path('/home/yuqing/ctaNew/predictfun/reviews/r3_adv2_20261121/eval/predictions.csv.gz')
SIDE = {0: 'bid', 1: 'ask'}
CANDS = ('SMOOTH', 'RICH')
ORDER = ['(-inf, -10]', '(-10, -5]', '(-5, -2]', '(-2, -1]', '(-1, -0.5]', '(-0.5, 0)', '0 (exact)', '(0, 0.5]', '(0.5, 1]', '(1, 2]', '(2, 5]', '(5, 10]', '(10, inf]']


def f_(x, nd=2):
    if x is None or (isinstance(x, float) and not np.isfinite(x)) or (isinstance(x, str) and x in ('', 'nan', 'None')):
        return '—'
    v = float(x)
    return f'{int(v):,}' if v.is_integer() and abs(v) < 1e15 else f'{v:,.{nd}f}'


def table(head, body):
    esc = lambda c: str(c).replace('|', '\\|')
    out = ['| ' + ' | '.join(esc(h) for h in head) + ' |', '|' + '---|' * len(head)]
    out += ['| ' + ' | '.join(esc(c) for c in r) + ' |' for r in body]
    return '\n'.join(out)


def checks():
    J = json.loads((HERE / 'checks.json').read_text())
    print('### Declared checks\n')
    print(table(['check', 'value'], [[k, str(v).lower()] for k, v in sorted(J['checks'].items())] + [['all_true', str(J['all_true']).lower()]]) + '\n')
    fp = HERE / 'checks_falsifier.json'
    if fp.exists():
        F = json.loads(fp.read_text())
        print('Falsifier (each check on a clean copy of its real inputs, then after each corruption):\n')
        print(table(['check', 'case', 'value'], [[k, c, str(x).lower()] for k, v in sorted(F['cases'].items()) for c, x in v.items()] + [['falsifier_ok', '', str(F['falsifier_ok']).lower()]]) + '\n')


def identity():
    ev = json.loads((EV / 'eval.json').read_text()); i = ev['identity']
    print('### Reuse of the existing labels (checked before any reading)\n')
    print(table(['rows', "R3's rows", 'same keys', 'labels equal', 'weights equal', 'RICH predictions equal'],
                [[f_(i['rows']), f_(i['r3_rows']), str(i['same_keys']).lower(), str(i['y_equal']).lower(), str(i['w_equal']).lower(), str(i['rich_pred_equal']).lower()]]) + '\n')


def errors():
    rd = pd.read_csv(EV / 'readings.csv'); g = rd[rd.group == 'all']
    print('### Prediction error (all ok POST rows; sampling-weighted; cents squared)\n')
    body = []
    for (day, s), x in g.groupby(['day', 'side'], sort=True):
        r = {c: x[x.candidate == c].iloc[0] for c in CANDS}; z = r['SMOOTH']
        body.append([day, SIDE[s], f_(z.rows), f_(z.meaningful_originals), f_(z.wmse_zero), f_(z.wmse_train_mean), f_(r['SMOOTH'].wmse), f_(r['RICH'].wmse),
                     f_(r['SMOOTH'].wmse / z.wmse_zero, 4), f_(r['RICH'].wmse / z.wmse_zero, 4), f_(r['SMOOTH'].wmae), f_(r['RICH'].wmae), f_(z.wmae_zero),
                     f_(r['SMOOTH'].spearman, 3), f_(r['RICH'].spearman, 3)])
    print(table(['day', 'side', 'rows', 'meaningful nonzero originals', 'MSE zero', 'MSE train mean', 'MSE SMOOTH', 'MSE RICH', 'SMOOTH / zero', 'RICH / zero',
                 'MAE SMOOTH', 'MAE RICH', 'MAE zero', 'Spearman SMOOTH', 'Spearman RICH'], body) + '\n')


def values():
    rd = pd.read_csv(EV / 'readings.csv'); g = rd[rd.group == 'all']
    print('### Local entry value per opportunity (cents; sampling-weighted; pi0-conditional, never a policy return; never posting = 0)\n')
    body = []
    for (day, s), x in g.groupby(['day', 'side'], sort=True):
        r = {c: x[x.candidate == c].iloc[0] for c in CANDS}
        body.append([day, SIDE[s], f_(r['SMOOTH'].oracle, 3), f_(r['SMOOTH'].always1, 3), f_(r['SMOOTH'].value, 3), f_(r['RICH'].value, 3), f_(r['SMOOTH'].regret, 3),
                     f_(r['SMOOTH'].missed_positive, 1), f_(r['SMOOTH'].admitted_negative, 1), f_(r['RICH'].admitted_negative, 1), f_(r['SMOOTH'].admitted_negative_tail, 1),
                     f_(r['SMOOTH'].tail_events)])
    print(table(['day', 'side', 'oracle', 'always POST', 'SMOOTH', 'RICH', 'SMOOTH regret', 'SMOOTH missed + (sum w)', 'SMOOTH admitted − (sum w)',
                 'RICH admitted − (sum w)', 'SMOOTH admitted − with |Y| > 30 c', '|Y| > 30 c events'], body) + '\n')


def predicates():
    pr = pd.read_csv(EV / 'predicates.csv')
    print('### Declared predicates (per day / side; lower MSE alone is not success; zero admissions = no demonstrated entry skill)\n')
    print(table(['day', 'side', 'candidate', 'admissions', 'status', 'selected POST value > 0', 'screen', 'above draw mean', 'above draw p95', 'realized',
                 'draw mean', 'draw p95'],
                [[r.day, SIDE[r.side], r.candidate, f_(r.admissions), r.status, str(r.post_value_positive).lower(), r.screen, str(r.above_draw_mean).lower(),
                  str(r.above_draw_p95).lower(), f_(r.realized, 4), f_(r.draw_mean, 4), f_(r.draw_p95, 4)] for _, r in pr.sort_values(['day', 'side', 'candidate']).iterrows()]) + '\n')


def screens():
    sc = pd.read_csv(EV / 'screens.csv')
    print('### Plan v2 screens (POST / WAIT groups)\n')
    print(table(['day', 'side', 'candidate', 'group', 'verdict', 'originals', 'meaningful nonzero', 'exact nonzero', 'mean pred', 'mean Y', 'why'],
                [[r.day, SIDE[r.side], r.candidate, r.group, r.verdict, f_(r.originals), f_(r.meaningful_originals), f_(r.exact_nonzero_originals), f_(r.wmean_pred, 3),
                  f_(r.wmean_y, 3), r.why] for _, r in sc.sort_values(['candidate', 'day', 'side', 'group']).iterrows()]) + '\n')


def calibration():
    cal = pd.read_csv(EV / 'calibration.csv')
    print('### Calibration, pooled over days and sides by each bin\'s weight (review 371aab56 item 5: w_sum pooling)\n')
    body = []
    for c in CANDS:
        x = cal[cal.candidate == c]
        for b in ORDER:
            z = x[x.bin == b]
            if len(z):
                W = z.w_sum.sum(); body.append([c, b, f_(z.rows.sum()), f_(W, 1), f_((z.w_sum * z.wmean_pred).sum() / W, 3), f_((z.w_sum * z.wmean_y).sum() / W, 3), f_(z.nonzero.sum())])
    print(table(['candidate', 'prediction bin', 'rows', 'weight', 'weighted mean pred', 'weighted mean Y', 'nonzero labels'], body) + '\n')
    r3 = pd.read_csv(R3_PRED, float_precision='round_trip'); r3 = r3[(r3['head'] == 'post') & (r3.support == 'supported')]
    import importlib.util
    spec = importlib.util.spec_from_file_location('ae', '/home/yuqing/ctaNew/predictfun/research/pf_adv2_eval.py'); sys.path.insert(0, '/home/yuqing/ctaNew/predictfun/research')
    ae = importlib.util.module_from_spec(spec); spec.loader.exec_module(ae)
    r3 = r3.assign(bin=r3.pred_RICH.map(ae.cbin)); body = []
    for b in ORDER:
        z = r3[r3.bin == b]
        if len(z):
            body.append([b, f_(len(z)), f_(z.w.sum(), 1), f_((z.w * z.pred_RICH).sum() / z.w.sum(), 3), f_((z.w * z.Y).sum() / z.w.sum(), 3)])
    print("R3's RICH POST calibration recomputed with sampling-weight pooling from R3's predictions (supersedes the R3 receipt's pooled table, "
          'which averaged weighted cell means by row counts):\n')
    print(table(['prediction bin', 'rows', 'weight', 'weighted mean pred', 'weighted mean Y'], body) + '\n')


def strata():
    st = pd.read_csv(EV / 'strata.csv')
    print('### Strata (both days and sides pooled: counts summed, values weight-pooled; no sparse-cell skill claim)\n')
    body = []
    for (dim, val), x in st.groupby(['dim', 'value'], sort=True):
        W = x.w_sum.sum(); body.append([dim, val, f_(x.rows.sum()), f_(x.meaningful_originals.sum()), f_(x.posts_SMOOTH.sum()), f_(x.posts_RICH.sum()),
                                         f_((x.w_sum * x.value_SMOOTH).sum() / W, 3), f_((x.w_sum * x.value_RICH).sum() / W, 3), f_((x.w_sum * x.wmean_y).sum() / W, 3)])
    print(table(['dimension', 'value', 'rows', 'meaningful originals', 'SMOOTH posts', 'RICH posts', 'SMOOTH value', 'RICH value', 'mean Y'], body) + '\n')


def control():
    ct = pd.read_csv(EV / 'control.csv')
    print('### Local action-information control (500 draws, seed 20261121; mandatory POST anchors)\n')
    print(table(['day', 'side', 'candidate', 'anchors', 'strata', 'singleton', 'no variation', 'realized', 'draw mean', 'lift', 'draw p05', 'draw p95', 'share ≥ realized'],
                [[r.day, SIDE[int(r.side)] if pd.notna(r.side) else 'both', r.candidate, f_(r.anchors), f_(r.strata), f_(r.singleton_strata), f_(r.no_variation_strata),
                  f_(r.realized, 4), f_(r.draw_mean, 4), f_(r.lift, 4), f_(r.draw_p05, 4), f_(r.draw_p95, 4), f_(r.share_draws_ge_realized, 3)]
                 for _, r in ct.sort_values(['day', 'candidate', 'side'], na_position='last').iterrows()]) + '\n')


def replay():
    E = pd.read_csv(RP / 'episodes.csv'); J = json.loads((RP / 'replay.json').read_text())
    print('### Replay: episode totals per policy (cents; each world valued once; 5 s episodes; R3\'s 384-world manifest)\n')
    body = []
    for (day, p), x in E.groupby(['day', 'policy'], sort=True):
        fs = x.filled_shares.sum()
        body.append([day, p, f_(len(x)), f_(x.value.sum(), 1), f_(x.value_bid.sum(), 1), f_(x.value_ask.sum(), 1), f_(x.fills.sum()), f_(fs, 0), f_(x.generations.sum()),
                     f_(x.policy_pulls.sum()), f_((x.generations == 0).sum()), f_((x.fills == 0).sum()), f_(x.loss.sum(), 1), f_(x.severe.sum()),
                     f_(x.r3_match.sum()) if 'r3_match' in x and x.r3_match.notna().any() else '—'])
    print(table(['day', 'policy', 'worlds', 'value', 'bid value', 'ask value', 'fills', 'filled shares', 'generations', 'policy pulls', 'no entry', 'no fill',
                 'loss (sum)', 'V < −30 c', 'equal to R3 (controls)'], body) + '\n')
    D = pd.DataFrame(J['diffs'])
    print('Paired differences (per world, same tape; SMOOTH − ENTRY_ONLY is the primary comparison):\n')
    print(table(['day', 'policy', 'versus', 'worlds', 'sum of differences', 'worlds above', 'worlds below'],
                [[r.day, r.policy, r.versus, f_(r.n), f_(r.diff_sum, 1), f_(r.positive), f_(r.negative)] for _, r in D.sort_values(['day', 'versus', 'policy']).iterrows()]) + '\n')
    print('### Replay: quoting, retention, inventory and markouts (sums over worlds; ms of the 5,000 ms episodes)\n')
    body = []
    for (day, p), x in E.groupby(['day', 'policy'], sort=True):
        n = len(x) * 5000.
        body.append([day, p, f_(x.quote_ms_bid.sum() / n, 3), f_(x.quote_ms_ask.sum() / n, 3), f_(x.quote_ms_both.sum() / n, 3), f_(x.first_review_pulls.sum()),
                     f_(x.reviewed_generations.sum()), f_(x.inv_peak.max()), f_((x.position.abs() > 1e-9).sum()), f_(x.m0.sum(), 1), f_(x.m100.sum(), 1), f_(x.m1000.sum(), 1),
                     f_(x.missing100.sum()), f_(x.missing1000.sum())])
    print(table(['day', 'policy', 'bid quoted', 'ask quoted', 'both', 'first-review pulls', 'reviewed generations', 'peak |inventory|', 'terminal inventory worlds',
                 'markout fill', 'markout +100 ms', 'markout +1 s', 'missing +100 ms', 'missing +1 s'], body) + '\n')


def churn():
    p = RP / 'churn.csv'
    if not p.exists() or p.stat().st_size < 2:
        return
    Ch = pd.read_csv(p)
    print('### First live reviews (each reviewed generation; the POST advantage at the post and the KEEP advantage at the review are DIFFERENT '
          'targets, not one forecast reversing; review 371aab56 item 2)\n')
    body = []
    for (day, pol, a), x in Ch.groupby(['day', 'policy', 'action'], sort=True):
        dm = x.d_pf_mid.dropna()
        body.append([day, pol, a, f_(len(x)), f_(x.dt.mean(), 1), f_(x.value_post.mean(), 3), f_(x.value_review.mean(), 3), f_((dm != 0).sum()), f_(dm.abs().mean(), 3),
                     f_(x.d_fp_queue.mean(), 2)])
    print(table(['day', 'policy', 'first-review action', 'generations', 'mean ms post → review', 'mean POST advantage at the post', 'mean retention value at the review',
                 'mid moved (count)', 'mean |Δ mid|', 'mean Δ queue ahead'], body) + '\n')


def exposure():
    p = RP / 'exposure.csv'
    if not p.exists() or p.stat().st_size < 2:
        return
    X_ = pd.read_csv(p); X_ = X_[X_.policy == 'SMOOTH']; parts = X_.key.str.split('|')
    X_ = X_.assign(state=parts.str[0], status=parts.str[1], dv=parts.str[2]); m_ = X_[(X_.state == 'absent') & (X_.status == 'model') & X_.dv.notna()].copy()
    m_['dim'] = m_.dv.str.split('=').str[0]; m_['value'] = m_.dv.str.split('=', n=1).str[1]; body = []
    for (dim, val), x in m_[m_.dim.isin(['c_gen', 'c_other'])].groupby(['dim', 'value'], sort=True):
        n_all = m_[m_.dim == dim].n.sum(); body.append([dim, val, f_(x.n.sum()), f_(x.n.sum() / n_all, 3) if n_all else '—'])
    print('### SMOOTH\'s own entry decisions in the replay by generation and the other side\'s state (callback exposure, not independent samples)\n')
    print(table(['dimension', 'value', 'SMOOTH absent model decisions', 'share'], body) + '\n')


def reached():
    p = RP / 'reached.csv'
    if not p.exists() or p.stat().st_size < 2:
        return
    R = pd.read_csv(p)
    print('### Reached SMOOTH entry states (frozen pi0 arms): both POST heads on identical inputs at the same states\n')
    body = []
    for day, x in R[R.status == 'ok'].groupby('day', sort=True):
        y = x.Y.values
        body.append([day, f_(len(x)), f_(int((y != 0).sum())), f_(y.mean(), 3), f_(np.mean((x.pred_SMOOTH.values - y) ** 2), 2), f_(np.mean((x.pred_RICH.values - y) ** 2), 2),
                     f_(np.mean(y ** 2), 2), f_(x.pred_SMOOTH.mean(), 3), f_(x.pred_RICH.mean(), 3), f_((x.pred_SMOOTH > 0).sum()), f_((x.pred_RICH > 0).sum())])
    print(table(['day', 'pairs', 'nonzero', 'mean Y', 'MSE SMOOTH', 'MSE RICH', 'MSE zero', 'mean SMOOTH prediction', 'mean RICH prediction', 'SMOOTH > 0', 'RICH > 0'], body) + '\n')
    st = R.groupby(['day', 'status']).size().reset_index(name='n')
    print(table(['day', 'status', 'pairs'], [[r.day, r.status, f_(r.n)] for _, r in st.iterrows()]) + '\n')


if __name__ == '__main__':
    checks(); identity(); errors(); values(); predicates(); screens(); calibration(); strata(); control(); replay(); churn(); exposure(); reached()
