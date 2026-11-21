"""R3 ADV2-RICH v1 receipt tables: markdown copies of the round's outputs (checks.json, the evaluation CSVs, the replay CSVs / JSON), so
that every number in the receipt is copied by code. A formatter: it computes no predicate. Each number is a value read from an output, a
sum, difference or ratio of such values (a mean is a sum over a count), or a count of rows.
Usage: r3_tables.py [data dir] -> markdown on stdout"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('/home/yuqing/ctaNew/data/predict_fun/research/adv2_rich_v1_20261121')
EV, RP = OUT / 'eval', OUT / 'replay_v1'
SIDE = {0: 'bid', 1: 'ask'}


def f_(x, nd=2):
    if x is None or (isinstance(x, float) and not np.isfinite(x)) or (isinstance(x, str) and x in ('', 'nan', 'None')):
        return '—'
    v = float(x)
    return f'{int(v):,}' if v.is_integer() and abs(v) < 1e15 else f'{v:,.{nd}f}'


def table(head, body):
    esc = lambda c: str(c).replace('|', '\\|')                                          # a pipe inside a cell (|Y|, |inventory|) is escaped
    out = ['| ' + ' | '.join(esc(h) for h in head) + ' |', '|' + '---|' * len(head)]
    out += ['| ' + ' | '.join(esc(c) for c in r) + ' |' for r in body]
    return '\n'.join(out)


def checks():
    J1 = json.loads((HERE / 'checks.json').read_text()); v2 = HERE / 'checks_v2.json'; J = json.loads(v2.read_text()) if v2.exists() else J1
    print('### Declared checks\n')
    print(table(['check', 'v1 (r3_checks.py, frozen)', 'v2 (r3_checks_v2.py; G superseded)' if v2.exists() else '—'],
                [[k, str(J1['checks'][k]).lower(), str(J['checks'][k]).lower() if v2.exists() else ''] for k in sorted(J1['checks'])]
                + [['all_true', str(J1['all_true']).lower(), str(J['all_true']).lower() if v2.exists() else '']]) + '\n')
    fp = HERE / 'checks_falsifier.json'
    if fp.exists():
        F = json.loads(fp.read_text())
        print('Falsifier v1 (each check on a clean copy of its inputs, then after one corruption):\n')
        print(table(['check', 'clean copy', 'corrupted copy'], [[k, str(v['clean']).lower(), str(v['corrupted']).lower()] for k, v in sorted(F['cases'].items())]
                    + [['falsifier_ok', str(F['falsifier_ok']).lower(), '']]) + '\n')
    fp2 = HERE / 'checks_falsifier_v2.json'
    if fp2.exists():
        F2 = json.loads(fp2.read_text())
        print('Falsifier v2 (G2 on copies of the real parts; E and H as in v1):\n')
        print(table(['case', 'corruption applied', 'G2'], [[k, str(v['applied']).lower(), str(v['value']).lower()] for k, v in sorted(F2['cases'].items())]
                    + [['falsifier_ok', '', str(F2['falsifier_ok']).lower()]]) + '\n')
    return J


def errors():
    rd = pd.read_csv(EV / 'readings.csv'); g = rd[rd.group == 'all']
    print('### Prediction error against the baselines (all ok rows of the group; sampling-weighted; cents squared / cents)\n')
    body = []
    for (day, s, h), x in g.groupby(['day', 'side', 'head'], sort=True):
        r = {c: x[x.candidate == c].iloc[0] for c in ('RICH', 'FAIR', 'MECH')}; z = r['RICH']
        body.append([day, SIDE[s], h, f_(z.rows), f_(z.meaningful_originals), f_(z.wmse_zero), f_(z.wmse_train_mean), f_(r['RICH'].wmse), f_(r['FAIR'].wmse), f_(r['MECH'].wmse),
                     f_(r['RICH'].wmse / z.wmse_zero, 3), f_(r['RICH'].wmae), f_(z.wmae_zero), f_(r['RICH'].spearman, 3)])
    print(table(['day', 'side', 'head', 'rows', 'meaningful nonzero originals', 'MSE zero', 'MSE train mean', 'MSE RICH', 'MSE FAIR', 'MSE MECH',
                 'RICH / zero', 'MAE RICH', 'MAE zero', 'Spearman RICH'], body) + '\n')


def decisions():
    rd = pd.read_csv(EV / 'readings.csv'); g = rd[rd.group == 'all']
    print('### Local decision value per opportunity (cents; sampling-weighted; pi0-conditional, never a policy return)\n')
    body = []
    for (day, s, h), x in g.groupby(['day', 'side', 'head'], sort=True):
        r = {c: x[x.candidate == c].iloc[0] for c in ('RICH', 'FAIR', 'MECH')}
        body.append([day, SIDE[s], h, f_(r['RICH'].oracle, 3), f_(r['RICH'].always1, 3), f_(r['RICH'].value, 3), f_(r['FAIR'].value, 3), f_(r['MECH'].value, 3),
                     f_(r['RICH'].regret, 3), f_(r['RICH'].missed_positive, 1), f_(r['RICH'].admitted_negative, 1), f_(r['RICH'].tail_events)])
    print(table(['day', 'side', 'head', 'oracle', 'always POST / KEEP', 'RICH', 'FAIR', 'MECH', 'RICH regret', 'RICH missed + (sum w)', 'RICH admitted − (sum w)',
                 '|Y| > 30 c events'], body) + '\n')
    print('d = 1 means POST (post head) or KEEP (keep head); always-0 (never post / always pull) is 0 by definition.\n')


def screens():
    sc = pd.read_csv(EV / 'screens.csv')
    print('### Plan v2 prediction screens (per day / side / head / selected group)\n')
    body = [[r.day, SIDE[r.side], r['head'], r.candidate, r.group, r.verdict, f_(r.originals), f_(r.meaningful_originals), f_(r.exact_nonzero_originals),
             f_(r.wmean_pred, 3), f_(r.wmean_y, 3), r.why] for _, r in sc.sort_values(['candidate', 'day', 'side', 'head', 'group']).iterrows()]
    print(table(['day', 'side', 'head', 'candidate', 'group', 'verdict', 'originals', 'meaningful nonzero', 'exact nonzero', 'mean pred', 'mean Y', 'why'], body) + '\n')
    cnt = sc.groupby(['candidate', 'verdict']).size().unstack(fill_value=0)
    print('Verdict counts:\n')
    print(table(['candidate'] + list(cnt.columns), [[c] + [f_(v) for v in cnt.loc[c]] for c in cnt.index]) + '\n')


def calibration():
    cal = pd.read_csv(EV / 'calibration.csv'); x = cal[cal.candidate == 'RICH']
    order = ['(-inf, -10]', '(-10, -5]', '(-5, -2]', '(-2, -1]', '(-1, -0.5]', '(-0.5, 0)', '0 (exact)', '(0, 0.5]', '(0.5, 1]', '(1, 2]', '(2, 5]', '(5, 10]', '(10, inf]']
    print('### RICH fixed-bin calibration (supported rows; both days and sides pooled per head by row-weighted sums)\n')
    body = []
    for h in ('post', 'keep'):
        y = x[x['head'] == h]
        for b in order:
            z = y[y.bin == b]
            if not len(z):
                continue
            n = z.rows.sum(); wp = (z.wmean_pred * z.rows).sum() / n; wy = (z.wmean_y * z.rows).sum() / n
            body.append([h, b, f_(n), f_(z.nonzero.sum()), f_(wp, 3), f_(wy, 3)])
    print(table(['head', 'prediction bin', 'rows', 'nonzero labels', 'mean pred (row-weighted over strata)', 'mean Y (row-weighted over strata)'], body) + '\n')


def control():
    ct = pd.read_csv(EV / 'control.csv')
    print('### Local action-information control (500 draws, seed 20261121; anchors: mandatory POST and sampled KEEP picks)\n')
    body = []
    for _, r in ct.sort_values(['day', 'head', 'candidate'], na_position='last').iterrows():
        body.append([r.day, r['head'] if isinstance(r['head'], str) else 'both', r.candidate, f_(r.anchors), f_(r.strata), f_(r.singleton_strata), f_(r.no_variation_strata),
                     f_(r.realized, 4), f_(r.draw_mean, 4), f_(r.lift, 4), f_(r.draw_p05, 4), f_(r.draw_p95, 4), f_(r.share_draws_ge_realized, 3)])
    print(table(['day', 'head', 'candidate', 'anchors', 'strata', 'singleton', 'no variation', 'realized', 'draw mean', 'lift', 'draw p05', 'draw p95',
                 'share of draws ≥ realized'], body) + '\n')


def replay():
    E = pd.read_csv(RP / 'episodes.csv'); J = json.loads((RP / 'replay.json').read_text())
    print('### Replay: episode totals per policy (cents; each world valued once; 5 s episodes)\n')
    body = []
    for (day, p), x in E.groupby(['day', 'policy'], sort=True):
        fs = x.filled_shares.sum()
        body.append([day, p, f_(len(x)), f_(x.value.sum(), 1), f_(x.value_bid.sum(), 1), f_(x.value_ask.sum(), 1), f_(x.fills.sum()), f_(fs, 0), f_(x.generations.sum()),
                     f_(x.policy_pulls.sum()), f_(100. * x.effective_cancels.sum() / fs, 1) if fs else '—', f_(100. * x.generations.sum() / fs, 1) if fs else '—',
                     f_((x.generations == 0).sum()), f_((x.fills == 0).sum()), f_(x.loss.sum(), 1), f_(x.severe.sum()), f_((~x.cap_ok.astype(bool)).sum())])
    print(table(['day', 'policy', 'worlds', 'value', 'bid value', 'ask value', 'fills', 'filled shares', 'generations', 'policy pulls', 'cancels / 100 filled shares',
                 'generations / 100 filled shares', 'worlds with no entry', 'worlds with no fill', 'loss (sum)', 'V < −30 c', 'cap_ok false'], body) + '\n')
    print('Paired full-policy differences (per world, same tape):\n')
    D = pd.DataFrame(J['diffs'])
    body = [[r.day, r.policy, r.versus, f_(r.n), f_(r.diff_sum, 1), f_(r.positive), f_(r.negative)] for _, r in D.sort_values(['day', 'versus', 'policy']).iterrows()]
    print(table(['day', 'policy', 'versus', 'worlds', 'sum of differences', 'worlds above', 'worlds below'], body) + '\n')
    print('### Replay: quoting, retention and inventory (sums over worlds; ms of the 5,000 ms episodes)\n')
    body = []
    for (day, p), x in E.groupby(['day', 'policy'], sort=True):
        n = len(x) * 5000.
        body.append([day, p, f_(x.quote_ms_bid.sum() / n, 3), f_(x.quote_ms_ask.sum() / n, 3), f_(x.quote_ms_either.sum() / n, 3), f_(x.quote_ms_both.sum() / n, 3),
                     f_(x.best_ms_both.sum() / n, 3), f_(x.first_review_pulls.sum()), f_(x.reviewed_generations.sum()), f_(x.inv_integral.sum() / n, 2), f_(x.inv_peak.max()),
                     f_((x.position.abs() > 1e-9).sum()), f_(x.m100.sum(), 1), f_(x.missing1000.sum())])
    print(table(['day', 'policy', 'bid quoted', 'ask quoted', 'either', 'both', 'both at best', 'first-review pulls', 'reviewed generations', 'mean |inventory|',
                 'peak |inventory|', 'worlds with terminal inventory', 'markout +100 ms (sum)', 'missing +1 s endpoints'], body) + '\n')


def reached():
    p = RP / 'reached.csv'
    if not p.exists() or p.stat().st_size == 0:
        return
    R = pd.read_csv(p)
    print('### Reached-state labels (RICH\'s own states; frozen pi0 arms) against the predictions at those states\n')
    body = []
    for (day, h), x in R[R.status == 'ok'].groupby(['day', 'head'], sort=True):
        y = x.Y.values
        body.append([day, h, f_(len(x)), f_(int((y != 0).sum())), f_(y.mean(), 3)] + [f_(np.mean((x[f'pred_{c}'].values - y) ** 2), 2) for c in ('RICH', 'FAIR', 'MECH')]
                    + [f_(np.mean(y ** 2), 2), f_(x.pred_RICH.mean(), 3)])
    print(table(['day', 'head', 'pairs', 'nonzero', 'mean Y', 'MSE RICH', 'MSE FAIR', 'MSE MECH', 'MSE zero', 'mean RICH prediction'], body) + '\n')
    st = R.groupby(['day', 'head', 'status']).size().reset_index(name='n')
    print(table(['day', 'head', 'status', 'pairs'], [[r.day, r['head'], r.status, f_(r.n)] for _, r in st.iterrows()]) + '\n')


def coverage():
    tr, ev = pd.read_csv(EV / 'coverage_train.csv'), pd.read_csv(EV / 'coverage.csv')
    print('### Coverage by support dimension (rows and meaningful-nonzero originals; both days and both sides summed; evaluation groups = RICH\'s '
          'selected action)\n')
    for h, (a1, a2) in (('post', ('POST', 'WAIT')), ('keep', ('KEEP', 'PULL'))):
        t_, e_ = tr[(tr['head'] == h) & (tr.group == 'all')], ev[ev['head'] == h]; body = []
        for (dim, val), x in t_.groupby(['dim', 'value'], sort=True):
            def g(grp, col):
                z = e_[(e_.group == grp) & (e_.dim == dim) & (e_.value.astype(str) == str(val))]
                return f_(z[col].sum()) if len(z) else '0'
            body.append([dim, val, f_(x.rows.sum()), f_(x.meaningful_originals.sum()), g('all', 'rows'), g('all', 'meaningful_originals'), g(a1, 'rows'), g(a2, 'rows'), g('fallback', 'rows')])
        for (dim, val), x in e_[e_.group == 'all'].groupby(['dim', 'value'], sort=True):
            if not len(t_[(t_.dim == dim) & (t_.value.astype(str) == str(val))]):
                body.append([dim, val, '0', '0', f_(x.rows.sum()), f_(x.meaningful_originals.sum()), '', '', ''])
        print(f'{h} head:\n')
        print(table(['dimension', 'value', 'train rows (Oct 24-25)', 'train meaningful originals', 'eval rows (Oct 26-27)', 'eval meaningful originals', f'eval {a1}',
                     f'eval {a2}', 'eval fallback'], body) + '\n')


def churn():
    p = RP / 'churn.csv'
    if not p.exists() or p.stat().st_size < 2:
        return
    Ch = pd.read_csv(p)
    print('### Churn: each reviewed generation\'s POST and first live review (recorded values and changes; means over generations)\n')
    body = []
    for (day, pol, a), x in Ch.groupby(['day', 'policy', 'action'], sort=True):
        dm = x.d_pf_mid.dropna()
        body.append([day, pol, a, f_(len(x)), f_(x.dt.mean(), 1), f_(x.value_post.mean(), 3), f_(x.value_review.mean(), 3), f_((dm != 0).sum()), f_(dm.abs().mean(), 3),
                     f_(x.d_fair_edge.mean(), 3), f_(x.d_fp_queue.mean(), 2), f_(x.d_fp_remaining.mean(), 2), f_(x.d_pf_spread.mean(), 3)])
    print(table(['day', 'policy', 'first-review action', 'generations', 'mean ms post → review', 'mean value at post', 'mean value at review', 'mid moved (count)',
                 'mean |Δ mid|', 'mean Δ fair_edge', 'mean Δ queue ahead', 'mean Δ remaining', 'mean Δ spread'], body) + '\n')
    print('value = the deciding policy\'s prediction (candidates) or the teacher\'s V (PI0 decisions); ENTRY_PA has none at review.\n')


def exposure():
    p = RP / 'exposure.csv'
    if not p.exists() or p.stat().st_size < 2:
        return
    X_ = pd.read_csv(p); X_ = X_[X_.policy == 'RICH']; X_['parts'] = X_.key.str.split('|'); X_['state'] = X_.parts.str[0]; X_['status'] = X_.parts.str[1]
    X_['dv'] = X_.parts.str[2]
    top = X_[X_.dv.isna()].groupby(['day', 'state', 'status']).n.sum().reset_index()
    print('### RICH\'s own-state exposure in the replay: callbacks by state and status\n')
    print(table(['day', 'state', 'status', 'callbacks'], [[r.day, r.state, r.status, f_(r.n)] for _, r in top.iterrows()]) + '\n')
    tr = pd.read_csv(EV / 'coverage_train.csv'); tr = tr[tr.group == 'all']
    m_ = X_[(X_.status == 'model') & X_.dv.notna()].copy(); m_['dim'] = m_.dv.str.split('=').str[0]; m_['value'] = m_.dv.str.split('=', n=1).str[1]
    body = []
    for (st, dim, val), x in m_.groupby(['state', 'dim', 'value'], sort=True):
        h = 'post' if st == 'absent' else 'keep'; n_all = m_[(m_.state == st) & (m_.dim == dim)].n.sum()
        t_ = tr[(tr['head'] == h) & (tr.dim == dim)]; t_v = t_[t_.value.astype(str) == str(val)].rows.sum(); t_all = t_.rows.sum()
        body.append([st, dim, val, f_(x.n.sum()), f_(x.n.sum() / n_all, 3) if n_all else '—', f_(t_v), f_(t_v / t_all, 3) if t_all else '—'])
    print('RICH model decisions by support dimension against the training rows of the same head (absent = post, live = keep):\n')
    print(table(['state', 'dimension', 'value', 'RICH decisions', 'share', 'training rows', 'training share'], body) + '\n')


if __name__ == '__main__':
    checks(); errors(); decisions(); screens(); calibration(); coverage(); control(); replay(); churn(); exposure(); reached()
