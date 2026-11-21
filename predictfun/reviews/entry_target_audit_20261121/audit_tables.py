"""ENTRY-TARGET-AUDIT-v1 receipt tables: markdown copies of the audit's outputs (and review d6d5d55e item 2's renamed R4E tables), so that every
number in the receipt is copied by code. A formatter: it computes no predicate; each number is a value read from an output, a sum,
difference or ratio of such values, or a count of rows.
Usage: audit_tables.py [data dir] -> markdown on stdout"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('/home/yuqing/ctaNew/data/predict_fun/research/entry_target_audit_v1_20261121')
R4E = Path('/home/yuqing/ctaNew/predictfun/reviews/r4e_entry_smooth_20261121')
SIDE = {0: 'bid', 1: 'ask', '0': 'bid', '1': 'ask'}
CLASSES = ('no_post_fork_fills', 'economic_identical', 'same_cash_inventory', 'offset_by_mark', 'meaningful')
SCORE = {('entry', 'SMOOTH'): 'A: SMOOTH POST − WAIT advantage', ('entry', 'RICH'): 'A: RICH POST − WAIT advantage', ('entry', 'PI0'): 'PI0 V_post',
         ('review', 'PI0'): 'PI0 V_retain', ('review', 'RICH'): 'A: RICH KEEP − PULL advantage', ('review', 'PA'): '— (pull at review)'}


def f_(x, nd=2):
    if x is None or (isinstance(x, float) and not np.isfinite(x)) or (isinstance(x, str) and x in ('', 'nan', 'None')):
        return '—'
    v = float(x)
    return f'{int(v):,}' if v.is_integer() and abs(v) < 1e15 else f'{v:,.{nd}f}'


def table(head, body):
    esc = lambda c: str(c).replace('|', '\\|')
    out = ['| ' + ' | '.join(esc(h) for h in head) + ' |', '|' + '---|' * len(head)]
    return '\n'.join(out + ['| ' + ' | '.join(esc(c) for c in r) + ' |' for r in body])


def rd(name):
    return pd.read_csv(OUT / name, float_precision='round_trip')


def checks():
    J = json.loads((HERE / 'checks.json').read_text())
    print('### Declared checks\n')
    print(table(['check', 'value'], [[k, str(v).lower()] for k, v in sorted(J['checks'].items())] + [['all_true', str(J['all_true']).lower()]]) + '\n')
    F = json.loads((HERE / 'checks_falsifier.json').read_text())
    print('Falsifier (clean copy, then one corruption each):\n')
    print(table(['check', 'clean', 'corrupted', 'applied'], [[k, str(v['clean']).lower(), str(v['corrupted']).lower(), str(v['applied']).lower()] for k, v in sorted(F['cases'].items())]
                + [['falsifier_ok', str(F['falsifier_ok']).lower(), '', '']]) + '\n')


def statuses():
    st = rd('statuses.csv')
    print('### Population: POST rows by set, day and status (every status kept)\n')
    print(table(['set', 'day', 'status', 'rows'], [[r.set, r.day, r.status, f_(r.n)] for _, r in st.iterrows()]) + '\n')


def taxonomy():
    G = rd('groups.csv')
    print('### Accounting classes (weighted shares; exact zeros and residues counted; Oct 24–25 in-sample, Oct 26–27 adaptive)\n')
    body = [[r.set, r.day, SIDE.get(r.side, r.side), r.pick, f_(r.rows), f_(r.w_sum, 1)] + [f_(r[f'share_{c}'], 3) for c in CLASSES]
            + [f_(r.meaningful), f_(r.positive), f_(r.negative), f_(r.exact_zero), f_(r.residue)] for _, r in G.iterrows()]
    print(table(['set', 'day', 'side', 'pick', 'rows', 'weight'] + list(CLASSES) + ['meaningful', 'positive', 'negative', 'exact zero', 'residue'], body) + '\n')


def components():
    G = rd('groups.csv'); G = G[G.pick == 'all']
    print('### The decomposition Y = V_initial + D_focal + D_opposite (cents; weighted means and absolute totals)\n')
    body = [[r.set, r.day, SIDE.get(r.side, r.side), f_(r.wmean_Y, 3), f_(r.wmean_v_initial, 3), f_(r.wmean_d_focal, 3), f_(r.wmean_d_opposite, 3), f_(r.abs_Y, 1),
             f_(r.abs_v_initial, 1), f_(r.abs_d_following, 1), f_(r.abs_Y / r.abs_v_initial, 3) if r.abs_v_initial else '—', f_(r.opposite_sign_share, 3), f_(r.top5_abs_share, 3)]
            for _, r in G.iterrows()]
    print(table(['set', 'day', 'side', 'mean Y', 'mean V_initial', 'mean D_focal', 'mean D_opposite', 'Σ w |Y|', 'Σ w |V_initial|', 'Σ w |D_following|',
                 'Σ|Y| / Σ|V_initial|', 'V_initial, D_following opposite signs', 'top-5 |Y| share'], body) + '\n')


def evidence():
    G = rd('groups.csv'); G = G[G.pick == 'all']
    print('### The forced generation\'s fills and the WAIT arm\'s next POST (weighted shares; no refusal or queue cause asserted)\n')
    body = [[r.set, r.day, SIDE.get(r.side, r.side), f_(r.init_no_fill, 3), f_(r.init_partial, 3), f_(r.init_full, 3), f_(r.wait_next_0_100, 3), f_(r.wait_next_100_500, 3),
             f_(r.wait_next_500_1000, 3), f_(r.wait_next_1000_inf, 3), f_(r.wait_no_later_post, 3)] for _, r in G.iterrows()]
    print(table(['set', 'day', 'side', 'initial: no fill', 'partial', 'full', 'WAIT next POST ≤ 100 ms', '100–500 ms', '500–1,000 ms', '> 1 s', 'no later POST'], body) + '\n')


def strata():
    S = rd('strata.csv')
    print('### Strata (weighted; sparse cells are counts, not claims)\n')
    body = [[r.set, r.dim, r.value, f_(r.rows), f_(r.share_meaningful, 3), f_(r.init_no_fill, 3), f_(r.wmean_v_initial, 3), f_(r.wmean_d_following, 3), f_(r.wmean_Y, 3)]
            for _, r in S.iterrows()]
    print(table(['set', 'dimension', 'value', 'rows', 'meaningful', 'initial no fill', 'mean V_initial', 'mean D_following', 'mean Y'], body) + '\n')


def crosstab():
    X = rd('crosstab.csv')
    print('### The frozen models\' actions against the components (cross-tab only; Oct 24–25 in-sample)\n')
    body = [[r.set, r.candidate, r.day, SIDE.get(r.side, r.side), r.action, f_(r.rows), f_(r.w_sum, 1), f_(r.wmean_Y, 3), f_(r.wmean_v_initial, 3), f_(r.wmean_d_following, 3),
             f_(r.share_meaningful, 3), f_(r.init_no_fill, 3)] for _, r in X.iterrows()]
    print(table(['set', 'model', 'day', 'side', 'action', 'rows', 'weight', 'mean Y', 'mean V_initial', 'mean D_following', 'meaningful', 'initial no fill'], body) + '\n')


def reached():
    Q = rd('reached_summary.csv') if (OUT / 'reached_summary.csv').exists() else None
    if Q is None or not len(Q):
        return
    print('### R4E\'s reached SMOOTH entry pairs (a world / side-balanced diagnostic; unweighted; never mixed with the sample)\n')
    body = [[r.day, f_(r.rows)] + [f_(r[f'n_{c}']) for c in CLASSES] + [f_(r.wmean_Y, 3), f_(r.wmean_v_initial, 3), f_(r.wmean_d_following, 3), f_(r.init_no_fill, 3)] for _, r in Q.iterrows()]
    print(table(['day', 'pairs'] + list(CLASSES) + ['mean Y', 'mean V_initial', 'mean D_following', 'initial no fill'], body) + '\n')


def predicates():
    js = json.loads((OUT / 'audit.json').read_text())
    print('### Declared predicates (thresholds: meaningful < 0.10 and no initial fill ≥ 0.50; Σ|Y| ≤ 0.5 Σ|V_initial|)\n')
    print(table(['set', 'meaningful share', 'initial no-fill share', 'Σ w |V_initial|', 'Σ w |Y|', 'rare_execution', 'cancellation', 'diagnosis'],
                [[k, f_(v['share_meaningful'], 3), f_(v['init_no_fill'], 3), f_(v['abs_v_initial'], 1), f_(v['abs_Y'], 1), str(v['rare_execution']).lower(), str(v['cancellation']).lower(),
                  v['diagnosis']] for k, v in sorted(js['predicates'].items())]) + '\n')


def r4e_names():
    ch = pd.read_csv(R4E / 'replay' / ('churn.csv.gz' if (R4E / 'replay' / 'churn.csv.gz').exists() else 'churn.csv'), float_precision='round_trip')
    print('### Review d6d5d55e item 2: R4E\'s first-review table with each leg\'s score and policy named\n')
    body = []
    for (day, pol, bp, br, a), x in ch.groupby(['day', 'policy', 'by_post', 'by_review', 'action'], sort=True, dropna=False):
        body.append([day, pol, bp, SCORE.get(('entry', bp), str(bp)), f_(x.value_post.mean(), 3), br, SCORE.get(('review', br), str(br)), f_(x.value_review.mean(), 3), a, f_(len(x))])
    print(table(['day', 'policy', 'entry leg', 'entry score', 'mean entry score', 'review leg', 'review score', 'mean review score', 'first-review action', 'generations'], body) + '\n')
    print('The two legs\' scores are different quantities (different targets, alternatives and horizons); their difference is not a forecast change.\n')
    pr = pd.read_csv(R4E / 'eval' / 'predicates.csv'); rdg = pd.read_csv(R4E / 'eval' / 'readings.csv'); body = []
    for _, r in pr.sort_values(['day', 'side', 'candidate']).iterrows():
        g = rdg[(rdg.day == r.day) & (rdg.side == r.side) & (rdg.candidate == r.candidate) & (rdg.group == 'POST')]
        body.append([r.day, SIDE[r.side], r.candidate, f_(r.admissions), f_(g.wmean_y.iloc[0], 4) if len(g) else '—', f_(r.realized, 4), f_(r.draw_mean, 4), f_(r.draw_p95, 4)])
    print('R4E\'s predicates with the two populations named (the control uses the mandatory anchors; the selected mean uses all sampled rows):\n')
    print(table(['day', 'side', 'candidate', 'admissions', 'all-sampled selected-POST weighted mean', 'mandatory-anchor control value', 'draw mean', 'draw p95'], body) + '\n')


if __name__ == '__main__':
    checks(); predicates(); statuses(); taxonomy(); components(); evidence(); strata(); crosstab(); reached(); r4e_names()
