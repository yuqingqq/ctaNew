"""R0 (plan v2) / review 0026ab85: the paired pipeline and its evaluation, v2 against v1, declared before v2 ran. v2 reuses v1's models (no refit),
rebases pair_qm to the current decision's mid and makes the reached rows admission-aware, so:
 (1) every model file is v1's, byte for byte;
 (2) every label column v1 and v2 share is identical on every row except pair_qm; on every v2 row pair_v = sign * (pf_mid - p) * pair_q + pair_qm
     within 1e-6 c; the v1 rows failing that identity are counted per day and kind beside the review's counts;
 (3) every prediction column is identical;
 (4) the POST diagnostics are unchanged; the reached rows of A shrink to its admitted orders (the review: 126,063 -> 67,776 on Oct 26, 144,060 ->
     76,731 on Oct 27, summed over sides), with the corrected groups and gates printed;
 (5) every evaluation value (policy x side x scenario) is identical, and only the reached-state shift changes.
Reads one day at a time. Usage (run_light): compare_v1_v2.py [research data dir]"""
import json
import sys
import numpy as np
import pandas as pd
R = sys.argv[1] if len(sys.argv) > 1 else 'data/predict_fun/research'
P1, P2, E1_, E2_ = (f'{R}/{x}' for x in ('pipeline_paired_20261119', 'pipeline_paired_v2_20261119', 'pipeline_paired_eval_20261119', 'pipeline_paired_eval_v2_20261119'))
j1, j2 = json.loads(open(f'{P1}/paired.json').read()), json.loads(open(f'{P2}/paired.json').read())
ok = {'MODELS IDENTICAL': j1['models'] == j2['models'] and len(j1['models']) == 44 and j2['reuse'] is not None}
review_qm = {'20261024': (1338, 22626), '20261025': (957, 16793), '20261026': (942, 13808), '20261027': (747, 10568)}
lab_same, ident, rev_ok = True, True, True
for d in ('20261024', '20261025', '20261026', '20261027'):
    a, b = pd.read_parquet(f'{P1}/labels_{d}.parquet'), pd.read_parquet(f'{P2}/labels_{d}.parquet')
    common = [c for c in a.columns if c in b.columns and c != 'pair_qm']
    same = len(a) == len(b) and all((a[c].astype(str).values == b[c].astype(str).values).all() if a[c].dtype.kind in 'OSUb' or str(a[c].dtype) == 'category'
                                    else np.array_equal(a[c].values, b[c].values, equal_nan=True) for c in common)
    sg = np.where(b.side.values == 0, 1., -1.); edge = sg * (b.pf_mid.values.astype(float) - b.p.values.astype(float))
    idn2 = np.abs(b.pair_v.values - (edge * b.pair_q.values + b.pair_qm.values)); idn1 = np.abs(a.pair_v.values - (edge * a.pair_q.values + a.pair_qm.values))
    kinds = b.kind.astype(str).values; mis = tuple(int(((idn1 > 1e-6) & (kinds == k)).sum()) for k in ('post', 'retain')); chg = int((np.abs(a.pair_qm.values - b.pair_qm.values) > 1e-6).sum())
    print(f'{d}: {len(b):,} rows; shared label columns identical except pair_qm: {same}; v2 identity max |error| {idn2.max():.2e} c; v1 rows failing it (POST, RETAIN) '
          f'{mis}, the review {review_qm[d]}; rows whose pair_qm changed {chg:,}')
    lab_same &= same; ident &= bool((idn2 <= 1e-6).all()); rev_ok &= mis == review_qm[d]
ok['LABELS IDENTICAL EXCEPT pair_qm'] = lab_same; ok['v2 VALUE IDENTITY ON EVERY ROW'] = ident; ok['v1 IDENTITY FAILURES = THE REVIEW COUNTS'] = rev_ok
pred_same = True
for d in ('20261026', '20261027'):
    a, b = pd.read_parquet(f'{P1}/pred_{d}.parquet'), pd.read_parquet(f'{P2}/pred_{d}.parquet')
    cols = ['pi0_V', 'pi0_ok', 'NEU_V', 'NEU_ok', 'A_V', 'AF_V', 'AM_V', 'pair_v', 'pair_q']
    s_ = all(np.array_equal(a[c].values, b[c].values, equal_nan=True) for c in cols) and len(a) == len(b); pred_same &= s_
    print(f'{d}: predictions {cols} identical: {s_}')
ok['PREDICTIONS IDENTICAL'] = pred_same
rv = {'20261026': (126063, 67776), '20261027': (144060, 76731)}; rch_ok = True
for d in ('20261026', '20261027'):
    for s in ('0', '1'):
        for h in ('A', 'AF', 'AM'):
            x1, x2 = j1['diagnostics'][d][s][h], j2['diagnostics'][d][s][h]
            post_same = x1['post_rank'] == x2['post_rank'] and x1['groups']['POST selected'] == x2['groups']['POST selected']
            print(f"{d} side {s} {h:<2}: POST unchanged {post_same}; reached rows {x1['reached_rows']:,} -> {x2['reached_rows']:,} (admitted orders {x2['admitted_orders']:,}); "
                  f"gate {x1['gate']} -> {x2['gate']} {x2['gate_groups']}; " + '; '.join(f"{k}: n {v['n']:,} predicted {v['predicted']:+,.1f} realized {v['realized']:+,.1f}" for k, v in x2['groups'].items()))
            rch_ok &= post_same
    got = tuple(sum(j[ 'diagnostics'][d][s]['A']['reached_rows'] for s in ('0', '1')) for j in (j1, j2))
    print(f'{d}: A reached rows (both sides) v1 -> v2 {got}, the review {rv[d]}'); rch_ok &= got == rv[d]
ok['POST DIAGNOSTICS UNCHANGED, A REACHED = THE REVIEW'] = rch_ok
ev_same = True
for d in ('20261026', '20261027'):
    a, b = json.loads(open(f'{E1_}/eval_{d}.json').read()), json.loads(open(f'{E2_}/eval_{d}.json').read())
    same = a['values'] == b['values'] and a['null_A'] == b['null_A'] and a['admissions'] == b['admissions'] and a['pulled'] == b['pulled']
    rd = {k: (a['readings'].get(k), b['readings'].get(k)) for k in set(a['readings']) | set(b['readings']) if a['readings'].get(k) != b['readings'].get(k)}
    print(f"{d}: evaluation values, null, admissions and pulls identical: {same}; readings that differ: {rd}; shift v1 {a['shift']} -> v2 {b['shift']}")
    ev_same &= same
ok['EVALUATION VALUES IDENTICAL'] = ev_same
for k, v in ok.items():
    print(f'{k}: {v}')
print(f'R0 RECEIPT CHECKS ALL TRUE: {all(ok.values())}')
