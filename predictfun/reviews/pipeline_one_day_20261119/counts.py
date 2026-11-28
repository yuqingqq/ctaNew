"""The counts the one-day report quotes, computed from the saved benchmark predictions and the policy table (no number typed by hand)."""
import sys
import pandas as pd
R = sys.argv[1] if len(sys.argv) > 1 else 'data/predict_fun/research'
P = pd.read_csv(f'{R}/pipeline_policy_20261119/policy_20261027.csv'); B = 'M50|A58|L37|N100'
a, m = P[P.rule == 'A+0'].set_index('scenario'), P[P.rule == 'MODEL-J'].set_index('scenario'); n = len(m)
print(f'scenarios {n}; MODEL-J value above A+0 in {(m.V > a.V).sum()}/{n} (not at {", ".join(m.index[m.V <= a.V])}); above PULL-ALL in {(m.V > m.PULLALL_value).sum()}/{n} '
      f'({", ".join(m.index[m.V > m.PULLALL_value])}); above KEEP in {(m.V > m.KEEP).sum()}/{n}; MODEL-J E > 0 in {(m.E > 0).sum()}/{n}; A+0 E > 0 in {(a.E > 0).sum()}/{n}')
print(f'at B (c): KEEP {m.loc[B].KEEP:+,.0f}; A+0 {a.loc[B].V:+,.0f}; MODEL-J {m.loc[B].V:+,.0f}; PULL-ALL {m.loc[B].PULLALL_value:+,.0f}; MODEL-J - PULL-ALL {m.loc[B].V - m.loc[B].PULLALL_value:+,.0f}; '
      f'pulls A+0 {int(a.loc[B].pulls):,}, MODEL-J {int(m.loc[B].pulls):,}')
pr = pd.read_parquet(f'{R}/pipeline_bench_20261119/pred_20261027.parquet', columns=['kind', 'oid', 'j', 'J_V', 'value_m50'])
r = pr[pr.kind == 'retain']; hit = r[r.J_V < 0].groupby('oid').j.min(); k = r.oid.nunique()
print(f'RETAIN decisions {len(r):,} of {k:,} orders; predicted value below 0 in {100 * (r.J_V < 0).mean():.1f}%; orders pulled {len(hit):,} ({100 * len(hit) / k:.1f}%), '
      f'at the first live step {int((hit == 2).sum()):,} ({100 * (hit == 2).mean():.1f}% of the pulled)')
po = pr[pr.kind == 'post']; print(f'POST rows {len(po):,}; predicted POST value above 0 in {100 * (po.J_V > 0).mean():.1f}%; realized mean value of those {po[po.J_V > 0].value_m50.mean():+.2f} c '
                                  f'({int((po.J_V > 0).sum()):,} orders), of all {po.value_m50.mean():+.2f} c')
