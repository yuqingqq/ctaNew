"""The counts the v2 report quotes, computed from the saved policy tables and benchmark predictions of both days (no number typed by hand)."""
import sys
import pandas as pd
R = sys.argv[1] if len(sys.argv) > 1 else 'data/predict_fun/research'; B = 'M50|A58|L37|N100'
for day in ('20261026', '20261027'):
    P = pd.read_csv(f'{R}/pipeline_policy_v2_20261119/policy_{day}.csv'); a, m = P[P.rule == 'A+0'].set_index('scenario'), P[P.rule == 'MODEL-J'].set_index('scenario'); n = len(m)
    up = m.V > m.PULLALL_value
    print(f'{day}: MODEL-J value above PULL-ALL in {up.sum()}/{n} ({", ".join(m.index[up])}); above A+0 in {(m.V > a.V).sum()}/{n}; above KEEP in {(m.V > m.KEEP).sum()}/{n}; '
          f'E > 0 in {(m.E > 0).sum()}/{n}; at B: KEEP {m.loc[B].KEEP:+,.0f}, A+0 {a.loc[B].V:+,.0f}, MODEL-J {m.loc[B].V:+,.0f}, PULL-ALL {m.loc[B].PULLALL_value:+,.0f}, '
          f'MODEL-J - PULL-ALL {m.loc[B].V - m.loc[B].PULLALL_value:+,.0f}; MODEL-J - PULL-ALL summed over the 19 scenarios {(m.V - m.PULLALL_value).sum():+,.0f}')
    pr = pd.read_parquet(f'{R}/pipeline_bench_v2_20261119/pred_{day}.parquet', columns=['kind', 'oid', 'j', 'J_V', 'J_ok', 'value_m50'])
    r = pr[pr.kind == 'retain']; acts = r.J_ok.astype(bool) & (r.J_V < 0); hit = r[acts].groupby('oid').j.min(); k = r.oid.nunique()
    print(f'   RETAIN decisions {len(r):,} of {k:,} orders; coherent {100 * r.J_ok.mean():.2f}%; predicted loss among coherent {100 * (r[r.J_ok.astype(bool)].J_V < 0).mean():.1f}%; '
          f'orders pulled {len(hit):,} ({100 * len(hit) / k:.1f}%), at the first live step {int((hit == 2).sum()):,} ({100 * (hit == 2).mean():.1f}%)')
    po = pr[pr.kind == 'post']; pos = po.J_ok.astype(bool) & (po.J_V > 0)
    print(f'   POST rows {len(po):,}; predicted positive {100 * pos.mean():.1f}% ({int(pos.sum()):,}); their realized mean {po[pos].value_m50.mean():+.2f} c; all {po.value_m50.mean():+.2f} c')
