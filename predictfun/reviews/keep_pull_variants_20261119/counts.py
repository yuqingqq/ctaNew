"""Counts the report quotes, computed from the run's saved CSV tables (pf_keep_pull_variants.py, Oct 23) - no number is typed by hand."""
import sys
import pandas as pd
R = sys.argv[1] if len(sys.argv) > 1 else 'data/predict_fun/research/keep_pull_variants_20261119'
P = pd.read_csv(f'{R}/variants_20261023_policies.csv'); Q = pd.read_csv(f'{R}/variants_20261023_compare.csv'); B, M250 = 'M50|A58|L37|N100', 'M250|A58|L37|N100'
n = P.scenario.nunique(); print(f'scenarios {n}')
for p, g in P.groupby('policy', sort=False):
    b = g[g.scenario == B].iloc[0]; m = g[g.scenario == M250].iloc[0]
    print(f"{p:<5} E>0 in {(g.E > 0).sum()}/{n}; V>0 in {(g.V > 0).sum()}/{n}; V>KEEP in {(g.score > 0).sum()}/{n}; V>V(PULL-ALL) in {(g.score > g.PULLALL).sum()}/{n}; "
          f"at B: E {b.E:+,.0f} V {b.V:+,.0f} pulls {b.pulls:,}; at M250: E {m.E:+,.0f} V {m.V:+,.0f}; negative-E scenarios: {', '.join(g[g.E <= 0].scenario.str.split('|').str[0] + '/' + g[g.E <= 0].scenario.str.split('|').str[1:].str.join('|'))}")
for c, g in Q.groupby('candidate', sort=False):
    print(f"{c:<5} dV>0 in {(g.dV > 0).sum()}/{n} (min {g.dV.min():+,.0f}, max {g.dV.max():+,.0f}; dV<=0 at {', '.join(g[g.dV <= 0].scenario)}); dE>0 in {(g.dE > 0).sum()}/{n}; dV_ny>0 in {(g.dV_ny > 0).sum()}/{n}")
