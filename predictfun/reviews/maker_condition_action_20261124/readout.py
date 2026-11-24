"""Report-only rendering of every frozen comparison and sequence day/clock row."""
import json
from pathlib import Path
import pandas as pd
H=Path(__file__).resolve().parent
m=pd.read_csv(H/'outputs/overview.csv',float_precision='round_trip')
s=pd.read_csv(H/'outputs/sequence_contrast.csv',float_precision='round_trip')
print('ALL 64 CONDITION / ACTION COMPARISONS; ratios except retreat risk differences')
for r in m.itertuples():
    print(f'{r.layer:28} {r.action:24} {r.condition:14} effect={r.effect:+.4f} rates={100*r.exposed_rate:.2f}%/{100*r.reference_rate:.2f}% n={r.n1:g}/{r.n0:g} same_sign={r.same_direction_all_views} original={r.verdict} range={r.effect_min:+.4f}/{r.effect_max:+.4f}')
print('\nALL POOLED SEQUENCE CLOCK/PANEL ROWS')
print(s[s.day=='all'].to_csv(index=False))
print('\nALL D0 FULL SEQUENCE DAY ROWS')
print(s[(s.variant=='D0')&(s.panel=='full')].to_csv(index=False))
o1=s[(s.outcome=='O1')&(s.day!='all')]
print('O1 day signs (descriptive, no significance):',json.dumps({str(k):{'days':len(g),'positive':int((g.rd>0).sum()),'negative':int((g.rd<0).sum())} for k,g in o1.groupby(['variant','panel'])}))
