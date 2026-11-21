"""Review-only summaries of independently reconciled saved ledgers; no new candidate."""
from pathlib import Path
import json
import pandas as pd
P=Path(__file__).resolve().parent
F=pd.read_csv(P/'independent_arms.csv',dtype={'day':str},float_precision='round_trip')
v=['end','q','mo1s','post1s_to_END','late_or_missing_END']
out={}
for day in sorted(F.day.unique())+['all']:
 x=F if day=='all' else F[F.day==day]
 a=x.groupby('arm')[v].sum();d=a.loc['guard']-a.loc['disabled']
 assert abs(d['end']-d['mo1s']-d['post1s_to_END']-d['late_or_missing_END'])<1e-6
 out[day]=dict(delta=d.to_dict(),baseline=a.loc['disabled'].to_dict(),guard=a.loc['guard'].to_dict())
err=(F.mean_abs_exact-F.mean_abs_grid).abs()
out['inventory_grid_error']=dict(max_mean_abs_shares=float(err.max()),max_band_fraction=float((F.band_fraction_exact-F.band_fraction_grid).abs().max()))
(P/'totals.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
