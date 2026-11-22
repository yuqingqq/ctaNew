"""Describe the predeclared tables; no horizon or condition selection."""
from pathlib import Path
import json,re
import pandas as pd
R=Path('/home/yuqing/ctaNew');O=Path(__file__).resolve().parent
H=pd.read_csv(O/'horizons.csv');T=pd.read_csv(O/'transitions.csv');C=pd.read_csv(O/'conditions.csv')
f=H[(H.arm=='FULL')&(H.cohort=='common')&(H.day=='all')&(H.side=='all')].set_index('horizon_ms')
out=dict(common_full_fills=int(f.loc['1000','valid']),common_full_shares=float(f.loc['1000','q']),
         full_common_per_share={k:float(f.loc[k,'per_share']) for k in ('0','50','100','1000','5000','30000','END')},
         full_5s_net_below_1s=bool(f.loc['5000','net']<f.loc['1000','net']),full_30s_net_below_1s=bool(f.loc['30000','net']<f.loc['1000','net']))
g=H[(H.arm=='FULL')&(H.cohort=='common')&(H.day!='all')&(H.side=='all')]
out['negative_days_by_horizon']={k:int((g[g.horizon_ms==k].net<0).sum()) for k in ('1000','5000','30000')}
out['recovery_full']={str(int(z.horizon_ms)):dict(fraction_of_1s_negative_fills=z.recovered_fills/z.negative_1s_fills,
 fraction_of_1s_negative_shares=z.recovered_shares/z.negative_1s_shares,fraction_of_original_1s_loss=z.recovered_original_loss_cents/z.original_loss_cents) for z in T[T.arm=='FULL'].itertuples()}
q=C[(C.condition=='pressure_flow')&(C.day=='all')&(C.side=='all')&(C.horizon_ms==1000)]
out['joint_cells']=q[['band','rows','common','no_fill']].to_dict('records')
rx=re.compile(r'^(\S+) tag=(\S+) rc=(\d+) peak_rss_mib=(\d+) wall_s=([\d.]+) cpu_user_s=([\d.]+)')
names=['ladder_rest','ladder_audit','ladder_guard','ladder_pressure','ladder_pressure_lag','ladder_pressure_ablation','ladder_pressure_residual','ladder_pressure_positive']
logs=[p for n in names for p in (R/'predictfun/reviews'/f'{n}_20261121'/'logs').rglob('*.log')]
logs += [p for n in names for p in (R/'predictfun/reviews/automatic'/f'20261121_{n}_review').glob('*.log')]
logs += list(O.glob('*.log'))
rec=[m for p in logs for ln in p.read_text().splitlines() if (m:=rx.match(ln))]
own=[m for p in O.glob('*.log') for ln in p.read_text().splitlines() if (m:=rx.match(ln))]
out['resource_asof_before_summary_job']=dict(cumulative_hours=sum(float(m[6]) for m in rec)/3600,own_hours=sum(float(m[6]) for m in own)/3600)
assert out['resource_asof_before_summary_job']['cumulative_hours']<2 and out['resource_asof_before_summary_job']['own_hours']<.01
(O/'readings.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
