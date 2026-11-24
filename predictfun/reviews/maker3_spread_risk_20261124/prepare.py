import json
import pandas as pd
import core as C
assert not (C.H/'manifest.json').exists()
pop=C.population();paths=[C.V/'econ/fills_valued.parquet',C.MR/'v2/matched_gap_r1/per_market.csv',C.C/'take_logic_v2/panel.parquet',C.C/'take_logic_v2/events.parquet',
 C.H.parent/'maker_action_population_20261124/outputs/orders.csv',C.H.parent/'maker_timing_estimates_20261124/outputs/post_estimates.csv.gz',
 C.H.parent/'maker3_cap_skew_20261124/observations.json',C.D/'maker3_cap_skew_20261124/take_context_panel.parquet']
for day in C.DAYS:
 paths +=[C.V/f'tob_{day}.parquet',C.D/f'step1/{day}/registry.parquet']
 reg=pd.read_parquet(paths[-1]);key='m' if 'm' in reg else 'mid';v=reg[reg[key].isin(pop[pop.day==day].m)];assert len(v)==len(pop[pop.day==day]) and v.fee_bps.eq(200).all()
for r in pop.itertuples():
 paths.append(C.V/f'life/{r.day}_{r.m}_fills.parquet')
 for arm in ['BASE','WIDE_PASSIVE']:
  p=C.D/f'maker3_cap_skew_20261124/{r.m}_{arm}.json.gz'
  if p.exists():paths.append(p)
sources=[C.H/n for n in ['contract.md','core.py','build.py','risk.py','selftest.py','check.py','prepare.py']]
C.dump(C.H/'manifest.json',dict(inputs={str(p.relative_to(C.R)):C.sha(p) for p in paths},sources={str(p.relative_to(C.R)):C.sha(p) for p in sources},asof='2026-11-24',new_candidates=0,descriptive_screens=10,validation=False))
print('prepared source/input freeze')
