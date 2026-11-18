"""Corrected ledger v5 on Oct 23: where do ST's CANCEL_FIRST fills come from? Under upper-only ST a mapped leg has no lower bound, so an event
can be wholly after a cancel only through a leg without a unique settlement id (its S3 fallback). Computed, not assumed."""
import sys
import numpy as np
import pandas as pd
d = '/home/yuqing/ctaNew/data/predict_fun/research/keep_pull_exec_windows_v5_20261118_20261023/'
F = pd.read_parquet(d + 'ledger_fills.parquet'); FE = pd.read_parquet(d + 'ledger_fill_events.parquet'); E = pd.read_parquet(d + 'ledger_events.parquet')
cf = F[(F.cls_ST_CP == 0) & F.pulled] if 'pulled' in F else F[F.cls_ST_CP == 0]
ev = FE[FE.fid.isin(cf.fid)].merge(E[['day', 'group', 'ST_lo', 'ST_lo_type']], on=['day', 'group'], how='left')
fin = ev.groupby('fid').ST_lo.apply(lambda x: bool(np.isfinite(x.values).all()))
lines = [f"ST|CP CANCEL_FIRST fills (pulled): {len(cf)}; their candidate events {len(ev)}, every one with a finite ST lower end: {bool(fin.all())} "
         f"({int(fin.sum())} of {len(fin)} fills); lower-end types {ev.ST_lo_type.value_counts().to_dict()}",
         f"events on the day with an unknown (-inf) ST lower end: {int(np.isneginf(E.ST_lo.values).sum())} of {len(E)}"]
open(sys.argv[1], 'w').write('\n'.join(lines) + '\n'); print('\n'.join(lines))
