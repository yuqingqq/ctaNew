"""Could we be first at a new best price? (2026-11-03, part of the speed-race check declared 23:14Z.) With a 1-tick spread a new
best bid at p can only appear once the asks at p are gone (and mirror for asks). Per new best price (live 0-240 s, 8 read days):
the vacate time = first frame after the last frame (within 5 s) whose opposite best was at or through p; gap = creation - vacate,
both Predict venue time. Our reach after a vacate: seen a median 53.1 ms later (minimum 13.2), order live 40 ms after sending
-> 93.1 ms (53.2 at the feed minimum). 'wide' = no opposite order at or through p in the 5 s before (posting inside a wider spread).
Usage: python3 -m predictfun.research.pf_vacate_gap
"""
import os
import numpy as np, pandas as pd
from . import pf_io as io

DAYS = ['20261023', '20261024', '20261025', '20261026', '20261027', '20261031', '20261101', '20261102']
EDGES = [0.0, 13.2, 53.2, 93.1, 250.0, 1000.0]; LAB = ['same update', '0-13 ms', '13-53 ms', '53-93 ms', '93-250 ms', '250-1000 ms', '1-5 s', 'wide']
rows = []
for d in DAYS:
    S1 = f'{io.OUT}/step1/{d}'
    bk = pd.concat([pd.read_parquet(f'{S1}/books/{f}', columns=['mid', 't_ms', 'b1_p', 'a1_p']) for f in sorted(os.listdir(f'{S1}/books'))]).sort_values(['mid', 't_ms'], kind='stable')
    reg = pd.read_parquet(f'{S1}/registry.parquet', columns=['mid', 'ws']).set_index('mid')
    for mid, g in bk.groupby('mid', sort=False):
        if mid not in reg.index:
            continue
        ws = float(reg.loc[mid].ws) * 1000.0; tf = g.t_ms.values.astype(float); b1 = g.b1_p.values; a1 = g.a1_p.values
        lw = (tf >= ws) & (tf < ws + 240000.0)
        for side, best, opp in ((0, b1, a1), (1, a1, b1)):
            imp = np.r_[False, (best[1:] >= 0) & (best[:-1] >= 0) & ((best[1:] > best[:-1]) if side == 0 else (best[1:] < best[:-1]))] & lw
            for k in np.flatnonzero(imp):
                p = best[k]; j = k - 1; lim = tf[k] - 5000.0
                while j >= 0 and tf[j] >= lim and not (opp[j] >= 0 and ((opp[j] <= p) if side == 0 else (opp[j] >= p))):
                    j -= 1
                if j < 0 or tf[j] < lim:
                    rows.append((d, 'wide', np.nan)); continue
                gap = tf[k] - tf[j + 1]
                rows.append((d, LAB[0] if j + 1 == k else LAB[min(int(np.searchsorted(EDGES, gap, side='right')), 6)], gap))
    del bk
    io.mem_check(f'day {d}')
R = pd.DataFrame(rows, columns=['day', 'cls', 'gap'])
c = R.cls.value_counts(normalize=True).reindex(LAB).fillna(0)
print(f'new best prices {len(R)} (8 days, live 0-240 s): gap from the price becoming free to someone posting there')
print('  ' + '  '.join(f'{k} {v:.1%}' for k, v in c.items()))
nw = R[R.cls != 'wide']
print(f"  of non-wide: posted in the same book update {np.mean(nw.cls == 'same update'):.1%}; later than 93.1 ms (our median reach) {np.mean(nw.gap > 93.1):.1%}; "
      f"later than 53.2 ms (our reach at the feed minimum) {np.mean(nw.gap > 53.2):.1%}")
io.write_json(f'{io.OUT}/step4/report_vacate_gap.json', dict(n=len(R), classes=c.round(4).to_dict(), later_than_93=float(np.mean(nw.gap > 93.1)), later_than_53=float(np.mean(nw.gap > 53.2))))
