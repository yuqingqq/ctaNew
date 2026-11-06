"""Step 1b: timing/side semantics on ISOLATED trade cells (no other REST volume at the same
market/side/price within +-3 s, and none on the opposite side at that price within +-3 s).
Usage: python3 -m predictfun.research.pf_step1_isolated 20261023"""
import sys, bisect, collections
import numpy as np, pandas as pd
from . import pf_io as io
day = sys.argv[1]; S1 = f'{io.OUT}/step1/{day}'
L = pd.read_parquet(f'{S1}/legs.parquet'); L = L[L.phase == 'live']
P = pd.read_parquet(f'{S1}/pend_inc.parquet')
cells = L.groupby(['mid', 'exec_s', 'side', 'ptick'], as_index=False).agg(shares=('shares', 'sum'), n=('shares', 'size'))
byk = collections.defaultdict(list)
for mid, s, side, pt in zip(cells.mid.values, cells.exec_s.values, cells.side.values, cells.ptick.values):
    byk[(mid, pt)].append((s, side))
for v in byk.values(): v.sort()
iso = []
for i, (mid, s, side, pt) in enumerate(zip(cells.mid.values, cells.exec_s.values, cells.side.values, cells.ptick.values)):
    v = byk[(mid, pt)]; j = bisect.bisect_left(v, (s - 3, -1)); ok = True
    while j < len(v) and v[j][0] <= s + 3:
        if v[j] != (s, side): ok = False; break
        j += 1
    if ok: iso.append(i)
C = cells.iloc[iso]
pidx = collections.defaultdict(list)
for mid, t, side, pt, inc in zip(P.mid.values, P.t_ms.values, P.side.values, P.ptick.values, P.inc.values):
    pidx[(mid, side, pt)].append((t, inc))
for v in pidx.values(): v.sort()
pk = {k: [a for a, _ in v] for k, v in pidx.items()}
out = {}
for lab, flip in (('same', False), ('flipped', True)):
    offs = []; ratio = []; found = 0
    for mid, s, side, pt, sh in zip(C.mid.values, C.exec_s.values, C.side.values, C.ptick.values, C.shares.values):
        k = (mid, 1 - side if flip else side, pt)
        if k not in pk: continue
        ks = pk[k]; a = bisect.bisect_left(ks, s * 1000 - 2500); b = bisect.bisect_left(ks, s * 1000 + 2500)
        if b > a:
            found += 1
            tt = [ks[x] - s * 1000 for x in range(a, b)]; inc = [pidx[k][x][1] for x in range(a, b)]
            j = int(np.argmin(np.abs(np.array(inc) - sh)))   # increment closest in size
            offs.append(tt[j]); ratio.append(inc[j] / sh)
    offs = np.array(offs); ratio = np.array(ratio)
    out[lab] = dict(n_iso=int(len(C)), found=found, share_found=round(found / max(len(C), 1), 3),
                    offset_ms_pct=dict(zip(['p05', 'p25', 'p50', 'p75', 'p95'], np.percentile(offs, [5, 25, 50, 75, 95]).round(0).tolist())) if len(offs) else None,
                    share_offset_in_0_1000=round(float(((offs >= 0) & (offs < 1000)).mean()), 3) if len(offs) else None,
                    share_offset_in_m500_500=round(float(((offs >= -500) & (offs < 500)).mean()), 3) if len(offs) else None,
                    size_ratio_pct=dict(zip(['p10', 'p50', 'p90'], np.percentile(ratio, [10, 50, 90]).round(3).tolist())) if len(ratio) else None,
                    share_size_exact=round(float((np.abs(ratio - 1) < 0.01).mean()), 3) if len(ratio) else None)
io.write_json(f'{S1}/isolated_recon.json', out)
for k, v in out.items(): print(k, v)
