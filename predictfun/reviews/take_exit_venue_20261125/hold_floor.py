"""Is the historical hold about 150 ms? Distribution of print - (last Binance futures mid change in the competitor's direction before the
print, within 2 s) over the competitor events of lead_venues.py. Diagnostic. Output (write-once): hold_floor.json.
Usage: PF_SLOT=claude predictfun/research/run_light.sh hold_floor -- python3 predictfun/reviews/take_exit_venue_20261125/hold_floor.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

H = Path(__file__).resolve().parent
sys.path.insert(0, str(H))
import lead_venues as LV  # noqa: E402


def main():
    fn = H / 'hold_floor.json'
    LV.C.require(not fn.exists(), 'hold_floor.json exists (write-once)')
    E = pd.read_parquet(LV.OUT / 'lead_events.parquet', columns=['day', 'A', 'd', 'cls'])
    out = []
    for day in LV.DAYS:
        bk, bm = LV.OD.seg_arrays(LV.OD.SEG_OF[day]); bk = bk.astype(float)
        X = E[E.day == day]
        for a, d in zip(X.A.values + LV.HOLD, X.d.values):
            j = int(np.searchsorted(bk, a, 'right') - 1); dd = np.nan
            while j > 0 and bk[j] >= a - 2000.:
                if d * (bm[j] - bm[j - 1]) > 0:
                    dd = a - bk[j]; break
                j -= 1
            out.append(dd)
    D = np.asarray(out); ok = np.isfinite(D)
    h, e = np.histogram(D[ok], bins=np.arange(0, 1001, 25))
    res = dict(asof='2026-11-25', events=len(D), with_change_within_2s=float(ok.mean()), bins_ms=e.tolist(), counts=h.tolist(),
               quantiles_ms={str(q): float(np.quantile(D[ok], q)) for q in (.05, .1, .25, .5, .75, .9)},
               share_within_150ms=float(np.mean(D[ok] < 150.)), share_150_250ms=float(np.mean((D[ok] >= 150.) & (D[ok] < 250.))), share_ge_250ms=float(np.mean(D[ok] >= 250.)))
    LV.TR.dump(fn, res)
    print(json.dumps({k: v for k, v in res.items() if k not in ('bins_ms', 'counts')}))


if __name__ == '__main__':
    main()
