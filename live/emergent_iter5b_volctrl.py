"""iter5b: is iter5's tfi / signed_volume_z incremental signal a VOL proxy (the iter2 trap)?

iter5 found tfi & signed_volume_z carry both-era partial-IC beyond price+book/flow (+0.029/+0.034 @5m).
But the control set lacked realized vol — and the program's recurring lesson is that flow = a vol proxy.
Add vol proxies (|tr_30m|, |tr_1h|) to the control set and re-measure. If the partial-IC collapses/destabilizes
=> vol proxy (same wall). If it survives ~unchanged => genuinely orthogonal (but still 5m = sub-cost per iter4c).

Run:  python3 -m live.emergent_iter5b_volctrl
"""
from __future__ import annotations

import glob

import numpy as np
import pandas as pd

from live.flow_harness import CUT, ci, partial_xsic
from live.emergent_harness import EXT
from live.emergent_iter5_richatoms import CONTROLS, load_ext

FEATS = ["tfi", "signed_volume_z"]


def main():
    need = list(set(FEATS + CONTROLS + ["tr_30m", "tr_1h"]))
    D = load_ext(["symbol", "bar_time", *need, "fwd_5m", "fwd_30m"])
    D["absr30"] = D["tr_30m"].abs()
    D["absr1h"] = D["tr_1h"].abs()
    print(f"panel {len(D):,} rows\n", flush=True)
    m_oos = (D["bar_time"] < CUT).to_numpy()
    m_rec = (D["bar_time"] >= CUT).to_numpy()
    volc = ["absr30", "absr1h"]
    print(f"  {'atom':<16}{'h':<7}{'ctrl':<12}{'OOS partial [95% CI]':<30}"
          f"{'REC partial [95% CI]':<30}both", flush=True)
    for feat in FEATS:
        base = [c for c in CONTROLS if c != feat]
        for h in ("fwd_5m", "fwd_30m"):
            for tag, ctrl in (("price+bf", base), ("+vol", base + volc)):
                ao, lo, uo = ci(partial_xsic(D, feat, ctrl, h, row_mask=m_oos))
                ar, lr, ur = ci(partial_xsic(D, feat, ctrl, h, row_mask=m_rec))
                both = "YES" if ((lo > 0 or uo < 0) and (lr > 0 or ur < 0)
                                 and np.sign(ao) == np.sign(ar)) else "no"
                print(f"  {feat:<16}{h:<7}{tag:<12}{f'{ao:+.4f}[{lo:+.4f},{uo:+.4f}]':<30}"
                      f"{f'{ar:+.4f}[{lr:+.4f},{ur:+.4f}]':<30}{both}", flush=True)
        print("", flush=True)


if __name__ == "__main__":
    main()
