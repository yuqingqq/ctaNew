"""VALIDATE the derived model at the walk-forward/embargoed level (not raw IC):
Claim: edge ≈ low-vol (idio_vol) + reversal (return_1d); the other 12 V0 features are dead weight;
factors stable both eras.

Uses the deployed pipeline (v0_feature_ablation.gen = expanding per-symbol RidgeCV + exit_time purge +
1d embargo). Compares full-14 vs 2-factor vs each 1-factor, both eras, day-clustered CIs, and the PAIRED
(full − 2factor) difference (CI spans 0 = the 12 extra features add nothing = model validated).
Run: python3 -u -m live.validate_model
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from live.v0_feature_ablation import build_panel, gen, perbar_ic, paired_ci, V0, RECENT_CUTS, OOS_CUTS

VOL, MOM = "idio_vol_to_btc_1d", "return_1d"


def day_ci(ic, nb=3000, seed=1):
    s = pd.DataFrame({"v": ic.values}, index=pd.to_datetime(ic.index, utc=True))
    s["d"] = s.index.floor("1D")
    g = [x["v"].values for _, x in s.groupby("d")]
    rng = np.random.default_rng(seed); k = len(g)
    b = [np.concatenate([g[i] for i in rng.integers(0, k, k)]).mean() for _ in range(nb)]
    return float(ic.mean()), *np.percentile(b, [2.5, 97.5])


def main():
    PAN = build_panel()
    print(f"panel {len(PAN):,} rows | validating: edge ≈ [{VOL} + {MOM}] vs full-14\n", flush=True)
    for era, cuts in (("RECENT", RECENT_CUTS), ("OOS", OOS_CUTS)):
        full = perbar_ic(gen(PAN, V0, cuts))
        two = perbar_ic(gen(PAN, [VOL, MOM], cuts))
        vol = perbar_ic(gen(PAN, [VOL], cuts))
        mom = perbar_ic(gen(PAN, [MOM], cuts))
        print(f"===== {era} =====", flush=True)
        for name, ic in (("full-14", full), ("2-factor", two), (f"vol-only", vol), ("mom-only", mom)):
            m, lo, hi = day_ci(ic)
            print(f"    {name:<10} {m:+.4f} [{lo:+.4f},{hi:+.4f}]", flush=True)
        d, lo, hi = paired_ci(full, two)
        verdict = "12 extra feats ADD NOTHING (validated)" if lo < 0 < hi else "full BEATS 2-factor"
        print(f"    full − 2factor: {d:+.4f} [{lo:+.4f},{hi:+.4f}]  → {verdict}", flush=True)
        print("", flush=True)
    print("VALIDATEDONE", flush=True)


if __name__ == "__main__":
    main()
