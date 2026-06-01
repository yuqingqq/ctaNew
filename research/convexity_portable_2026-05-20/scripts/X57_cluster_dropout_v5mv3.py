"""X57 — Cluster dropout for V5_minus_v3_7cx — verify no hidden cluster fragility.

X35 showed V0 depended on AI cluster (TAO+VVV).
X55 will tell us about V5_minus_v3 + AI.
This script tests dropping each cluster (major/L1/defi/memes/ai/other_alt) from
V5_minus_v3_7cx predictions on canonical HL-50.
"""
from __future__ import annotations
import csv, sys, importlib.util, json
from pathlib import Path
import pandas as pd

REPO = Path("/home/yuqing/ctaNew")
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
CACHE = REPO / "research/convexity_portable_2026-05-20/results/_cache"
spec = importlib.util.spec_from_file_location("x6",
    REPO / "research/convexity_portable_2026-05-20/scripts/X6_controlled_matrix.py")
x6 = importlib.util.module_from_spec(spec); spec.loader.exec_module(x6)

with open(REPO / "config/clusters_v1.json") as f: CLUSTERS = json.load(f)


def main():
    apd = pd.read_parquet(CACHE / "x54_V5_minus_v3_7cx_preds.parquet")
    ref_m = x6.run_sleeve_on_preds(CACHE / "x54_V5_minus_v3_7cx_preds.parquet", "x57_baseline")
    ref = ref_m.get("sharpe", 0) or 0
    print(f"V5_minus_v3_7cx HL-50 baseline: {ref:+.2f}")

    print(f"\nCluster dropout:")
    for cname, syms in CLUSTERS.items():
        drop = set(syms)
        apd_d = apd[~apd["symbol"].isin(drop)]
        n_kept = apd_d["symbol"].nunique()
        tmp = CACHE / f"x57_drop_{cname}_preds.parquet"
        apd_d.to_parquet(tmp, index=False)
        m = x6.run_sleeve_on_preds(tmp, f"x57_drop_{cname}")
        sh = m.get("sharpe", 0) or 0
        print(f"  drop {cname:<18} ({len(drop & set(apd['symbol'].unique()))} syms removed, "
              f"{n_kept} kept): Sharpe={sh:+.2f} (Δ {sh-ref:+.2f})", flush=True)


if __name__ == "__main__":
    main()
