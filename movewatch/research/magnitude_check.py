"""Moved from the session scratchpad on 2026-09-28 so every quoted number has its producing
code in the repo (CLAUDE.md rules 10/12). Run from movewatch/: python3 -m research.magnitude_check"""
import numpy as np, pandas as pd
from research.replay import OUT_DIR, PANEL_DIR, load_panel, mem
from research.common import Stale, is_tradfi, panel_meta
syms, idx, C, V = load_panel("2026-08-01", "2026-09-28")
LIQ = np.load(OUT_DIR / "blocks_2026-08-01_2026-09-28_r15.npz")["LIQ"]
f = pd.read_parquet(OUT_DIR / "post_signal" / "signals_delay1.parquet")      # entry 1 min after the alert
f = f[(f["kind"] == "NEW") & (f["time"] < pd.Timestamp("2026-09-20", tz="UTC"))]
rng = np.random.default_rng(7)
stale = Stale(panel_meta(PANEL_DIR, OUT_DIR / "panel_2026-08-01_2026-09-28", syms, idx), len(syms))
T = C.shape[1]
# baseline: random (coin, minute) pairs among liquid coins (>= $5M/day), Aug 2 - Sep 19, same entry convention
base = {60: [], 240: []}
while len(base[240]) < 20000:
    i = int(rng.integers(0, C.shape[0])); t = int(rng.integers(1440, T - 1441))
    if LIQ[i, t // 15] < 5e6 or not np.isfinite(C[i, t]) or is_tradfi(syms[i]) or stale.coin(i, t, t + 240):
        continue
    for k in (60, 240):
        if np.isfinite(C[i, t + k]):
            base[k].append(abs(np.log(C[i, t + k] / C[i, t])))
mem("magnitude")
print(f"signals n={len(f)} (Aug 2 - Sep 19, NEW, live strict setting) | random liquid coin-minutes n={len(base[240])}")
for k in (60, 240):
    s = np.abs(f[f"r{k}"].dropna().to_numpy())
    b = np.array(base[k])
    print(f"  next {k//60}h |move|: signals median {100*np.median(s):.1f}%  (p90 {100*np.percentile(s,90):.1f}%)"
          f"  vs random median {100*np.median(b):.1f}% (p90 {100*np.percentile(b,90):.1f}%)  -> {np.median(s)/np.median(b):.1f}x larger")
up = (f["r60"] > 0).mean()
print(f"  direction: move continued after 1h in {100*up:.0f}% of signals (a coin flip is 50%)")
