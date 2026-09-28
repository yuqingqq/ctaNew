"""Moved from the session scratchpad on 2026-09-28 so every quoted number has its producing
code in the repo (CLAUDE.md rules 10/12). Run from movewatch/: python3 -m research.qnt_presets"""
import numpy as np
from research.replay import (load_panel, candidate_table, triggers, run_alerts, mem, OUT_DIR, DEFAULT_RULES)
from movewatch.alerts import AlertCfg
syms, idx, C, V = load_panel("2026-08-01", "2026-09-28")
z = np.load(OUT_DIR / "blocks_2026-08-01_2026-09-28_r15.npz"); SIG, V1M, LIQ = z["SIG"], z["V1M"], z["LIQ"]
cand = candidate_table(C, V, SIG, V1M, LIQ, DEFAULT_RULES, 15, pre_floor=1.5, pre_z=1.0, pre_v=1.0)
qi = syms.index("QNTUSDT"); cq = cand[cand.i == qi]
mem("qnt presets")
presets = {"<=40 msgs/day": (3.0, 1.25, 1.0, 5e6), "<=60": (2.0, 1.5, 1.0, 1e7),
           "<=80": (2.0, 1.0, 1.5, 5e6), "<=120": (1.5, 1.0, 1.0, 5e6)}
for name, (fm, zm, vm, lq) in presets.items():
    r = DEFAULT_RULES.scaled(fm, zm, vm); r.liq_min_usd = lq
    al = run_alerts(triggers(cq, r), C, syms, idx, AlertCfg())
    up = al[al.direction > 0] if len(al) else al
    first = up.iloc[0] if len(up) else None
    print(f"\n{name}: {len(al)} QNT messages 09-20..27" if len(al) else f"\n{name}: none")
    for a in al.itertuples():
        if str(a.time) < "2026-09-20": continue
        print(f"   {str(a.time)[:16]} {a.kind:9s} {a.direction:+d} ${a.price:.2f} {a.move_pct:+.1f}% {a.horizon if isinstance(a.horizon, str) else ''}")

# QNT on the 09-24 news move: raw 4h return, sigma-based z, volume multiple (baseline block before the move)
import math, pandas as pd
t = idx.get_indexer([pd.Timestamp("2026-09-24 16:00", tz="UTC")])[0]
c = np.asarray(C[qi], np.float64); v = np.asarray(V[qi], np.float64)
b = t // 15
r4 = math.log(c[t] / c[t - 240]); sig = float(SIG[qi, b]); v1m = float(V1M[qi, b])
print(f"\nQNT 09-24 16:00: 4h {100*(math.exp(r4)-1):+.1f}%  sigma1m {100*sig:.3f}%  z(4h) {r4/(sig*math.sqrt(240)):.1f}"
      f"  vol x{v[t-239:t+1].sum()/(v1m*240):.1f}  liq24h ${LIQ[qi, b]/1e6:.1f}M")
