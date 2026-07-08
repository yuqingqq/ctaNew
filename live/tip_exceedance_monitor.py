"""Tip exceedance-CUSUM monitor (V4_PERFORMANCE §7 forward-monitoring item).

Monitors the per-cycle tip legs as EVENT streams (threshold exceedances) with one-sided Bernoulli
CUSUMs — robust to the mean-lurching problem of trailing averages (the tip mean is tail-driven).
Monitoring-only: output feeds de-gross + human review per §7; it never switches anything.

PRE-REGISTERED DESIGN (2026-07-08, before evaluation on 2024+):
- Reference period for ALL baselines and thresholds: 2023 side cycles ONLY.
- Events (side regime): m1 long-leg jackpot drought (event: long tip > +500 bps; alt hypothesis
  p1 = p0/2); m2 jackpot surge (p1 = 2·p0); m3 short-leg squeeze surge (event: short tip < −500;
  p1 = 2·p0); m4 long-leg body decay (event: long tip > 0; p1 = p0 − 0.10).
- CUSUM: S_t = max(0, S_{t-1} + log f1(x)/f0(x)); alarm at S_t > h; after alarm, S resets.
- h per monitor = 99th percentile of the max-CUSUM over one reference-year, under stationary block
  bootstrap (block = 30 cycles, 2000 draws) of the 2023 stream → ≈1% false-alarm/year by design.
- Evaluation (blind): run 2024-01 → 2026-06, report alarm dates; success = alarms cluster at the
  known era transitions with lag reported honestly; failures reported as-is.
Live usage: same code pointed at the forward ledger's cycles (bot state), baselines FROZEN as the
constants printed by the validation run.
"""
import sys
import numpy as np, pandas as pd

def cusum_stream(x, p0, p1, h):
    """One-sided Bernoulli CUSUM; returns alarm indices."""
    l1 = np.log(p1 / p0); l0 = np.log((1 - p1) / (1 - p0))
    S, alarms = 0.0, []
    for i, xi in enumerate(x):
        S = max(0.0, S + (l1 if xi else l0))
        if S > h:
            alarms.append(i); S = 0.0
    return alarms

def calib_h(ref, p0, p1, n_year, blocks=30, draws=2000, q=0.99, seed=7):
    rng = np.random.default_rng(seed)
    ref = np.asarray(ref, dtype=bool)
    mx = []
    l1 = np.log(p1 / p0); l0 = np.log((1 - p1) / (1 - p0))
    for _ in range(draws):
        idx = []
        while len(idx) < n_year:
            st = rng.integers(0, len(ref) - blocks)
            idx.extend(range(st, st + blocks))
        x = ref[np.array(idx[:n_year])]
        S, m = 0.0, 0.0
        for xi in x:
            S = max(0.0, S + (l1 if xi else l0)); m = max(m, S)
        mx.append(m)
    return float(np.quantile(mx, q))

def main():
    src = sys.argv[1] if len(sys.argv) > 1 else "live/V4_GATE_MODEL_DATASET.parquet"
    d = pd.read_parquet(src, columns=["open_time", "macro_regime", "long_edge_bps", "short_edge_bps", "period"])
    d["open_time"] = pd.to_datetime(d["open_time"], utc=True)
    s = d[d.macro_regime == "side"].dropna(subset=["long_edge_bps", "short_edge_bps"]).sort_values("open_time")
    ref = s[s.period == "2023"]
    ev = s[s.period != "2023"].reset_index(drop=True)
    n_year = len(ref)   # one reference-year of side cycles
    MONS = {
        "m1 jackpot drought (long>+500, rate halves)": (ref.long_edge_bps > 500, ev.long_edge_bps > 500, 0.5, "down"),
        "m2 jackpot surge (long>+500, rate doubles)":  (ref.long_edge_bps > 500, ev.long_edge_bps > 500, 2.0, "up"),
        "m3 squeeze surge (short<-500, rate doubles)": (ref.short_edge_bps < -500, ev.short_edge_bps < -500, 2.0, "up"),
        "m4 long-body decay (long>0, rate -10pp)":     (ref.long_edge_bps > 0, ev.long_edge_bps > 0, None, "down"),
    }
    print(f"reference 2023: n={n_year} side cycles")
    for name, (r, x, mult, _) in MONS.items():
        p0 = float(np.mean(r))
        p1 = (p0 * mult) if mult else max(p0 - 0.10, 0.02)
        p1 = min(max(p1, 1e-3), 0.999)
        h = calib_h(r.to_numpy(), p0, p1, n_year)
        alarms = cusum_stream(x.to_numpy(), p0, p1, h)
        dates = [str(ev.open_time.iloc[i].date()) for i in alarms]
        print(f"\n{name}: p0={p0:.3f} p1={p1:.3f} h={h:.2f}")
        print(f"  alarms ({len(dates)}): {dates}")

if __name__ == "__main__":
    main()
