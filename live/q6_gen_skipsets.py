"""Q6 placebo skip-set generator — COMMITTED (defect-repeat of 2026-07-07; the first version was
inline-only and dose-mismatched: it matched cycles where the whole book was de-grossed (355/919)
instead of sf05's true suppressed-entry dose (581 ins / 1,931 OOS per sleeves.csv), so every
placebo was under-dosed by ~40-50% and the specificity verdict was VOID — results-review F1).

Correct dose: the TRUE suppressed-entry mask = cycles where the baseline opened a new side sleeve
but sf05 did not (from sleeves.csv comparison). Placebo sets are contiguous blocks placed within
baseline side-entry cycles, matched per window on (a) total suppressed-entry count and (b) the
real run-length distribution. Seeds 1..50 per window, deterministic, one parquet per seed.

Usage: python3 live/q6_gen_skipsets.py <q6_state_dir> <out_dir>
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd

Q6 = Path(sys.argv[1]); OUT = Path(sys.argv[2]); OUT.mkdir(parents=True, exist_ok=True)

def entry_cycles(state):
    """cycles (open_time) where a NEW sleeve with nonzero weights was opened."""
    sl = pd.read_csv(state / "sleeves.csv", parse_dates=["open_time"])
    # a sleeve row per cycle; entry = sleeve_json non-empty for the newest sleeve at that cycle
    col = "weights_json" if "weights_json" in sl.columns else sl.columns[-1]
    sl["has_entry"] = sl[col].astype(str).str.len() > 2   # "{}" = empty
    return sl.groupby("open_time")["has_entry"].last()

for win in ("ins", "oos"):
    base_dir, sf_dir = Q6 / f"bitcheck_{win}", Q6 / f"sf05_{win}"
    cy = pd.read_csv(base_dir / "cycles.csv", parse_dates=["open_time"]).sort_values("open_time")
    eb, ev = entry_cycles(base_dir), entry_cycles(sf_dir)
    m = cy[["open_time", "regime"]].copy()
    m["base_entry"] = m["open_time"].map(eb).fillna(False)
    m["sf_entry"] = m["open_time"].map(ev).fillna(False)
    supp = (m["regime"] == "side") & m["base_entry"] & (~m["sf_entry"])
    n_dose = int(supp.sum())
    # run-length distribution of the TRUE suppressed mask
    runs, c = [], 0
    for x in supp.to_numpy():
        if x: c += 1
        elif c: runs.append(c); c = 0
    if c: runs.append(c)
    # candidate placement domain: side cycles where the baseline actually entered
    dom = np.where((m["regime"] == "side").to_numpy() & m["base_entry"].to_numpy())[0]
    print(f"{win}: true dose {n_dose} suppressed entries in {len(runs)} runs "
          f"(mean {np.mean(runs):.1f}, max {max(runs)}); domain {len(dom)} side-entry cycles")
    for seed in range(1, 51):
        r = np.random.default_rng(seed)
        mask = np.zeros(len(m), bool)
        placed = 0
        for L in sorted(runs, reverse=True):
            for _ in range(500):
                st = int(r.choice(dom))
                seg = slice(st, st + L)
                if st + L <= len(m) and (m["regime"].iloc[seg] == "side").all() \
                   and m["base_entry"].iloc[seg].all() and not mask[seg].any():
                    mask[seg] = True; placed += L; break
        if placed != n_dose:
            print(f"  seed {seed}: WARNING placed {placed}/{n_dose}")
        pd.DataFrame({"open_time": m.loc[mask, "open_time"]}).to_parquet(OUT / f"{win}_seed{seed}.parquet", index=False)
print("SKIPSETS2DONE")
