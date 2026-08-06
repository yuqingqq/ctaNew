"""iter1c: gap-recovery control on the iter1 manifold DRIFT (descriptive robustness).

iter1/1b: the joint structure de-universalizes and gains dimension OOS->REC (not composition). REC
has ~6x more gap-recovered bars; do those inflate the drift? Recompute the structure on CLEAN bars
only (window_data_valid_5min & NOT any_raw_gap_5min) vs ALL valid bars, on a symbol sample. If the
drift (universality drop + effdim rise) persists on clean bars, it is not a gap artifact.

Run:  python3 -m live.emergent_iter1c_gapctrl
"""
from __future__ import annotations

import glob

import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed

from live.flow_harness import CUT, SRC
from live.emergent_harness import (SLIM_FLOW, clean_std, corr_similarity,
                                   corr_spectrum, participation_ratio)

MIN_ROWS = 800
CUTv = np.datetime64(CUT.tz_convert(None))


def sym_struct(sym: str):
    files = sorted(glob.glob(f"{SRC}/{sym}/*.parquet"))
    if not files:
        return None
    cols = ["bar_time", "window_data_valid_5min", "any_raw_gap_5min", *SLIM_FLOW]
    d = pd.concat([pd.read_parquet(f, columns=cols) for f in files], ignore_index=True)
    d = d[d["window_data_valid_5min"].fillna(False)]
    if d.empty:
        return None
    bt = pd.to_datetime(d["bar_time"], utc=True).to_numpy("datetime64[ns]")
    gap = d["any_raw_gap_5min"].fillna(False).to_numpy()
    X = d[SLIM_FLOW].to_numpy(float)
    out = {}
    for era, emask in (("OOS", bt < CUTv), ("REC", bt >= CUTv)):
        for tag, gmask in (("all", np.ones(len(d), bool)), ("clean", ~gap)):
            m = emask & gmask & np.isfinite(X).all(axis=1)
            if m.sum() < MIN_ROWS:
                continue
            C, w, _ = corr_spectrum(clean_std(X[m]))
            out[(era, tag)] = (C, participation_ratio(w), int(m.sum()))
    return sym, out


def agg(results, era, tag):
    Cs = [o[(era, tag)][0] for _, o in results if (era, tag) in o]
    effs = [o[(era, tag)][1] for _, o in results if (era, tag) in o]
    if len(Cs) < 5:
        return None
    rng = np.random.default_rng(2)
    sims = []
    for _ in range(3000):
        i, j = rng.choice(len(Cs), 2, replace=False)
        s = corr_similarity(Cs[i], Cs[j])
        if np.isfinite(s):
            sims.append(s)
    return np.median(sims), np.median(effs), len(Cs)


def main():
    syms = sorted(p.name for p in SRC.iterdir() if p.is_dir())
    sample = syms[::4][:44]   # stride sample spanning liquid..thin
    print(f"gap-control sample: {len(sample)} symbols\n", flush=True)
    results = []
    with ProcessPoolExecutor(max_workers=10) as ex:
        futs = {ex.submit(sym_struct, s): s for s in sample}
        for f in as_completed(futs):
            r = f.result()
            if r:
                results.append(r)
    print(f"{'era':<5}{'bars':<8}{'universality (med pairwise)':<30}{'effdim (med)'}", flush=True)
    for era in ("OOS", "REC"):
        for tag in ("all", "clean"):
            a = agg(results, era, tag)
            if a:
                sim, eff, n = a
                print(f"  {era:<5}{tag:<8}{sim:+.3f}{'':<24}{eff:.2f}   (n_syms={n})", flush=True)
    print("\nDrift is a gap artifact ONLY if the OOS→REC universality drop / effdim rise "
          "vanishes on 'clean' bars.", flush=True)


if __name__ == "__main__":
    main()
