"""iter-018 follow-up: EXT per-episode + episode-LOFO only (the part that crashed). Cheap:
rebuild EXT, run P1/P2/P3 once each (no placebos), print episode tables + LOFO + save parquet."""
from __future__ import annotations
import numpy as np, pandas as pd
import iter018_dynamic_exit as M

EXT_PREDS = M.EXT_PREDS
EXT_EPISODES = M.EXT_EPISODES


def main():
    U = M.build_universe(EXT_PREDS, "EXT")
    times = U["times"]
    CONV = 0.0; DIV = 0.05
    pol = {}
    for p in ("P1", "P2", "P3"):
        pnl, avgh, nex, _ = M.run_policy(U, 4.5e-4, p, conv_band=CONV, div_band=DIV, only_side=True)
        pol[p] = pnl
        s = M.stats(pnl)
        print(f"  {p}: Sharpe {s['sharpe']:+.2f} maxDD {s['maxDD']:+.0f} Calmar {s['calmar']:+.2f} "
              f"totPnL {s['totPnL']:+.0f} avgHold {avgh:.2f} earlyExit {nex}", flush=True)
    M.episode_pnl("EXT", times, pol, EXT_EPISODES)
    M.episode_lofo("EXT", times, pol["P1"], pol["P2"], EXT_EPISODES, "P2")
    M.episode_lofo("EXT", times, pol["P1"], pol["P3"], EXT_EPISODES, "P3")
    out = {"open_time": pd.to_datetime(times), "fold": [U["fold_by_time"].get(t, -1) for t in times],
           "regime": U["regimes"]}
    for p in ("P1", "P2", "P3"): out[f"pnl_{p}"] = pol[p]
    pd.DataFrame(out).to_parquet(M.OUT/"iter018_dynexit_EXT.parquet", index=False)
    print("  saved -> iter018_dynexit_EXT.parquet", flush=True)


if __name__ == "__main__":
    main()
