"""DEPRECATED (2026-07-11 audit #1) — SUPERSEDED by live/gap_guard_panel.py. This script OVER-MASKED 175
VALID labels (incl. all 174 at 2026-06-04) because it keyed on PANEL row-spacing (a construction artifact),
not real raw-kline gaps, and masked only LABELS (leaving row-based FEATURES gap-contaminated). Use
gap_guard_panel.py, which keys on real 5-min gaps and guards features+labels+cross-sectional/target recompute.

--- original docstring ---
Audit fix (2026-07-10): surgically NaN gap-crossing forward labels in the deployable panel.

The X70/X132 panel's forward return used row-based .shift(-48) while exit_time = open_time+4h, so at
data gaps a "4h" label became a multi-week return (corrupt label + defeated purge -> leak). This is
equivalent to the source fix (X70 target_alpha, now gap-safe): a forward label is valid ONLY if the
bar at open_time + HORIZON*5min (= next 4h grid bar) exists for that symbol; otherwise the +4h bar is
missing (gap or end-of-data) and the forward return is undefined -> NaN. Cleaning the existing panel
this way reproduces a from-scratch rebuild (gap labels are the only corruption; all else verified OK).
Writes panel_expanded_v0_clean.parquet; does NOT overwrite the original (provenance).
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
REPO = Path("/home/yuqing/ctaNew")
SRC = REPO/"outputs/vBTC_features/panel_expanded_v0.parquet"
OUT = REPO/"outputs/vBTC_features/panel_expanded_v0_clean.parquet"
FWD = pd.Timedelta(hours=4)   # 4h-sampled panel; +HORIZON*5min = +4h = next grid bar

def main():
    PAN = pd.read_parquet(SRC)
    PAN["open_time"] = pd.to_datetime(PAN["open_time"], utc=True)
    PAN = PAN.sort_values(["symbol","open_time"]).reset_index(drop=True)
    print(f"panel: {len(PAN)} rows, {PAN.symbol.nunique()} syms, {PAN.open_time.min()}..{PAN.open_time.max()}")
    # resolution sanity: median per-symbol spacing
    d = PAN.groupby("symbol")["open_time"].diff().dt.total_seconds().div(3600)
    print(f"per-symbol bar spacing (h): median {d.median():.1f}, 4h-share {100*(d.eq(4).mean()):.1f}%, >4h(gap) {int((d>4).sum())}")
    n_ret_nan0 = PAN["return_pct"].isna().sum(); n_alp_nan0 = PAN["alpha_vs_btc_realized"].isna().sum()
    print(f"pre-fix NaN: return_pct {n_ret_nan0}, alpha {n_alp_nan0}")

    # CORRUPT = internal-gap pre-edge bar: the NEXT same-symbol bar EXISTS but is >4h ahead, so this
    # bar's stored forward label spans the gap. (End-of-data last bars have no next bar -> NaT -> not
    # flagged; they are post-fit-cut and unused, left as-is for a minimal fix.)
    nxt = PAN.groupby("symbol")["open_time"].shift(-1)
    gap_h = (nxt - PAN["open_time"]).dt.total_seconds() / 3600.0
    corrupt = (gap_h > 4.01).fillna(False).to_numpy()
    print(f"\ngap-crossing labels to NaN (next same-symbol bar >4h away): {int(corrupt.sum())}")
    cd = PAN.loc[corrupt, "open_time"].dt.strftime("%Y-%m-%d")
    print("corrupt-label dates (top):\n" + cd.value_counts().head(8).to_string())
    # the flagged 2025-02-28 gap-edge cycle
    edge = PAN[(PAN.open_time == pd.Timestamp("2025-02-28 20:00", tz="UTC"))]
    print(f"\n2025-02-28 20:00 bars: {len(edge)}, of which corrupt-flagged: {int(corrupt[PAN.open_time == pd.Timestamp('2025-02-28 20:00', tz='UTC')].sum())}")
    if len(edge):
        print(f"  their stored return_pct: mean {edge['return_pct'].mean()*100:+.1f}%  (corrupt 22-day return mislabeled as 4h)")

    # APPLY: NaN the corrupted forward labels
    PAN.loc[corrupt, ["return_pct","alpha_vs_btc_realized"]] = np.nan
    print(f"\npost-fix NaN: return_pct {PAN['return_pct'].isna().sum()} (+{PAN['return_pct'].isna().sum()-n_ret_nan0}), "
          f"alpha {PAN['alpha_vs_btc_realized'].isna().sum()} (+{PAN['alpha_vs_btc_realized'].isna().sum()-n_alp_nan0})")
    PAN.to_parquet(OUT)
    print(f"wrote {OUT.name} ({len(PAN)} rows). Original untouched.")
    print("PANELFIXDONE")

if __name__ == "__main__":
    main()
