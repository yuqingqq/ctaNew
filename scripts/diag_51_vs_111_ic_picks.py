"""Compare IC-ranker picks: 51-panel vs 111-panel at matched boundaries.

For each rebalance boundary (~quarterly), compute per-symbol:
  - trailing 180d Spearman IC on EACH panel
  - n_obs supporting that IC estimate
  - bootstrap SE of that IC
  - rank within panel's eligible set
  - picked (top-15) flag

Side-by-side comparison table, plus targeted analysis of:
  A. Symbols picked in 111 but NOT 51 — what's their 51-panel IC? Why displaced?
  B. Symbols picked in 51 but NOT 111 — were they replaced by noise spikes or real lift?
  C. Rank-15/16 IC gap on each panel — is the 111 cutoff in a noisier region?
  D. Distribution of (n_obs, SE, IC) for the three groups:
     (i) overlapping symbols picked in both,
     (ii) new symbols picked only in 111,
     (iii) overlapping symbols dropped from 111.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

REPO = Path("/home/yuqing/ctaNew")
sys.path.insert(0, str(REPO))

APD_51 = REPO / "outputs/vBTC_audit_panel/all_predictions.parquet"
APD_111 = REPO / "outputs/vBTC_audit_panel_111_correct/all_predictions.parquet"
KLINES_DIR = REPO / "data/ml/test/parquet/klines"

IC_WIN_D = 180
IC_UPD_D = 90
TOP_N = 15
MIN_OBS = 100
MIN_HIST = 60
HORIZON = 48
BAR_MS = 5 * 60 * 1000
OOS_FOLDS = list(range(1, 10))


def to_ms_int(s):
    ts = pd.to_datetime(s)
    if hasattr(ts.dtype, "tz") and ts.dtype.tz is not None:
        ts = ts.dt.tz_convert("UTC").dt.tz_localize(None)
    return ts.astype("datetime64[ms]").astype("int64").to_numpy()


def get_listings():
    L = {}
    for d in KLINES_DIR.iterdir():
        if not d.is_dir(): continue
        m5 = d / "5m"
        if not m5.exists(): continue
        f = sorted(m5.glob("*.parquet"))
        if not f: continue
        try: L[d.name] = pd.Timestamp(f[0].stem, tz="UTC")
        except Exception: pass
    return L


def per_sym_metrics(apd, b_ms, window_ms, listings, panel_syms, rng):
    """At boundary b, compute per-symbol IC, n_obs, bootstrap SE.
    Returns DataFrame indexed by symbol with columns [ic, n_obs, se]."""
    ts = pd.Timestamp(b_ms, unit="ms", tz="UTC")
    cutoff = ts - pd.Timedelta(days=MIN_HIST)
    elig = {s for s in panel_syms if listings.get(s) and listings[s] <= cutoff}
    past = apd[(apd["t_int"] >= b_ms - window_ms) & (apd["t_int"] < b_ms) &
               (apd["exit_t_int"] <= b_ms) & (apd["symbol"].isin(elig))]
    rows = []
    for sym, g in past.groupby("symbol"):
        n = len(g)
        if n < MIN_OBS: continue
        p = g["pred"].to_numpy(); a = g["alpha_A"].to_numpy()
        ic = pd.Series(p).rank().corr(pd.Series(a).rank())
        # bootstrap SE
        ics_boot = []
        for _ in range(100):
            idx = rng.integers(0, n, n)
            ic_b = pd.Series(p[idx]).rank().corr(pd.Series(a[idx]).rank())
            ics_boot.append(ic_b)
        se = float(np.nanstd(ics_boot))
        rows.append({"symbol": sym, "ic": float(ic), "n_obs": n, "se": se})
    df = pd.DataFrame(rows).sort_values("ic", ascending=False).reset_index(drop=True)
    df["rank"] = df.index + 1
    df["picked"] = (df["rank"] <= TOP_N).astype(int)
    return df


def make_boundaries(apd, ref_panel_sample_ts):
    """Use the same boundary anchoring as production: t0 = first OOS sample time,
    boundaries every IC_UPD_D × 288 × 5min apart."""
    oos = apd[apd["fold"].isin(OOS_FOLDS)]
    times = sorted(oos["open_time"].unique())
    sampled = times[::HORIZON]
    t0_ms = int(pd.Timestamp(sampled[0]).timestamp() * 1000)
    update_ms = IC_UPD_D * 288 * BAR_MS
    boundaries = sorted({t0_ms + ((int(pd.Timestamp(t).timestamp()*1000) - t0_ms) // update_ms) * update_ms
                         for t in sampled})
    return boundaries


def main():
    print("=== 51 vs 111 panel: IC-ranker pick decomposition ===\n", flush=True)
    apd_51 = pd.read_parquet(APD_51)
    apd_51 = apd_51.dropna(subset=["alpha_A"]).copy()
    apd_51["t_int"] = to_ms_int(apd_51["open_time"])
    apd_51["exit_t_int"] = to_ms_int(apd_51["exit_time"])

    apd_111 = pd.read_parquet(APD_111)
    apd_111 = apd_111.dropna(subset=["alpha_A"]).copy()
    apd_111["t_int"] = to_ms_int(apd_111["open_time"])
    apd_111["exit_t_int"] = to_ms_int(apd_111["exit_time"])

    listings = get_listings()
    panel_first_51 = apd_51.groupby("symbol")["open_time"].min()
    panel_first_111 = apd_111.groupby("symbol")["open_time"].min()
    for sd in (panel_first_51, panel_first_111):
        for s, t in sd.items():
            if s not in listings:
                t = t.tz_convert("UTC") if t.tz is not None else t.tz_localize("UTC")
                listings[s] = t

    syms_51 = set(apd_51["symbol"].unique())
    syms_111 = set(apd_111["symbol"].unique())

    boundaries_51 = make_boundaries(apd_51, None)
    boundaries_111 = make_boundaries(apd_111, None)
    print(f"51 boundaries: {[pd.Timestamp(b,unit='ms',tz='UTC').strftime('%Y-%m-%d') for b in boundaries_51]}",
          flush=True)
    print(f"111 boundaries: {[pd.Timestamp(b,unit='ms',tz='UTC').strftime('%Y-%m-%d') for b in boundaries_111]}\n",
          flush=True)

    # Match boundaries by proximity. For each 51 boundary, find nearest 111 boundary.
    window_ms = IC_WIN_D * 288 * BAR_MS
    pairs = []
    for b51 in boundaries_51:
        b111 = min(boundaries_111, key=lambda x: abs(x - b51))
        pairs.append((b51, b111))

    rng = np.random.default_rng(11)
    pd.set_option("display.width", 220); pd.set_option("display.max_columns", 30)
    pd.set_option("display.float_format", lambda x: f"{x:+.4f}")

    # Focus on boundary 1 (the worst case in the 111 run)
    summary_rows = []
    for b51, b111 in pairs:
        ts51 = pd.Timestamp(b51, unit="ms", tz="UTC")
        ts111 = pd.Timestamp(b111, unit="ms", tz="UTC")
        print("\n" + "=" * 110, flush=True)
        print(f"BOUNDARY  51:{ts51.strftime('%Y-%m-%d')}  vs  111:{ts111.strftime('%Y-%m-%d')}",
              flush=True)
        print("=" * 110, flush=True)
        m51 = per_sym_metrics(apd_51, b51, window_ms, listings, syms_51, rng)
        m111 = per_sym_metrics(apd_111, b111, window_ms, listings, syms_111, rng)
        m51 = m51.rename(columns={"ic":"ic_51","n_obs":"n_51","se":"se_51",
                                   "rank":"rank_51","picked":"picked_51"})
        m111 = m111.rename(columns={"ic":"ic_111","n_obs":"n_111","se":"se_111",
                                     "rank":"rank_111","picked":"picked_111"})
        # Outer merge so we see all symbols
        merged = m51.merge(m111, on="symbol", how="outer")
        merged["in_51"] = merged["ic_51"].notna().astype(int)
        merged["in_111"] = merged["ic_111"].notna().astype(int)

        # Categorise
        def cat(r):
            if r["picked_51"] == 1 and r["picked_111"] == 1: return "both"
            if r["picked_51"] == 1 and r["picked_111"] != 1: return "51_only"
            if r["picked_51"] != 1 and r["picked_111"] == 1:
                return "new_111" if r["in_51"] == 0 else "51_overlap_111_only"
            return "neither"
        merged["picked_cat"] = merged.apply(cat, axis=1)

        # Rank-15/16 gap on each panel
        if len(m51) >= TOP_N + 1:
            gap_51 = float(m51["ic_51"].iloc[TOP_N - 1] - m51["ic_51"].iloc[TOP_N])
            top15_51 = m51["ic_51"].iloc[:TOP_N].values
        else:
            gap_51 = np.nan; top15_51 = []
        if len(m111) >= TOP_N + 1:
            gap_111 = float(m111["ic_111"].iloc[TOP_N - 1] - m111["ic_111"].iloc[TOP_N])
            top15_111 = m111["ic_111"].iloc[:TOP_N].values
        else:
            gap_111 = np.nan; top15_111 = []
        se_med_51 = float(np.nanmedian(m51["se_51"]))
        se_med_111 = float(np.nanmedian(m111["se_111"]))

        print(f"\n  51-panel: |elig|={len(m51)}, rank15-16 IC gap={gap_51:.4f}, median SE={se_med_51:.4f}, "
              f"signal/noise={gap_51/se_med_51:.2f}", flush=True)
        print(f"  111-panel: |elig|={len(m111)}, rank15-16 IC gap={gap_111:.4f}, median SE={se_med_111:.4f}, "
              f"signal/noise={gap_111/se_med_111:.2f}", flush=True)

        # Top-25 on each
        print("\n  --- TOP 25 by IC on EACH panel (side-by-side) ---", flush=True)
        a = m51.head(25)[["rank_51","symbol","ic_51","n_51","se_51"]].reset_index(drop=True)
        b = m111.head(25)[["rank_111","symbol","ic_111","n_111","se_111"]].reset_index(drop=True)
        for i in range(25):
            la = (f"{int(a.iloc[i]['rank_51']):>2}  {a.iloc[i]['symbol']:<14}  "
                  f"IC={a.iloc[i]['ic_51']:+.4f}  n={int(a.iloc[i]['n_51']):>5}  SE={a.iloc[i]['se_51']:.4f}")
            lb = (f"{int(b.iloc[i]['rank_111']):>2}  {b.iloc[i]['symbol']:<16}  "
                  f"IC={b.iloc[i]['ic_111']:+.4f}  n={int(b.iloc[i]['n_111']):>5}  SE={b.iloc[i]['se_111']:.4f}")
            print(f"    51 | {la}    ||    111 | {lb}", flush=True)

        # Detailed picked-category breakdown
        for cat_name in ["new_111", "51_only", "51_overlap_111_only", "both"]:
            sub = merged[merged["picked_cat"] == cat_name]
            if sub.empty: continue
            print(f"\n  >>> CATEGORY: {cat_name}  (n={len(sub)})", flush=True)
            cols = ["symbol","ic_51","rank_51","n_51","se_51",
                    "ic_111","rank_111","n_111","se_111"]
            sub2 = sub[cols].copy()
            for c in ("rank_51","rank_111","n_51","n_111"):
                sub2[c] = sub2[c].apply(lambda v: "" if pd.isna(v) else f"{int(v)}")
            print(sub2.to_string(index=False), flush=True)

        # Aggregate stats per category
        print(f"\n  >>> Group statistics for this boundary:", flush=True)
        for cat_name in ["both", "51_only", "new_111", "51_overlap_111_only"]:
            sub = merged[merged["picked_cat"] == cat_name]
            if sub.empty: continue
            n111_med = sub["n_111"].median() if "n_111" in sub else np.nan
            n51_med = sub["n_51"].median() if "n_51" in sub else np.nan
            se111_med = sub["se_111"].median() if "se_111" in sub else np.nan
            se51_med = sub["se_51"].median() if "se_51" in sub else np.nan
            ic111_med = sub["ic_111"].median() if "ic_111" in sub else np.nan
            ic51_med = sub["ic_51"].median() if "ic_51" in sub else np.nan
            print(f"    {cat_name:>20}: n={len(sub):>2}  "
                  f"med IC_51={ic51_med:+.4f}  med IC_111={ic111_med:+.4f}  "
                  f"med n_51={n51_med}  med n_111={n111_med}  "
                  f"med SE_51={se51_med:+.4f}  med SE_111={se111_med:+.4f}",
                  flush=True)

        summary_rows.append({
            "boundary_51": ts51.strftime('%Y-%m-%d'),
            "boundary_111": ts111.strftime('%Y-%m-%d'),
            "n_elig_51": len(m51), "n_elig_111": len(m111),
            "gap_51": gap_51, "gap_111": gap_111,
            "se_med_51": se_med_51, "se_med_111": se_med_111,
            "sn_51": gap_51 / se_med_51 if se_med_51 > 0 else np.nan,
            "sn_111": gap_111 / se_med_111 if se_med_111 > 0 else np.nan,
        })
        # Save full table for this boundary
        merged.to_csv(REPO / f"outputs/vBTC_51_vs_111_boundary_{ts51.strftime('%Y%m%d')}.csv",
                      index=False)

    print("\n" + "=" * 110)
    print("  SUMMARY ACROSS BOUNDARIES")
    print("=" * 110)
    print(pd.DataFrame(summary_rows).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
