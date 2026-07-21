"""Feature-level audit of the EXISTING V0_LEAN set through the STRATEGY'S REAL pipeline
(x6 preproc + per-symbol RidgeCV + HL=60 + exit_time purge + 1d embargo = gen_residual_target).
Answers: which feature carries the +0.030, what's dead weight, and does anything add on top?
  BASELINE  V0_LEAN rank-IC  (VALIDITY GATE: must reproduce ~+0.030 recent / +0.024 OOS)
  LOFO      drop each feature -> paired Δ vs baseline (Δ<0 CI<0 = feature CARRIES the edge)
  SINGLE    each feature alone -> raw rank-IC
  HEADROOM  add each funding feature (in panel, NOT in V0_LEAN) -> Δ (does it ADD on top?)
Both eras, day-clustered bootstrap CI on the paired delta. Same harness discipline as the OB work.

RESULTS (2026-07-20; VALIDITY GATE PASSED: baseline +0.0302 recent / +0.0210 OOS = strategy's honest edge):
- REDUNDANT SINGLE FACTOR: the full 14-feature model barely beats the best SINGLE feature — RECENT
  +0.0302 vs +0.0282 (return_1d); OOS +0.0210 vs +0.0202 (idio_vol_to_btc_1d). The other 13 features add
  only +0.0020 rec / +0.0008 oos. Every feature alone scores +0.012..+0.028; all are correlated
  momentum/vol/liquidity = ONE factor in 14 hats. Essentially no combination alpha.
- NO BOTH-ERA CARRIER (fragility): the LOFO "carries" set (Δ<0, CI<0) ROTATES by era — return_1d +
  btc_rvol_7d (RECENT) vs idio_vol_to_btc_1d + atr_pct (OOS); NONE carries CI<0 in both eras. The edge is
  a small, era-shifting composite = the fragile / universe-overfit local optimum, now quantified.
- NO FUNDING HEADROOM: adding funding_rate / _z_7d / _1d_change is within noise both eras (Δ -0.0007..+0.0006).
  12 of 14 features are individually redundant-to-dead-weight (dropping most is neutral; a few help a hair).
- VERDICT: no minable alpha in the existing V0 set via feature levers (confirms + quantifies "feature research
  closed"). The edge is thin (+0.021 OOS) and era-unstable; genuinely new alpha needs an ORTHOGONAL factor /
  new information, not more price/vol transforms. Caveats: this is the LINEAR per-symbol RidgeCV view (the
  machinery that reproduces +0.030); LOFO effect sizes are tiny (±0.001-0.002, near the noise floor) — the
  ROBUST takeaway is the REDUNDANCY (full model ~= best single feature), unambiguous in both eras.
"""
import os, sys
from pathlib import Path
import numpy as np, pandas as pd
import warnings; warnings.filterwarnings("ignore")
REPO = Path("/home/yuqing/ctaNew"); sys.path.insert(0, str(REPO))
os.environ["V4_PANEL"] = str(REPO / "outputs/vBTC_features/panel_expanded_v0_clean.parquet")
import live.train_twobook_models as tt
from sklearn.linear_model import RidgeCV
from scipy.stats import spearmanr

x6 = tt.x6; V0 = list(tt.V0_LEAN); EMB = pd.Timedelta(days=1); HL = 60.0
FUND = ["funding_rate", "funding_rate_z_7d", "funding_rate_1d_change"]
rng = np.random.default_rng(7)
RECENT_CUTS = [pd.Timestamp(t, tz="UTC") for t in ["2025-10-04", "2025-11-01", "2025-12-01", "2026-01-01",
               "2026-02-01", "2026-03-01", "2026-04-01", "2026-05-01", "2026-06-01", "2026-06-30"]]
OOS_CUTS = [pd.Timestamp(t, tz="UTC") for t in ["2023-06-01", "2023-09-01", "2023-12-01", "2024-03-01",
            "2024-06-01", "2024-09-01", "2024-12-01", "2025-03-01", "2025-06-01", "2025-09-01"]]


def build_panel():
    PAN = pd.read_parquet(tt.PANEL, columns=["symbol", "open_time", "exit_time",
                                             "alpha_vs_btc_realized"] + V0 + FUND)
    PAN["open_time"] = pd.to_datetime(PAN["open_time"], utc=True)
    PAN["exit_time"] = pd.to_datetime(PAN["exit_time"], utc=True)
    PAN = PAN[(PAN.open_time.dt.hour % 4 == 0) & (PAN.open_time.dt.minute == 0)]
    g = PAN.groupby("open_time")
    sd = g["alpha_vs_btc_realized"].transform("std").replace(0, np.nan)
    PAN["z_res"] = ((PAN["alpha_vs_btc_realized"] - g["alpha_vs_btc_realized"].transform("mean")) / sd).clip(-10, 10)
    return PAN.sort_values(["symbol", "open_time"]).reset_index(drop=True)


def gen(PAN, feats, cuts):
    rec = []
    for i in range(len(cuts) - 1):
        c0, c1 = cuts[i], cuts[i + 1]; fc = c0 - EMB
        tr = PAN[(PAN.exit_time < fc) & PAN["z_res"].notna()]
        te = PAN[(PAN.open_time >= c0) & (PAN.open_time < c1)]
        if tr.empty or te.empty:
            continue
        t_end = tr["open_time"].max()
        for sym, gg in tr.groupby("symbol"):
            if len(gg) < 300:
                continue
            try:
                s, h = x6.fit_preproc(gg, feats); X = x6.apply_preproc(gg, feats, s, h)
                w = np.exp(-((t_end - gg["open_time"]).dt.total_seconds().to_numpy() / 86400.0) / HL)
                m = RidgeCV(alphas=x6.RIDGE_ALPHAS).fit(X, gg["z_res"].to_numpy(), sample_weight=w)
                gte = te[te.symbol == sym]
                if len(gte):
                    rec.append(pd.DataFrame({"open_time": gte["open_time"].values,
                                             "alpha_A": gte["alpha_vs_btc_realized"].values,
                                             "pred": m.predict(x6.apply_preproc(gte, feats, s, h))}))
            except Exception:
                pass
    return pd.concat(rec, ignore_index=True) if rec else pd.DataFrame()


def perbar_ic(P):
    if P.empty:
        return pd.Series(dtype=float)
    return P.groupby("open_time").apply(
        lambda g: spearmanr(g["pred"], g["alpha_A"]).correlation if len(g) >= 5 else np.nan).dropna()


def paired_ci(ib, iv):
    j = pd.concat([ib.rename("a"), iv.rename("b")], axis=1).dropna()
    if len(j) < 5:
        return (np.nan, np.nan, np.nan)
    j["d"] = j["b"] - j["a"]; j["day"] = pd.to_datetime(j.index, utc=True).floor("1D")
    gg = [x["d"].values for _, x in j.groupby("day")]
    boot = [np.concatenate([gg[k] for k in rng.integers(0, len(gg), len(gg))]).mean() for _ in range(3000)]
    return (float(j["d"].mean()), *np.percentile(boot, [2.5, 97.5]))


def main():
    PAN = build_panel()
    print(f"panel rows {len(PAN):,} | {PAN.symbol.nunique()} syms | V0_LEAN={len(V0)} feats\n")
    for era, cuts in [("RECENT", RECENT_CUTS), ("OOS", OOS_CUTS)]:
        print(f"================= {era} (real per-symbol RidgeCV pipeline) =================")
        ib = perbar_ic(gen(PAN, V0, cuts))
        print(f"  BASELINE V0_LEAN rank-IC {ib.mean():+.4f}   [VALIDITY GATE ~+0.030 rec / +0.024 oos]\n")

        print("  LOFO  (drop feature -> Δ vs baseline; Δ<0 & CI<0 = feature CARRIES edge):")
        lofo = []
        for f in V0:
            d, lo, up = paired_ci(ib, perbar_ic(gen(PAN, [x for x in V0 if x != f], cuts)))
            lofo.append((f, d, lo, up))
        for f, d, lo, up in sorted(lofo, key=lambda t: (t[1] if np.isfinite(t[1]) else 0)):
            tag = "CARRIES (CI<0)" if (np.isfinite(up) and up < 0) else (
                  "drop-HELPS (CI>0)" if (np.isfinite(lo) and lo > 0) else "neutral")
            print(f"    -{f:26s} Δ {d:+.4f} [{lo:+.4f},{up:+.4f}]  {tag}", flush=True)

        print("\n  SINGLE-FEATURE rank-IC (each alone):")
        singles = [(f, perbar_ic(gen(PAN, [f], cuts)).mean()) for f in V0]
        for f, m in sorted(singles, key=lambda t: -(abs(t[1]) if np.isfinite(t[1]) else 0)):
            print(f"    {f:26s} {m:+.4f}", flush=True)

        print("\n  HEADROOM (add funding to V0_LEAN -> does it ADD on top?):")
        for f in FUND:
            d, lo, up = paired_ci(ib, perbar_ic(gen(PAN, V0 + [f], cuts)))
            tag = "ADDS (CI>0)" if (np.isfinite(lo) and lo > 0) else (
                  "hurts (CI<0)" if (np.isfinite(up) and up < 0) else "within noise")
            print(f"    +{f:26s} Δ {d:+.4f} [{lo:+.4f},{up:+.4f}]  {tag}", flush=True)
        print()
    print("V0ABLATIONDONE", flush=True)


if __name__ == "__main__":
    main()
