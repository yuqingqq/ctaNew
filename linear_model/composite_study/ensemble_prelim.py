"""§4 PRELIMINARY (triage, NOT a verdict): is the linear-idio book even
low-correlation to / additive with production V3.1? Go/no-go on whether the
full clean harness rebuild is warranted.

HONEST CAVEATS (printed): linear book is from the σ_idio-CONTAMINATED
harness (R3 #7) — not the clean rebuild; V3.1 PnL is its IN-SAMPLE
net_with_overlay (not honest-forward); blend weight is NOT nested-OOS
(in-sample-optimistic); both resampled to DAILY to neutralize the
4h-vs-24h-sleeve aggregation mismatch (R3 #14). Purpose: detect a pulse
worth the expensive clean rebuild, or kill fast. Production LGBM unaffected.
"""
from __future__ import annotations
import importlib.util, sys, warnings
from pathlib import Path
import numpy as np, pandas as pd

warnings.filterwarnings("ignore")
REPO = Path("/home/yuqing/ctaNew"); sys.path.insert(0, str(REPO))


def _imp(n, r):
    s = importlib.util.spec_from_file_location(n, REPO / r)
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m


s94 = _imp("s94", "linear_model/scripts/94_info_ceiling_d1.py")
s94b = _imp("s94b", "linear_model/scripts/94b_info_ceiling_d1_grouped.py")
OUTD = REPO / "linear_model/composite_study/results"


def dsh(x):  # daily Sharpe
    x = np.asarray(x, float)
    return float(x.mean()/x.std(ddof=1)*np.sqrt(365)) if x.std(ddof=1) > 1e-12 else np.nan


def main():
    print("=" * 92, flush=True)
    print("  §4 PRELIMINARY ensemble-with-production (triage; contaminated "
          "harness; in-sample — NOT a verdict)", flush=True)
    print("=" * 92, flush=True)
    dec, syms, btc, pan = s94.build(universe_oi=False)
    LEAK = s94.LEAK
    F = [c for c in dec.columns if c not in LEAK and
         pd.api.types.is_numeric_dtype(dec[c])]
    if "s_t" not in F:
        F.append("s_t")
    d = dec.dropna(subset=F + ["tz", "alpha_beta"]).reset_index(drop=True)
    rid, _ = s94b.grouped_oof(d, F)
    d["pred"] = rid
    d = d[~d["pred"].isna()].copy()
    d["pos"] = np.sign(d["pred"])
    n = d.groupby("open_time")["symbol"].transform("count")
    d["w"] = d["pos"] / n
    d = d.sort_values(["symbol", "open_time"])
    d["dw"] = d.groupby("symbol")["w"].diff().abs().fillna(d["w"].abs())
    lin = d.groupby("open_time").apply(
        lambda g: (g["w"]*g["alpha_beta"]).sum()*1e4 - g["dw"].sum()*1.0
    ).rename("lin_bps").reset_index()           # maker cost c=1.0/unit
    lin["time"] = pd.to_datetime(lin["open_time"], utc=True)

    pr = pd.read_csv(REPO/"outputs/vBTC_final_simulation/per_cycle_pnl.csv")
    pr["time"] = pd.to_datetime(pr["time"], utc=True)
    m = lin.merge(pr[["time", "net_with_overlay_bps", "net_raw_bps", "fold"]],
                  on="time", how="inner")
    print(f"  overlapping cycles = {len(m)} "
          f"({m.time.min().date()} → {m.time.max().date()})", flush=True)

    # daily resample (neutralize 4h vs 24h-sleeve aggregation mismatch)
    m["day"] = m["time"].dt.floor("1D")
    dday = m.groupby("day").agg(lin=("lin_bps", "sum"),
                                v31=("net_with_overlay_bps", "sum"),
                                fold=("fold", "first")).reset_index()
    L, V = dday_L, dday_V = dday["lin"].to_numpy(), dday["v31"].to_numpy()
    corr = float(np.corrcoef(L, V)[0, 1])
    shL, shV = dsh(L), dsh(V)
    print(f"\n  DAILY: linear Sharpe={shL:+.2f}  V31 Sharpe={shV:+.2f}  "
          f"corr(lin,V31)={corr:+.3f}", flush=True)
    # per-fold correlation (stability)
    pf = dday = dday.groupby("fold").apply(
        lambda g: np.corrcoef(g["lin"], g["v31"])[0, 1]
        if len(g) > 3 else np.nan)
    print("  per-fold corr: " +
          " ".join(f"f{int(k)}={v:+.2f}" for k, v in pf.items()), flush=True)
    # fixed-weight blends (IN-SAMPLE — not nested; optimistic)
    print("\n  blend  w_lin  Sharpe  (vs V31-alone "
          f"{shV:+.2f}; IN-SAMPLE optimistic)", flush=True)
    best = (-9, None)
    for wl in (0.0, 0.1, 0.2, 0.25, 0.3, 0.4, 0.5, 0.75, 1.0):
        b = wl*L + (1-wl)*V
        s = dsh(b)
        flag = " <-- max" if s > best[0] else ""
        if s > best[0]:
            best = (s, wl)
        print(f"   w_lin={wl:<4} Sharpe={s:+.2f}{flag}", flush=True)
    lift = best[0] - shV
    # variance-min weight (in-sample)
    vL, vV, cLV = L.var(), V.var(), np.cov(L, V)[0, 1]
    w_mv = (vV - cLV) / (vL + vV - 2*cLV)
    w_mv = float(np.clip(w_mv, 0, 1))
    print(f"\n  in-sample best blend Sharpe {best[0]:+.2f} @ w_lin={best[1]} "
          f"(lift vs V31 {lift:+.2f}); var-min w_lin≈{w_mv:.2f}", flush=True)

    promising = bool(abs(corr) < 0.3 and lift >= 0.3)
    if promising:
        v = (f"PRELIMINARY PULSE: low corr ({corr:+.2f}) + in-sample blend "
             f"lift {lift:+.2f} ≥ +0.3. WARRANTS the clean harness rebuild + "
             f"nested-OOS §4 (this number is in-sample/contaminated — NOT a "
             f"verdict; nested + clean σ_idio could erase it).")
    else:
        why = (f"corr {corr:+.2f} (need |corr|<0.3)" if abs(corr) >= 0.3
               else f"in-sample blend lift {lift:+.2f} (need ≥+0.3)")
        v = (f"PRELIMINARY NO-GO: {why}. Even on the optimistic in-sample/"
             f"contaminated read, the linear book is NOT a clearly "
             f"diversifying/additive sleeve vs V3.1. A clean rebuild is "
             f"unlikely to rescue Success-B (the only real profitable path) "
             f"— strong kill signal; confirm with one clean nested §4 before "
             f"closing. Production LGBM unaffected.")
    print(f"\n  TRIAGE VERDICT: {v}", flush=True)
    pd.DataFrame([{"corr": corr, "lin_sh": shL, "v31_sh": shV,
                   "best_blend_sh": best[0], "best_w_lin": best[1],
                   "lift_vs_v31": lift, "w_mv": w_mv,
                   "promising": promising, "verdict": v}]).to_csv(
        OUTD/"ensemble_prelim.csv", index=False)
    print(f"\nSaved {OUTD}/ensemble_prelim.csv", flush=True)


if __name__ == "__main__":
    main()
