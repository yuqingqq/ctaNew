"""§4 — clean nested ensemble-with-production (kill-fast; THE verdict).

Built on harness_v3 (causally clean per R1+R3 indep repro). R2's two
BLOCKING false-positive vectors are neutralized HERE:

  R2#1 cost asymmetry — linear book reported at BOTH MAKER (~1bps/unit,
       HL-maker assumption) AND full COST=2.25/unit (the rate V3.1's CSV
       actually baked in). The verdict must hold at COST=2.25, not be
       rescued by the maker discount alone.
  R2#2 fold-coverage / baseline survivorship — walk_forward covers only
       the OOS folds with strictly-past train data (3–9, ~1260 cyc). The
       V3.1 baseline is recomputed on the IDENTICAL matched cycle grid
       (NOT the headline +2.229 over all 1620). The prelim's exact sin.

Pre-registered gate (NOT moved): Success-B = nested-OOS blend Sharpe lift
≥ +0.30 vs the matched-grid V3.1 baseline AND paired block-bootstrap CI
on the lift excludes 0 AND no single fold > 60% of the cumulative lift
PnL. Nested weight = variance-min (parameter-free closed form, fit on
strictly-prior evaluable folds only; first evaluable fold ⇒ w=0). Primary
member = ridge_best (pre-registered: this IS the "linear" line; lgbm_es
secondary context). Pre-registered kill: lift CI includes 0 ⇒ narrow
honest negative, do NOT run §5. Production LGBM unaffected.
"""
from __future__ import annotations
import importlib.util
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
REPO = Path("/home/yuqing/ctaNew")
sys.path.insert(0, str(REPO))


def _imp(n, r):
    s = importlib.util.spec_from_file_location(n, REPO / r)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


H = _imp("harness_v3", "linear_model/composite_study/harness_v3.py")
ANN, MAKER, COST = H.ANN, H.MAKER, H.COST
V31_CSV = REPO / "outputs/vBTC_sleeve_horizon/per_cycle_V3.1_equal6_baseline.csv"
OUTD = REPO / "linear_model/composite_study/results"
OUTD.mkdir(parents=True, exist_ok=True)
BOOT_BLOCK = 12          # 2-day embargo on the 4h grid (6 cyc/day)
N_BOOT = 2000
LIFT_GATE = 0.30
WINDOW_CAP = 0.60


def sh(x):
    x = np.asarray(x, float)
    return float(x.mean() / x.std(ddof=1) * ANN) if x.std(ddof=1) > 1e-12 \
        else np.nan


def paired_lift_ci(lin, v31, w_series, block=BOOT_BLOCK, n_boot=N_BOOT,
                   seed=0):
    """Block-bootstrap the PAIRED Sharpe lift = Sharpe(w·lin+(1-w)·v31) −
    Sharpe(v31). Resample the same cycle blocks jointly for both legs."""
    lin = np.asarray(lin, float)
    v31 = np.asarray(v31, float)
    w = np.asarray(w_series, float)
    n = len(lin)
    rng = np.random.default_rng(seed)
    nb = int(np.ceil(n / block))
    out = np.empty(n_boot)
    for b in range(n_boot):
        st = rng.integers(0, n - block + 1, nb)
        idx = np.concatenate([np.arange(s, s + block) for s in st])[:n]
        bl = w[idx] * lin[idx] + (1 - w[idx]) * v31[idx]
        out[b] = sh(bl) - sh(v31[idx])
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def main():
    print("=" * 94, flush=True)
    print("  §4 — CLEAN NESTED ENSEMBLE-WITH-PRODUCTION (kill-fast; THE "
          "verdict)", flush=True)
    print("=" * 94, flush=True)
    t0 = time.time()
    dec, folds, btc, pan = H.load_panel()
    if not H.run_selfchecks(dec):
        print("\n  ABORT — self-checks failed.", flush=True)
        return
    print("\n  walk-forward (per-fold strict-past σ_idio + envelope)...",
          flush=True)
    pf = H.walk_forward(dec, folds, members=("ridge_best", "lgbm_es"))
    if not len(pf):
        print("  no predictions — abort.", flush=True)
        return
    cov_folds = sorted(pf["fold"].unique())
    print(f"  predicted folds = {cov_folds} "
          f"({pf.open_time.nunique()} cycles, {len(pf)} rows)", flush=True)

    # ---- V3.1 honest-forward, MATCHED to the linear book's grid ----
    v = pd.read_csv(V31_CSV)
    v["time"] = pd.to_datetime(v["time"], utc=True)
    # keep V3.1's own fold only for the headline; the matched-grid join
    # uses the HARNESS fold (from linear_book) so nested splits are
    # consistent with walk_forward's fold definition.
    v_hl = v[["fold", "net_pnl_bps"]].rename(columns={"net_pnl_bps": "v31"})
    v = v[["time", "net_pnl_bps"]].rename(
        columns={"time": "open_time", "net_pnl_bps": "v31"})
    v31_full = sh(v["v31"].to_numpy())                       # headline
    print(f"\n  V3.1 headline (all {len(v)} cyc, folds "
          f"{sorted(v_hl.fold.unique())}) Sharpe = {v31_full:+.3f} "
          f"(== documented +2.23)", flush=True)

    rows = []
    for member in ("ridge_best", "lgbm_es"):
        for cost, clab in ((MAKER, "MAKER"), (COST, "COST2.25")):
            lb = H.linear_book(pf, member, cost=cost)        # per-cycle net
            lb = lb.rename(columns={"net": "lin"})
            m = lb.merge(v, on="open_time", how="inner").sort_values(
                "open_time").reset_index(drop=True)
            # restrict V3.1 to the EXACT matched cycle grid (R2#2):
            v31_matched = sh(m["v31"].to_numpy())
            lin_sh = sh(m["lin"].to_numpy())
            corr_all = float(np.corrcoef(m["lin"], m["v31"])[0, 1])
            pf_corr = m.groupby("fold").apply(
                lambda g: np.corrcoef(g["lin"], g["v31"])[0, 1]
                if len(g) > 3 else np.nan)
            # ---- nested-OOS variance-min weight (fit on PRIOR folds) ----
            efolds = sorted(m["fold"].unique())
            w_ser = np.zeros(len(m))
            for i, f in enumerate(efolds):
                msk = (m["fold"] == f).to_numpy()
                if i == 0:
                    w_ser[msk] = 0.0                          # can't learn
                    continue
                prior = m[m["fold"].isin(efolds[:i])]
                L0, V0 = prior["lin"].to_numpy(), prior["v31"].to_numpy()
                vL, vV = L0.var(), V0.var()
                cLV = np.cov(L0, V0)[0, 1]
                den = vL + vV - 2 * cLV
                w = (vV - cLV) / den if abs(den) > 1e-12 else 0.0
                w_ser[msk] = float(np.clip(w, 0.0, 1.0))
            blend_nested = w_ser * m["lin"].to_numpy() + \
                (1 - w_ser) * m["v31"].to_numpy()
            nested_sh = sh(blend_nested)
            lift = nested_sh - v31_matched
            ci_lo, ci_hi = paired_lift_ci(m["lin"].to_numpy(),
                                          m["v31"].to_numpy(), w_ser)
            # 60%-window: per-fold cumulative lift-PnL share
            m2 = m.assign(diff=blend_nested - m["v31"].to_numpy())
            fold_pnl = m2.groupby("fold")["diff"].sum()
            tot = fold_pnl.sum()
            max_share = float((fold_pnl.abs() / abs(tot)).max()) \
                if abs(tot) > 1e-9 else np.nan
            folds_pos = int((fold_pnl > 0).sum())
            # in-sample best blend = labelled UPPER BOUND (NOT the verdict)
            ub = max(sh(wl * m["lin"].to_numpy() +
                        (1 - wl) * m["v31"].to_numpy())
                     for wl in np.linspace(0, 1, 21))
            tag = f"{member}/{clab}"
            print(f"\n  ── {tag} ──", flush=True)
            print(f"    matched grid: {len(m)} cyc, folds {efolds} | "
                  f"V3.1_matched Sharpe={v31_matched:+.3f} "
                  f"(headline {v31_full:+.2f}); linear standalone "
                  f"Sharpe={lin_sh:+.3f}", flush=True)
            print(f"    corr(lin,V31) all={corr_all:+.3f} | per-fold "
                  + " ".join(f"f{int(k)}={x:+.2f}"
                             for k, x in pf_corr.items()), flush=True)
            print(f"    NESTED var-min blend Sharpe={nested_sh:+.3f}  "
                  f"LIFT={lift:+.3f}  paired-CI[{ci_lo:+.3f},{ci_hi:+.3f}]",
                  flush=True)
            print(f"    folds+={folds_pos}/{len(efolds)}  "
                  f"max-fold lift share={max_share:.0%} "
                  f"(cap {WINDOW_CAP:.0%})  | in-sample UB="
                  f"{ub:+.3f} (NOT verdict)", flush=True)
            rows.append(dict(member=member, cost=clab, n=len(m),
                             v31_matched=v31_matched, lin_sh=lin_sh,
                             corr=corr_all, nested_sh=nested_sh,
                             lift=lift, ci_lo=ci_lo, ci_hi=ci_hi,
                             folds_pos=folds_pos, n_folds=len(efolds),
                             max_share=max_share, insample_ub=ub))

    # ---- pre-registered verdict on PRIMARY = ridge_best @ COST2.25 ----
    R = pd.DataFrame(rows)
    prim = R[(R.member == "ridge_best") & (R.cost == "COST2.25")].iloc[0]
    prim_mk = R[(R.member == "ridge_best") & (R.cost == "MAKER")].iloc[0]
    g_lift = bool(prim["lift"] >= LIFT_GATE)
    g_ci = bool(prim["ci_lo"] > 0.0)
    g_win = bool(prim["max_share"] <= WINDOW_CAP)
    g_folds = bool(prim["folds_pos"] >= int(np.ceil(prim["n_folds"] * 2 / 3)))
    passB = g_lift and g_ci and g_win
    print("\n" + "=" * 94, flush=True)
    print(f"  PRE-REGISTERED GATE (primary = ridge_best @ COST=2.25, "
          f"the rate V3.1's CSV baked in):", flush=True)
    print(f"    lift {prim['lift']:+.3f} ≥ +{LIFT_GATE:.2f} : "
          f"{'PASS' if g_lift else 'FAIL'}", flush=True)
    print(f"    paired-CI [{prim['ci_lo']:+.3f},{prim['ci_hi']:+.3f}] "
          f"excludes 0 : {'PASS' if g_ci else 'FAIL'}", flush=True)
    print(f"    no fold >{WINDOW_CAP:.0%} of lift "
          f"({prim['max_share']:.0%}) : {'PASS' if g_win else 'FAIL'}",
          flush=True)
    print(f"    folds+ {int(prim['folds_pos'])}/{int(prim['n_folds'])} "
          f"(≥⅔) : {'PASS' if g_folds else 'FAIL'} (context)", flush=True)
    print(f"    [robustness] same member @ MAKER lift "
          f"{prim_mk['lift']:+.3f} CI[{prim_mk['ci_lo']:+.3f},"
          f"{prim_mk['ci_hi']:+.3f}]", flush=True)
    if passB:
        verdict = (f"§4 SUCCESS-B PASS — ridge_best linear book is a "
                   f"portfolio-accretive sleeve: nested var-min blend lift "
                   f"{prim['lift']:+.3f} over matched-grid V3.1 "
                   f"({prim['v31_matched']:+.2f}), paired-CI excludes 0, "
                   f"no fold >60%. Holds at COST=2.25 (not a maker "
                   f"artifact). PROCEED to §7 strict nested forward "
                   f"validation before any adoption.")
    else:
        why = ("lift<+0.30" if not g_lift else
               "paired-CI includes 0" if not g_ci else "a fold >60%")
        verdict = (f"§4 SUCCESS-B FAIL ({why}) — at the honest cost rate "
                   f"V3.1 actually paid, the linear book does NOT add a "
                   f"robust sleeve. Per the PRE-REGISTERED KILL: the "
                   f"linear-idio line is a NARROW, HONEST negative (NOT "
                   f"'information-bounded'); do NOT run §5. §3.5 "
                   f"target-framing is the only remaining independent "
                   f"hypothesis. Production LGBM unaffected.")
    print(f"\n  VERDICT: {verdict}", flush=True)
    R.to_csv(OUTD / "ensemble_s4.csv", index=False)
    pd.DataFrame([{"passB": passB, "lift": prim["lift"],
                   "ci_lo": prim["ci_lo"], "ci_hi": prim["ci_hi"],
                   "verdict": verdict}]).to_csv(
        OUTD / "ensemble_s4_verdict.csv", index=False)
    print(f"\nSaved {OUTD}/ensemble_s4*.csv  Total {time.time()-t0:.0f}s",
          flush=True)


if __name__ == "__main__":
    main()
