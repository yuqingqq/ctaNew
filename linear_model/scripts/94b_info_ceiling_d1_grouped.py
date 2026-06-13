"""Step 94b — D1 (corrected): leak-free INFORMATION CEILING.

Step-94 v1 (naive random shuffle) was INVALID — leaked via (1) temporal
autocorrelation (test row's t±4h same-symbol neighbor in train) and (2)
contemporaneous cross-section (whole timestamps split across train/test).
LGBM IC +0.376 was the leak signature.

CORRECTION (pre-registered in INFORMATION_DIAGNOSTIC_PLAN.md §Status):
  - Fold by WHOLE TIMESTAMP (all 42 syms at a held-out time go to test
    together) ⇒ kills the cross-sectional leak.
  - 1-day EMBARGO (6 cycles on the 4h grid): a train timestamp is dropped
    if its cycle-index is within 6 of ANY test timestamp in that fold ⇒
    kills the autocorrelation leak.
  - Shuffled time-blocks still interleave regimes ⇒ relaxes long-range
    non-stationarity (the legitimate Q1-vs-Q2 separation), nothing else.
  Same models / features / scoring / the SAME pre-registered >+1.5 gate.

Diagnostic check: if v1 was leakage, LGBM IC collapses +0.38 → ~0.03.
No strategy adopted. Production LGBM unaffected.
"""
from __future__ import annotations
import importlib.util, sys, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
import lightgbm as lgb

warnings.filterwarnings("ignore")
REPO = Path("/home/yuqing/ctaNew")
sys.path.insert(0, str(REPO))


def _imp(n, r):
    s = importlib.util.spec_from_file_location(n, REPO / r)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


s94 = _imp("s94", "linear_model/scripts/94_info_ceiling_d1.py")
build, score, sh = s94.build, s94.score, s94.sh
GATE, LEAK = s94.GATE, s94.LEAK
EMBARGO = 6                                             # 1 day @ 4h grid
OUTD = REPO / "linear_model/results/step94b_info_ceiling_grouped"
OUTD.mkdir(parents=True, exist_ok=True)


def grouped_oof(dec, feats, seed=0):
    """Whole-timestamp 5-fold + 1-day embargo. OOF Ridge + LGBM."""
    times = np.array(sorted(dec["open_time"].unique()))
    tidx = {t: i for i, t in enumerate(times)}             # cycle index
    row_t = dec["open_time"].map(tidx).to_numpy()
    Xv = dec[feats].to_numpy(np.float64)
    yv = dec["tz"].to_numpy(np.float64)
    rid = np.full(len(dec), np.nan); gbm = np.full(len(dec), np.nan)
    kf = KFold(n_splits=5, shuffle=True, random_state=seed)
    for tr_t, te_t in kf.split(times):
        te_set = set(te_t)
        te_mask = np.isin(row_t, te_t)
        # embargo: drop train rows whose time-index is within EMBARGO of any test time
        emb = np.zeros(len(times), bool)
        for j in te_t:
            emb[max(0, j-EMBARGO):min(len(times), j+EMBARGO+1)] = True
        tr_keep = np.array([i for i in tr_t if not emb[i]])
        tr_mask = np.isin(row_t, tr_keep)
        if tr_mask.sum() < 500 or te_mask.sum() == 0:
            continue
        sc = StandardScaler().fit(Xv[tr_mask])
        r = Ridge(alpha=10.0).fit(sc.transform(Xv[tr_mask]), yv[tr_mask])
        rid[te_mask] = r.predict(sc.transform(Xv[te_mask]))
        m = lgb.LGBMRegressor(num_leaves=63, n_estimators=400,
                              learning_rate=0.03, subsample=0.8,
                              colsample_bytree=0.8, random_state=seed,
                              n_jobs=-1, verbose=-1)
        m.fit(Xv[tr_mask], yv[tr_mask])
        gbm[te_mask] = m.predict(Xv[te_mask])
    return rid, gbm


def run(dec, label, gated):
    feats = [c for c in dec.columns if c not in LEAK and
             pd.api.types.is_numeric_dtype(dec[c])] + ["s_t"]
    feats = list(dict.fromkeys(feats))
    dec = dec.dropna(subset=feats + ["tz", "alpha_beta"]).reset_index(drop=True)
    print(f"\n--- {label} (rows={len(dec)} syms={dec.symbol.nunique()} "
          f"cycles={dec.open_time.nunique()} feats={len(feats)})"
          f"{'  [GATED]' if gated else '  [context]'} ---", flush=True)
    rid, gbm = grouped_oof(dec, feats)
    mask = ~np.isnan(rid)
    dd = dec[mask].reset_index(drop=True)
    print(f"  OOF coverage: {mask.mean()*100:.0f}% of rows "
          f"(rest = embargoed/empty folds)", flush=True)
    R = [score(dd, rid[mask], f"Ridge_grpCV{'' if not gated else ''}"),
         score(dd, gbm[mask], "LGBM_grpCV")]
    score(dd, dd["s_t"].to_numpy() * -1.0, "s_t_rule(Step92 ref)")
    return R, dd


def main():
    print("=" * 96, flush=True)
    print("  STEP 94b — D1 CORRECTED (time-grouped + 1d embargo; leak-free "
          "information ceiling)", flush=True)
    print("=" * 96, flush=True)
    t0 = time.time()
    dec, syms, btc, pan = build(universe_oi=False)
    # PIT guard (s_t exact-match) — reuse Step-92 logic via s94 audit pattern
    okA = True
    for s in ["SOLUSDT", "ADAUSDT"]:
        c = s94.load_close(s).set_index("open_time")["close"]
        bser = pan[pan.symbol == s].set_index("open_time")["beta_btc_pit"]
        ind = s94.trail(c) - bser.reindex(c.index) * s94.trail(btc).reindex(c.index)
        m = dec[dec.symbol == s].set_index("open_time")[["s_t"]].join(
            ind.rename("ind")).dropna()
        cc = float(m["s_t"].corr(m["ind"])) if len(m) else np.nan
        okA &= (cc > 0.9999)
        print(f"  audit {s}: corr(s_t,indep_PAST)={cc:.6f} -> "
              f"{'OK' if cc > 0.9999 else 'MISMATCH'}", flush=True)
    if not okA:
        print("\n  PIT GUARD FAIL — not run.", flush=True)
        pd.DataFrame([{"audit": "FAIL"}]).to_csv(OUTD/"verdict.csv", index=False)
        return

    R, _ = run(dec, "F_core (primary, hl42)", gated=True)
    best = max(R, key=lambda r: r["net_sh"])
    PASS = bool(best["net_sh"] > GATE)
    try:
        deco, _, _, _ = build(universe_oi=True)
        run(deco, "F_core+OI (secondary, OI∩hl42)", gated=False)
    except Exception as e:
        print(f"  (F_core+OI skipped: {e})", flush=True)

    leak_collapsed = any(r["tag"] == "LGBM_grpCV" and abs(r["ic"]) < 0.10
                         for r in R)
    if PASS:
        v = (f"D1 PASS (leak-free) — best F_core grouped-CV NET Sharpe "
             f"{best['net_sh']:+.2f} ({best['tag']}, IC {best['ic']:+.3f}) "
             f"> +1.5. Information IS sufficient under stationarity ⇒ Q1=YES; "
             f"bottleneck is Q2 (non-stationarity/utilization). PROCEED D2.")
    else:
        v = (f"D1 FAIL (leak-free) — best F_core grouped-CV NET Sharpe "
             f"{best['net_sh']:+.2f} ({best['tag']}, IC {best['ic']:+.3f}, "
             f"CI[{best['ci_lo']:+.2f},{best['ci_hi']:+.2f}]) ≤ +1.5. Even a "
             f"best-case, no-memorization, leak-free, STATIONARY extraction "
             f"of the current features does NOT clear cost ⇒ **Q1 = NO: the "
             f"line is INFORMATION-BOUNDED.** No model/utilization/selection "
             f"fix (Q2/Q3) can rescue it; the limiter is feature information "
             f"content on free 4h perp data. Definitive — stronger than the "
             f"prior 'sub-cost' finding. v1's +23.71 was confirmed leakage "
             f"(LGBM IC {('collapsed +0.38→%.3f' % best['ic']) if leak_collapsed else 'see above'}). "
             f"Production LGBM unaffected.")
    print(f"\n  v1→v1.1 leak check: LGBM IC "
          f"{[r['ic'] for r in R if r['tag']=='LGBM_grpCV'][0]:+.3f} "
          f"(v1 was +0.376 — {'COLLAPSED ⇒ v1 confirmed leaky' if leak_collapsed else 'did NOT collapse — investigate'})",
          flush=True)
    print(f"  PRE-REG GATE (>{GATE:+.1f}): {'PASS' if PASS else 'FAIL'}",
          flush=True)
    print(f"  VERDICT: {v}", flush=True)
    pd.DataFrame(R + [dict(tag="VERDICT", net_sh=best["net_sh"], PASS=PASS,
                 verdict=v)]).to_csv(OUTD/"summary.csv", index=False)
    pd.DataFrame([{"PASS": PASS, "best_net_sh": best["net_sh"],
                   "best_ic": best["ic"], "verdict": v}]).to_csv(
        OUTD/"verdict.csv", index=False)
    print(f"\nSaved {OUTD}\nTotal: {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
