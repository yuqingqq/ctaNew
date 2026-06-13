"""§3.5 — TARGET-FRAMING test (the LAST independent hypothesis).

Plan flaw #8: the symmetric signed-residual-4h target may destroy the
asymmetric/magnitude structure the arc's own evidence implies (DDI:
short side carries the real alpha, long is mostly beta hedge; Step-80a:
squeeze/magnitude structure). §4 (the signed construction) is a
3/3-agent-confirmed honest negative. §3.5 asks the independent question:
is the *target framing*, not the construction, the limiter?

Pre-registered variants (4h-native; longer-horizon variant DEFERRED —
user fixed the goal to 4h and §3 horizon-sweep is gated off):
  T0  control       : signed tz_strict, naive sign book (== §4; reference).
  T1  short-only    : signed predictor, book holds ONLY the short leg
                      (DDI: shorts carry alpha) — equal-weight short
                      basket, no longs.
  T2  mag-conviction: a SECOND predictor on the magnitude target
                      |alpha_beta|/σ_strict; directional sign book
                      CONVICTION-WEIGHTED by per-cycle rank of predicted
                      magnitude (Step-80a + R2's legitimate fallback
                      construction, tested once, pre-registered).

Reuses harness_v3's 3-agent-cleared causal pieces UNCHANGED (preprocess,
strict_sigma_idio, model_envelope, fold masks) — harness untouched.
Same pre-registered gates, both cost rates (R2#1), matched-grid V3.1
(R2#2). Pre-registered: if NO variant clears Success-A OR Success-B →
close the linear line as a NARROW honest negative; wind down; no §5.
Production LGBM unaffected.
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
S4 = _imp("ensemble_s4", "linear_model/composite_study/ensemble_s4.py")
ANN, MAKER, COST, OOS = H.ANN, H.MAKER, H.COST, H.OOS
V31_CSV = REPO / "outputs/vBTC_sleeve_horizon/per_cycle_V3.1_equal6_baseline.csv"
OUTD = REPO / "linear_model/composite_study/results"
OUTD.mkdir(parents=True, exist_ok=True)
SA_GATE = 1.5            # Success-A standalone net Sharpe @ MAKER
SB_GATE = 0.30           # Success-B nested blend lift vs matched V3.1
WINDOW_CAP = 0.60
sh = S4.sh


def book_from_weights(df, wcol):
    """Per-cycle net bps from an arbitrary per-(symbol,cycle) weight col.
    Same accounting/units as harness_v3.linear_book (Σw·αβ·1e4 − Σ|Δw|·c)."""
    out = []
    for cost, clab in ((MAKER, "MAKER"), (COST, "COST2.25")):
        f = df.sort_values(["symbol", "open_time"]).copy()
        f["dw"] = f.groupby("symbol")[wcol].diff().abs().fillna(
            f[wcol].abs())
        p = f.groupby(["open_time", "fold"]).apply(
            lambda g: pd.Series({
                "gross": (g[wcol] * g["alpha_beta"]).sum() * 1e4,
                "cost": g["dw"].sum() * cost})).reset_index()
        p["net"] = p["gross"] - p["cost"]
        out.append((clab, p.sort_values("open_time").reset_index(drop=True)))
    return dict(out)


def eval_gate(pf_book, v, label):
    """Both pre-registered gates at BOTH costs. Returns a result dict +
    prints the line. Success-A = standalone net Sharpe>+1.5 @ MAKER, CI
    excl 0, ≥⅔ folds+, no fold>60%. Success-B = nested var-min blend lift
    ≥+0.30 vs matched-grid V3.1, paired CI excl 0, no fold>60%."""
    res = {}
    for clab, lb in pf_book.items():
        m = lb.rename(columns={"net": "lin"}).merge(
            v, on="open_time", how="inner").sort_values(
            "open_time").reset_index(drop=True)
        v31m = sh(m["v31"].to_numpy())
        lin_sh = sh(m["lin"].to_numpy())
        lo, hi = H.block_bootstrap_ci(  # CI on the standalone Sharpe
            m["lin"].to_numpy(), sh, block=S4.BOOT_BLOCK,
            n_boot=S4.N_BOOT)
        efolds = sorted(m["fold"].unique())
        fold_sh = m.groupby("fold")["lin"].apply(
            lambda g: sh(g.to_numpy()))
        fpos = int((fold_sh > 0).sum())
        fpnl = m.groupby("fold")["lin"].sum()
        # bounded concentration: largest single fold as a share of TOTAL
        # ABSOLUTE per-fold PnL (robust to mixed-sign folds summing ≈0,
        # unlike dividing by the signed total which can exceed 100%).
        share = float(fpnl.abs().max() / fpnl.abs().sum()) \
            if fpnl.abs().sum() > 1e-9 else np.nan
        # Success-B: nested var-min blend (same machinery as §4)
        w = np.zeros(len(m))
        for i, f in enumerate(efolds):
            msk = (m["fold"] == f).to_numpy()
            if i == 0:
                continue
            pr = m[m["fold"].isin(efolds[:i])]
            L0, V0 = pr["lin"].to_numpy(), pr["v31"].to_numpy()
            den = L0.var() + V0.var() - 2 * np.cov(L0, V0)[0, 1]
            wv = ((V0.var() - np.cov(L0, V0)[0, 1]) / den
                  if abs(den) > 1e-12 else 0.0)
            w[msk] = float(np.clip(wv, 0.0, 1.0))
        blend = w * m["lin"].to_numpy() + (1 - w) * m["v31"].to_numpy()
        lift = sh(blend) - v31m
        blo, bhi = S4.paired_lift_ci(m["lin"].to_numpy(),
                                     m["v31"].to_numpy(), w)
        saA = bool(clab == "MAKER" and lin_sh > SA_GATE and lo > 0
                   and fpos >= int(np.ceil(len(efolds) * 2 / 3))
                   and share <= WINDOW_CAP)
        saB = bool(lift >= SB_GATE and blo > 0 and share <= WINDOW_CAP)
        print(f"    [{clab:8}] standalone Sh={lin_sh:+.3f} "
              f"CI[{lo:+.2f},{hi:+.2f}] folds+={fpos}/{len(efolds)} "
              f"share={share:.0%} | nested-blend lift={lift:+.3f} "
              f"CI[{blo:+.2f},{bhi:+.2f}] (V31m={v31m:+.2f}) | "
              f"A={'PASS' if saA else '·'} B={'PASS' if saB else '·'}",
              flush=True)
        res[clab] = dict(label=label, cost=clab, lin_sh=lin_sh,
                         ci_lo=lo, ci_hi=hi, folds_pos=fpos,
                         n_folds=len(efolds), share=share, lift=lift,
                         blift_lo=blo, blift_hi=bhi, v31m=v31m,
                         passA=saA, passB=saB)
    return res


def main():
    print("=" * 94, flush=True)
    print("  §3.5 — TARGET-FRAMING (last independent hypothesis; "
          "pre-registered gate)", flush=True)
    print("=" * 94, flush=True)
    t0 = time.time()
    dec, folds, btc, pan = H.load_panel()
    if not H.run_selfchecks(dec):
        print("\n  ABORT — self-checks failed.", flush=True)
        return
    F_all, drop_all = H.preprocess(dec)
    feats = list(F_all.columns)
    sig = H.strict_sigma_idio(dec)
    base = dec.copy()
    base["sig"] = sig
    base["y_dir"] = (base["alpha_beta"] / base["sig"]).clip(-5, 5)
    base["y_mag"] = (base["alpha_beta"].abs() / base["sig"]).clip(0, 5)
    elig = ((~drop_all) & base["sig"].notna() & (base["sig"] > 1e-12)
            & base["y_dir"].notna())
    recs = []
    for fid in OOS:
        fo = folds[fid]
        tr = (dec["open_time"] < fo["cal_start"]) & elig
        ca = ((dec["open_time"] >= fo["cal_start"]) &
              (dec["open_time"] < fo["cal_end"]) &
              (dec["exit_time"] < fo["test_start"]) & elig)
        te = (dec["fold"] == fid) & elig
        if tr.sum() < 800 or ca.sum() < 50 or te.sum() < 20:
            continue
        Xtr, Xca, Xte = (F_all.loc[tr, feats].to_numpy(float),
                         F_all.loc[ca, feats].to_numpy(float),
                         F_all.loc[te, feats].to_numpy(float))
        pd_ = H.model_envelope(Xtr, base.loc[tr, "y_dir"].to_numpy(float),
                               Xca, base.loc[ca, "y_dir"].to_numpy(float),
                               Xte)
        pm_ = H.model_envelope(Xtr, base.loc[tr, "y_mag"].to_numpy(float),
                               Xca, base.loc[ca, "y_mag"].to_numpy(float),
                               Xte)
        r = base.loc[te, ["symbol", "open_time", "fold",
                          "alpha_beta"]].copy()
        r["dir"] = pd_["ridge_best"]
        r["dir_g"] = pd_["lgbm_es"]
        r["mag"] = pm_["ridge_best"]
        r["mag_g"] = pm_["lgbm_es"]
        recs.append(r)
        print(f"  fold {fid}: tr={tr.sum()} te={te.sum()}", flush=True)
    pf = pd.concat(recs, ignore_index=True)
    print(f"  predicted folds {sorted(pf.fold.unique())} "
          f"({pf.open_time.nunique()} cyc)", flush=True)

    v = pd.read_csv(V31_CSV)
    v["open_time"] = pd.to_datetime(v["time"], utc=True)
    v = v[["open_time", "net_pnl_bps"]].rename(
        columns={"net_pnl_bps": "v31"})

    nC = pf.groupby("open_time")["symbol"].transform("count")
    all_res = []
    for dcol, mcol, mtag in (("dir", "mag", "ridge"),
                             ("dir_g", "mag_g", "lgbm")):
        print(f"\n  ══ predictor = {mtag} ══", flush=True)
        # T0 control: naive sign, equal weight
        pf["wT0"] = np.sign(pf[dcol]).replace(0, 1.0) / nC
        # T1 short-only: hold ONLY the per-cycle short basket
        sh_mask = pf.groupby("open_time")[dcol].transform(
            lambda s: s <= s.median())
        nS = (sh_mask.astype(float)).groupby(pf["open_time"]).transform(
            "sum").replace(0, np.nan)
        pf["wT1"] = np.where(sh_mask, -1.0 / nS, 0.0)
        pf["wT1"] = pf["wT1"].fillna(0.0)
        # T2 mag-conviction: sign(dir) weighted by per-cycle rank(mag)
        mr = pf.groupby("open_time")[mcol].rank(pct=True)
        raw = np.sign(pf[dcol]).replace(0, 1.0) * mr
        den = raw.abs().groupby(pf["open_time"]).transform("sum").replace(
            0, np.nan)
        pf["wT2"] = (raw / den).fillna(0.0)
        for tcol, tlab in (("wT0", "T0-control-sign"),
                           ("wT1", "T1-short-only"),
                           ("wT2", "T2-mag-conviction")):
            print(f"  {tlab}:", flush=True)
            bk = book_from_weights(pf[["symbol", "open_time", "fold",
                                       "alpha_beta", tcol]], tcol)
            res = eval_gate(bk, v, f"{mtag}/{tlab}")
            all_res.extend(res.values())

    R = pd.DataFrame(all_res)
    R.to_csv(OUTD / "target_framing_s35.csv", index=False)
    anyA = bool(R["passA"].any())
    anyB = bool(R["passB"].any())
    win = R[R["passA"] | R["passB"]]
    print("\n" + "=" * 94, flush=True)
    if anyA or anyB:
        verdict = (f"§3.5 SIGNAL — {len(win)} variant(s) clear a "
                   f"pre-registered gate: "
                   + "; ".join(f"{w.label}@{w.cost}"
                               f"(A={w.passA},B={w.passB})"
                               for _, w in win.iterrows())
                   + ". The TARGET FRAMING (not the construction) was a "
                   f"limiter. PROCEED to §7 strict nested forward "
                   f"validation on the winning framing before any "
                   f"adoption. Do NOT over-claim — §7 is the real bar.")
    else:
        verdict = ("§3.5 NEGATIVE — no target framing (short-only, "
                   "magnitude-conviction; ridge & lgbm; both cost rates) "
                   "clears Success-A or Success-B vs matched-grid V3.1 "
                   "(+2.747). Combined with the 3/3-confirmed §4 FAIL and "
                   "the gated-off §3/§5: the linear β-residual line is a "
                   "NARROW, HONEST negative on free 4h Binance-perp data — "
                   "explicitly NOT 'information-bounded' (a sharper target "
                   "/ orthogonal data / longer horizon are untested), but "
                   "no remaining in-scope independent lever. WIND DOWN the "
                   "linear line; no §5. Production LGBM unaffected.")
    print(f"  VERDICT: {verdict}", flush=True)
    pd.DataFrame([{"anyA": anyA, "anyB": anyB, "verdict": verdict}]
                 ).to_csv(OUTD / "target_framing_s35_verdict.csv",
                          index=False)
    print(f"\nSaved {OUTD}/target_framing_s35*.csv  "
          f"Total {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
