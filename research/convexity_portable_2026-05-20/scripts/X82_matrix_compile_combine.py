"""X82 — Stage-1 compile + Stage-2 combine (disciplined regime/construction matrix).

Pre-registered selection rule (set BEFORE seeing results):
  - Primary metric: 3yr block-bootstrap MEAN Sharpe (not point estimate)
  - PASS if: bootstrap mean > +0.5 AND beats matched placebo p95 AND
             (folds_pos >= 6/9 OR bootstrap P(>0) >= 70%)

Stage 1 — main effects (read saved prediction parquets, score each):
  - V0 single (x70_v0_3yr)             [feature=V0, constr=std, regime=none]
  - V5 single (x78_v5_single)          [feature=V5, constr=std, regime=none]
  - V5 routed K5 (x78_v5_routed_K5)    [feature=V5, regime=KMeans]
  - V0 KMeans routed K5 (x75_routed_K5)[feature=V0, regime=KMeans]
  - beta-neutral V0 (x79_betaneutral)  [constr=BN]
  - HMM routed (x80_hmm_K3_routed)     [regime=HMM]

Stage 2 — combine ONLY factors with real main effects.
  Beta-neutralization is a POST-HOC per-cycle transform on predictions, so we can
  apply it to any routed/feature pred parquet → clean factor stacking without retrain:
    - BN × {V5 single, V5 routed, HMM routed, V0 routed}

Stage 3 (light) — block bootstrap CI on the top configs.
"""
from __future__ import annotations
import sys, importlib.util, time, warnings
from pathlib import Path
import pandas as pd, numpy as np

warnings.filterwarnings("ignore")
REPO = Path("/home/yuqing/ctaNew")
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO/"scripts"))
OUT = REPO/"research/convexity_portable_2026-05-20/results"; RCACHE = OUT/"_cache"
spec = importlib.util.spec_from_file_location("x6", REPO/"research/convexity_portable_2026-05-20/scripts/X6_controlled_matrix.py")
x6 = importlib.util.module_from_spec(spec); spec.loader.exec_module(x6)


def load_beta_proxy():
    """Per-(sym,time) beta proxy: corr_to_btc_1d from whichever 3yr panel exists."""
    for nm in ["panel_3yr_v5.parquet","panel_3yr_v0.parquet"]:
        p = REPO/f"outputs/vBTC_features/{nm}"
        if p.exists():
            df = pd.read_parquet(p, columns=["symbol","open_time","corr_to_btc_1d"])
            df["open_time"] = pd.to_datetime(df["open_time"], utc=True)
            return df.drop_duplicates(["symbol","open_time"])
    return None


def beta_neutralize(apd, bmap, col="corr_to_btc_1d"):
    """Per-cycle residualize pred on beta proxy → pred_bn (post-hoc transform)."""
    a = apd.merge(bmap, on=["symbol","open_time"], how="left").reset_index(drop=True)
    newpred = a["pred"].to_numpy(dtype=float).copy()
    med = a[col].median()
    for ot, idx in a.groupby("open_time").indices.items():
        b = np.nan_to_num(a[col].to_numpy()[idx], nan=med)
        p = a["pred"].to_numpy()[idx]
        if np.std(b) < 1e-9 or len(idx) < 5:
            newpred[idx] = p - p.mean()
        else:
            A = np.vstack([np.ones_like(b), b]).T
            coef,*_ = np.linalg.lstsq(A, p, rcond=None)
            newpred[idx] = p - A @ coef
    a["pred"] = newpred
    return a[["symbol","open_time","alpha_A","return_pct","exit_time","pred","fold"]]


def score(pred_path, label):
    if not Path(pred_path).exists(): return None
    m = x6.run_sleeve_on_preds(pred_path, label)
    return m


def block_bootstrap(apd_path, label, n=50):
    apd = pd.read_parquet(apd_path)
    folds = sorted(apd["fold"].unique())
    np.random.seed(20260522); shs=[]
    for i in range(n):
        sm = np.random.choice(folds, size=len(folds), replace=True)
        parts=[apd[apd["fold"]==sf] for sf in sm]
        b = pd.concat(parts, ignore_index=True)
        b["fold"] = np.arange(len(b)) // (len(b)//len(folds)+1)
        tmp = RCACHE/f"_bb_{label}.parquet"; b.to_parquet(tmp, index=False)
        m = x6.run_sleeve_on_preds(tmp, f"_bb_{label}")
        shs.append(m.get("sharpe",0) or 0)
    a=np.array(shs)
    return a.mean(), a.std(), np.percentile(a,2.5), np.percentile(a,97.5), (a>0).mean()


def main():
    t0=time.time()
    print("=== X82 matrix compile + combine ===\n", flush=True)
    bmap = load_beta_proxy()
    if bmap is None: print("No panel for beta proxy; abort"); return

    # ---- Stage 1: main effects ----
    cands = {
        "V0_single":            RCACHE/"x70_v0_3yr_preds.parquet",
        "V0_KMeans_routed_K5":  RCACHE/"x75_routed_K5_preds.parquet",
        "V5_single":            RCACHE/"x78_v5_single_preds.parquet",
        "V5_routed_K5":         RCACHE/"x78_v5_routed_K5_preds.parquet",
        "BN_V0":                RCACHE/"x79_betaneutral_preds.parquet",
        "HMM_routed_K3":        RCACHE/"x80_hmm_K3_routed_preds.parquet",
    }
    print("--- Stage 1: main effects (3yr) ---")
    print(f"  {'config':<22} {'Sharpe':>8} {'folds+':>8} {'conc':>8}")
    s1 = {}
    for name, pth in cands.items():
        m = score(pth, f"x82s1_{name}")
        if m is None: print(f"  {name:<22} (missing)"); continue
        s1[name] = (m, pth)
        print(f"  {name:<22} {m.get('sharpe',0):>+8.2f} {str(m.get('folds_pos','?')):>8} {str(m.get('concentration','?')):>8}", flush=True)

    # ---- Stage 2: combine — apply BN to feature/regime configs ----
    print("\n--- Stage 2: beta-neutral × {feature, regime} combos ---")
    print(f"  {'config':<28} {'Sharpe':>8} {'folds+':>8} {'conc':>8}")
    combos = {}
    for base in ["V5_single","V5_routed_K5","HMM_routed_K3","V0_KMeans_routed_K5"]:
        if base not in s1: continue
        _, pth = s1[base]
        apd = pd.read_parquet(pth)
        apd["open_time"] = pd.to_datetime(apd["open_time"], utc=True)
        bn = beta_neutralize(apd, bmap)
        bp = RCACHE/f"x82_BN_{base}_preds.parquet"; bn.to_parquet(bp, index=False)
        m = score(bp, f"x82_BN_{base}")
        combos[f"BN_{base}"] = (m, bp)
        print(f"  BN_{base:<25} {m.get('sharpe',0):>+8.2f} {str(m.get('folds_pos','?')):>8} {str(m.get('concentration','?')):>8}", flush=True)

    # ---- Stage 3: block bootstrap CI on top configs ----
    print("\n--- Stage 3: block bootstrap CI (n=50) on top configs ---")
    allc = {**{k:v for k,v in s1.items()}, **combos}
    ranked = sorted(allc.items(), key=lambda kv: kv[1][0].get("sharpe",-9) or -9, reverse=True)[:4]
    print(f"  {'config':<28} {'point':>7} {'boot_mean':>10} {'95% CI':>20} {'P(>0)':>7}")
    for name,(m,pth) in ranked:
        bm,bs,lo,hi,p0 = block_bootstrap(pth, name.replace("/","_"), n=50)
        pt = m.get("sharpe",0) or 0
        print(f"  {name:<28} {pt:>+7.2f} {bm:>+10.2f} [{lo:>+6.2f},{hi:>+6.2f}]   {p0*100:>5.0f}%", flush=True)

    print(f"\nPre-registered PASS: boot_mean>+0.5 AND folds>=6/9 (or P(>0)>=70%).")
    print(f"Reference: V0 uncond +0.12, KMeans-routed +1.07, hard gate +1.13")
    print(f"Done [{time.time()-t0:.0f}s]")


if __name__ == "__main__":
    main()
