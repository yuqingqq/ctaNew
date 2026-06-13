"""Step 90 (Stage 4): rigorous per-symbol OI evaluation — the determinant.

A per-symbol model is a per-symbol TIMING strategy (each symbol takes a
position in itself from its own prediction), so it is evaluated as an
equal-weight portfolio of 23 per-symbol sleeves at the 4h NON-overlapping
decision cadence, net of cost — NOT the cross-sectional K=3 book.

Variants:
  ridge_insample : per-symbol RidgeCV (Step-89 pred_ridge) — in-sample-fit ref
  signed_train   : per-symbol signed-equal, signs from FULL fold-k train IC
                   (Step-89 pred_signed) — estimator-robustness ref
  signed_nested  : per-symbol signed-equal, each fold's signs from STRICTLY
                   -PAST folds only (the Step-88 honest analog, per symbol)
                   — THE DECISIVE variant (no hindsight, no in-sample fit)
  placebo        : per-symbol RANDOM sign, matched turnover, 150 seeds

PRE-REGISTERED GATE (fixed before run; the per-symbol OI line is a GENUINE
edge iff signed_nested satisfies ALL):
  G1 portfolio net-of-cost annualized Sharpe block-bootstrap CI excludes 0
  G2 signed_nested net Sharpe > matched random-sign placebo p95 (150)
  G3 estimator-consistent: signed_train same sign & >0 AND ridge_insample
     does NOT >> signed_nested (in-sample >> nested = the K3 collapse)
  G4 not concentrated: >=60% of 23 symbols individually net-positive AND
     drop-top-2-symbol portfolio still net-positive
PASS all -> FIRST genuine generalizing edge of the investigation; the
per-symbol + OI hypothesis (more-stationary per-symbol relation, V3 77%)
holds under honest evaluation. ANY fail -> per-symbol OI is the same
non-stationary/fit artifact, just per-symbol; record honestly. No further
backtest beyond this portfolio sim.
"""
from __future__ import annotations
import importlib.util, sys, time, warnings
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


s64 = _imp("s64", "linear_model/scripts/64_meanrev_v2_backtest.py")
s89 = _imp("s89", "linear_model/scripts/89_per_symbol_oi_model.py")
s59 = s64.s59
from ml.research.alpha_v4_xs_1d import _multi_oos_splits, _slice
from ml.research.alpha_v4_xs import block_bootstrap_ci

OUTD = REPO / "linear_model/results/step90_per_symbol_oi_rigor"
OUTD.mkdir(parents=True, exist_ok=True)
PRED = REPO / "linear_model/results/step89_per_symbol_oi/per_symbol_oi_preds.parquet"
OOS = list(range(1, 10))
BLOCK = 48                                  # 4h decision cadence (bars)
COST = s64.COST                             # bps per unit |Δw| (=2.25)
ANN = np.sqrt(365.0 * 6.0)                  # 6 non-overlap 4h cycles/day
NF = [c + "_n" for c in s89.FEATS]
N_PLACEBO = 150


def decision_sample(df: pd.DataFrame) -> pd.DataFrame:
    """One row per symbol per 4h decision cycle (non-overlapping)."""
    g = sorted(df["open_time"].unique())[::BLOCK]
    return df[df["open_time"].isin(set(g))].copy()


def portfolio_net(pos_df: pd.DataFrame) -> pd.DataFrame:
    """pos_df: symbol, open_time, fold, pos, alpha_beta. Per-symbol net bps
    then equal-weight across symbols each cycle -> portfolio series."""
    d = pos_df.sort_values(["symbol", "open_time"]).copy()
    d["dpos"] = d.groupby("symbol")["pos"].diff().abs().fillna(d["pos"].abs())
    d["net"] = d["pos"] * d["alpha_beta"] * 1e4 - d["dpos"] * COST
    port = (d.groupby(["open_time", "fold"])["net"].mean()
            .reset_index().sort_values("open_time"))
    return port, d


def sharpe_ci(net: np.ndarray):
    if len(net) < 10 or np.std(net) < 1e-12:
        return np.nan, np.nan, np.nan, float(np.mean(net) if len(net) else np.nan)
    sh = float(net.mean() / net.std(ddof=1) * ANN)
    lo, hi = block_bootstrap_ci(
        net, statistic=lambda x: x.mean() / x.std(ddof=1) * ANN
        if x.std(ddof=1) > 1e-12 else 0.0, block_size=7, n_boot=1000)[1:]
    return sh, float(lo), float(hi), float(net.mean())


def main():
    print("=" * 96, flush=True)
    print("  STEP 90 (Stage 4): rigorous per-symbol OI eval — the determinant",
          flush=True)
    print("  PRE-REG PASS (signed_nested): G1 CI-excl-0 G2 >placebo-p95 "
          "G3 est-consistent & no in-sample>>nested G4 >=60% syms + drop-top2",
          flush=True)
    print("=" * 96, flush=True)
    t0 = time.time()

    # rebuild per-symbol normalized frame + folds (reuse Step-89 machinery)
    pan = pd.read_parquet(s89.PANEL, columns=["symbol", "open_time",
                          "alpha_beta", "sigma_idio", "autocorr_pctile_7d"])
    pan["open_time"] = pd.to_datetime(pan["open_time"], utc=True)
    oi = pd.read_parquet(s89.OI)
    oi["open_time"] = pd.to_datetime(oi["open_time"], utc=True)
    df = pan.merge(oi, on=["symbol", "open_time"], how="inner").sort_values(
        ["symbol", "open_time"]).reset_index(drop=True)
    df = s89.per_symbol_pit_z(df, s89.FEATS)
    folds = _multi_oos_splits(df)
    df["fold"] = -1
    for fid in range(len(folds)):
        te = _slice(df, folds[fid])[2]
        df.loc[te.index, "fold"] = fid
    P = pd.read_parquet(PRED)
    P["open_time"] = pd.to_datetime(P["open_time"], utc=True)
    syms = sorted(P["symbol"].unique())
    print(f"  {len(syms)} symbols; preds {len(P):,}", flush=True)

    # ---- signed_nested: per symbol, fold k signs from folds < k only ----
    nested_rows = []
    for s in syms:
        d = df[df.symbol == s]
        for k in OOS:
            if k >= len(folds):
                continue
            past = d[d["fold"].between(1, k - 1)] if k > 1 else \
                _slice(d, folds[0])[0]
            past = past.dropna(subset=NF + ["alpha_beta"])
            te = _slice(d, folds[k])[2].dropna(subset=NF).copy()
            if len(past) < 800 or len(te) < 50:
                continue
            sgn = np.array([np.sign(np.corrcoef(past[c], past["alpha_beta"])[0, 1])
                            if past[c].std() > 1e-12 and
                            past["alpha_beta"].std() > 1e-12 else 1.0
                            for c in NF], float)
            sc = te[NF].to_numpy(float) @ sgn
            te = te.assign(score=sc)
            nested_rows.append(te[["symbol", "open_time", "fold",
                                   "alpha_beta", "score"]])
    NST = pd.concat(nested_rows, ignore_index=True)

    def eval_variant(frame, scorecol):
        ds = decision_sample(frame).dropna(subset=[scorecol, "alpha_beta"])
        ds["pos"] = np.sign(ds[scorecol]).astype(float)
        port, perd = portfolio_net(ds[["symbol", "open_time", "fold",
                                       "pos", "alpha_beta"]])
        sh, lo, hi, mu = sharpe_ci(port["net"].to_numpy())
        # per-symbol net + de-concentration
        sym_net = perd.groupby("symbol")["net"].mean()
        pos_frac = float((sym_net > 0).mean())
        top2 = sym_net.abs().sort_values(ascending=False).head(2).index.tolist()
        port_d2, _ = portfolio_net(
            ds[~ds.symbol.isin(top2)][["symbol", "open_time", "fold",
                                       "pos", "alpha_beta"]])
        sh_d2 = sharpe_ci(port_d2["net"].to_numpy())[0]
        fp = sum(1 for _, g in port.groupby("fold") if g["net"].mean() > 0)
        return dict(sharpe=sh, lo=lo, hi=hi, net_bps=mu, sym_pos_frac=pos_frac,
                    drop2_sharpe=sh_d2, folds_pos=fp, top2=",".join(top2),
                    port=port)

    res = {}
    res["ridge_insample"] = eval_variant(P, "pred_ridge")
    res["signed_train"] = eval_variant(P, "pred_signed")
    res["signed_nested"] = eval_variant(NST, "score")

    # ---- placebo: random per-symbol sign, matched ----
    base = decision_sample(NST).dropna(subset=["score", "alpha_beta"])
    pl = []
    for sd in range(N_PLACEBO):
        rng = np.random.default_rng(sd)
        b = base.copy()
        rs = {s: rng.choice([-1.0, 1.0]) for s in b["symbol"].unique()}
        b["pos"] = b["symbol"].map(rs)
        pp, _ = portfolio_net(b[["symbol", "open_time", "fold", "pos",
                                 "alpha_beta"]])
        pl.append(sharpe_ci(pp["net"].to_numpy())[0])
    p95 = float(np.nanpercentile(pl, 95))

    rows = [{"variant": k, **{kk: vv for kk, vv in v.items() if kk != "port"}}
            for k, v in res.items()]
    pd.DataFrame(rows).to_csv(OUTD / "summary.csv", index=False)
    for k, v in res.items():
        print(f"  {k:16s} netSh={v['sharpe']:+.2f} CI[{v['lo']:+.2f},"
              f"{v['hi']:+.2f}] net={v['net_bps']:+.2f}bps/cyc "
              f"sym+={v['sym_pos_frac']*100:.0f}% dropT2 Sh={v['drop2_sharpe']:+.2f} "
              f"f+={v['folds_pos']}/9", flush=True)
    print(f"  placebo(random per-sym sign) p95={p95:+.2f} "
          f"mean={np.nanmean(pl):+.2f} max={np.nanmax(pl):+.2f}", flush=True)

    N = res["signed_nested"]
    T = res["signed_train"]
    R = res["ridge_insample"]
    G1 = bool(N["lo"] > 0)
    G2 = bool(N["sharpe"] > p95)
    G3 = bool(T["sharpe"] > 0 and np.sign(T["sharpe"]) == np.sign(N["sharpe"])
              and (R["sharpe"] - N["sharpe"] < 1.0))
    G4 = bool(N["sym_pos_frac"] >= 0.60 and N["drop2_sharpe"] > 0)
    allp = G1 and G2 and G3 and G4
    print("\n" + "=" * 96, flush=True)
    print(f"  GATES (signed_nested): G1 CI-excl-0={G1} | G2 >placebo-p95={G2}"
          f" | G3 est-consistent&no-collapse={G3} | G4 >=60%syms&dropT2={G4}",
          flush=True)
    if allp:
        v = (f"PASS — nested-honest per-symbol OI signing generalizes: "
             f"netSh {N['sharpe']:+.2f} CI[{N['lo']:+.2f},{N['hi']:+.2f}] "
             f">placebo p95 {p95:+.2f}, {N['sym_pos_frac']*100:.0f}% syms +, "
             f"drop-top2 {N['drop2_sharpe']:+.2f}, no in-sample collapse "
             f"(ridge {R['sharpe']:+.2f} / nested {N['sharpe']:+.2f}). FIRST "
             f"generalizing edge of the investigation — per-symbol+OI "
             f"hypothesis (V3 77% sign-stable) holds under honest eval. Next "
             f"= robustness + forward plan, NO further backtest.")
    else:
        fails = [g for g, ok in [("G1", G1), ("G2", G2), ("G3", G3),
                 ("G4", G4)] if not ok]
        v = (f"FAIL ({','.join(fails)}) — nested-honest per-symbol OI "
             f"signing netSh {N['sharpe']:+.2f} CI[{N['lo']:+.2f},"
             f"{N['hi']:+.2f}], placebo p95 {p95:+.2f}, ridge_insample "
             f"{R['sharpe']:+.2f}. The V2/V3 per-symbol promise did not "
             f"convert to a net-of-cost edge under honest (no-hindsight, "
             f"no-in-sample-fit) evaluation — per-symbol OI is the same "
             f"non-stationary/fit artifact, just per-symbol. Recorded "
             f"honestly. Production LGBM unaffected.")
    print(f"  VERDICT: {v}", flush=True)
    pd.DataFrame([{"all_pass": allp, "placebo_p95": p95, "verdict": v}]).to_csv(
        OUTD / "verdict.csv", index=False)
    print(f"\nSaved {OUTD}\nTotal: {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
