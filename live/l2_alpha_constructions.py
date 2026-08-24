"""GATED L2-alpha search on the STRATEGY'S REAL pipeline (x6 preproc + V0_LEAN(14) + per-symbol RidgeCV + HL=60
exp-decay + exit_time purge + 1d embargo), BOTH eras. Prior is strongly NULL. Baseline MUST reproduce
~+0.0301 recent / ~+0.017 oos or the harness is invalid.

Already ruled out this session: raw/sustained imbalance, dynamics, persistence, liquidity add NOTHING (imb_ewma
Delta rank-IC -0.0020 CI<0 recent). L2 imbalance is CONTINUATION not reversion. NEW angles tested here:

  C1  L2 x regime/dispersion interaction  (imb1 informative only when xs-dispersion / btc-vol high?)
  C2  short-leg selection: does an L2 add re-rank the SHORT candidates (bottom-tercile pred) better?
  C3  book-SHAPE not yet tried: l2_slope (both-era), l2_imb02 / l2_touch / microprice tilt (RECENT-only)
  C4  L2 fragility (asym1/slope/imbstd) as a SHORT-side risk/return feature specifically

ADOPT only if baseline validated AND Delta (rank-IC or short selection-spread) CI-off-zero and POSITIVE in BOTH eras.
Every L2 feature is the 4h book aggregate over [T,T+4h) shifted +4h -> known at decision bar T+4h (PIT).
"""
import os, sys, glob
from pathlib import Path
import numpy as np, pandas as pd
import warnings; warnings.filterwarnings("ignore")
REPO = Path("/home/yuqing/ctaNew"); sys.path.insert(0, str(REPO))
os.environ["V4_PANEL"] = str(REPO / "outputs/vBTC_features/panel_expanded_v0_clean.parquet")
import live.train_twobook_models as tt
from sklearn.linear_model import RidgeCV
from scipy.stats import spearmanr
from live.bookdepth_persist import persist_feats

x6 = tt.x6; V0_LEAN = list(tt.V0_LEAN); EMB = pd.Timedelta(days=1); HL = 60.0
rng = np.random.default_rng(7)
RECENT_CUTS = [pd.Timestamp(t, tz="UTC") for t in ["2025-10-04", "2025-11-01", "2025-12-01", "2026-01-01",
              "2026-02-01", "2026-03-01", "2026-04-01", "2026-05-01", "2026-06-01", "2026-06-30"]]
OOS_CUTS = [pd.Timestamp(t, tz="UTC") for t in ["2024-01-15", "2024-02-15", "2024-03-15", "2024-04-15",
            "2024-05-15", "2024-06-30"]]
RAW_L2 = ["l2_imb1", "l2_imb02", "l2_liq1", "l2_touch", "l2_slope", "l2_asym1", "l2_imbstd"]


def build_panel():
    # NB: return_1d and btc_rvol_7d are already in V0_LEAN -> do not list them twice (dup cols break groupby)
    PAN = pd.read_parquet(tt.PANEL, columns=["symbol", "open_time", "exit_time", "alpha_vs_btc_realized"] + V0_LEAN)
    PAN["open_time"] = pd.to_datetime(PAN["open_time"], utc=True)
    PAN["exit_time"] = pd.to_datetime(PAN["exit_time"], utc=True)
    PAN = PAN[(PAN.open_time.dt.hour % 4 == 0) & (PAN.open_time.dt.minute == 0)]
    rows = []
    for f in [x for x in glob.glob(str(REPO / "data/ml/cache/l2_*.parquet")) if "BTCUSDT" not in x]:
        sym = Path(f).stem[3:]
        d0 = pd.read_parquet(f)[RAW_L2].copy()
        d0.index = pd.to_datetime(d0.index, utc=True) + pd.Timedelta("4h")  # +4h -> PIT decision bar
        d0 = d0.sort_index()
        pf = persist_feats(d0["l2_imb1"])[["imb_ewma"]]            # sustained imbalance (already-null control)
        d0 = d0.join(pf)
        d0["symbol"] = sym; d0["open_time"] = d0.index
        rows.append(d0.reset_index(drop=True))
    L = pd.concat(rows, ignore_index=True)
    PAN = PAN.merge(L, on=["symbol", "open_time"], how="left")

    # ---- PIT conditioners (all trailing / decision-bar) ----
    g = PAN.groupby("open_time")
    PAN["xs_disp"] = g["return_1d"].transform("std")              # cross-sectional dispersion of trailing 1d ret
    # interactions (main effect l2_imb1 kept alongside so Ridge can center the interaction)
    PAN["imb1_x_disp"] = PAN["l2_imb1"] * PAN["xs_disp"]
    PAN["imb1_x_btcrvol"] = PAN["l2_imb1"] * PAN["btc_rvol_7d"]
    # microprice-vs-mid tilt proxy: near-touch imbalance minus 1% imbalance (is the lean concentrated at touch?)
    PAN["micro_tilt"] = PAN["l2_imb02"] - PAN["l2_imb1"]          # RECENT-only (needs imb02)

    # coverage masks
    PAN["cov_full"] = PAN["l2_imb1"].notna()                      # full-era L2 coverage
    PAN["cov_rec"] = PAN["l2_imb02"].notna()                      # RECENT-only features coverage

    # neutral-fill features so old/uncovered bars are 0 (and ~0 weight under HL=60)
    fill = ["imb_ewma"] + RAW_L2 + ["imb1_x_disp", "imb1_x_btcrvol", "micro_tilt", "xs_disp"]
    for c in fill:
        PAN[c] = PAN[c].fillna(0.0)

    # target = cross-sectional z of realized alpha
    sd = g["alpha_vs_btc_realized"].transform("std").replace(0, np.nan)
    PAN["z_res"] = ((PAN["alpha_vs_btc_realized"] - g["alpha_vs_btc_realized"].transform("mean")) / sd).clip(-10, 10)
    return PAN.sort_values(["symbol", "open_time"]).reset_index(drop=True)


def gen(PAN, feats, cuts):
    """Real per-symbol RidgeCV pipeline. Returns per-row (open_time, symbol, alpha_A, pred, cov_full, cov_rec)."""
    rec = []
    for i in range(len(cuts) - 1):
        c0, c1 = cuts[i], cuts[i + 1]; fc = c0 - EMB
        tr = PAN[(PAN.exit_time < fc) & PAN["z_res"].notna()]
        te = PAN[(PAN.open_time >= c0) & (PAN.open_time < c1)]
        t_end = tr["open_time"].max()
        for sym, gg in tr.groupby("symbol"):
            if len(gg) < 300: continue
            try:
                s, h = x6.fit_preproc(gg, feats); X = x6.apply_preproc(gg, feats, s, h)
                w = np.exp(-((t_end - gg["open_time"]).dt.total_seconds().to_numpy() / 86400.0) / HL)
                m = RidgeCV(alphas=x6.RIDGE_ALPHAS).fit(X, gg["z_res"].to_numpy(), sample_weight=w)
                gte = te[te.symbol == sym]
                if len(gte):
                    rec.append(pd.DataFrame({"open_time": gte["open_time"].values, "symbol": sym,
                                             "alpha_A": gte["alpha_vs_btc_realized"].values,
                                             "pred": m.predict(x6.apply_preproc(gte, feats, s, h)),
                                             "cov_full": gte["cov_full"].values, "cov_rec": gte["cov_rec"].values}))
            except Exception:
                pass
    return pd.concat(rec, ignore_index=True)


def perbar_ic(P, mask):
    P = P[P[mask]]
    return P.groupby("open_time").apply(
        lambda g: spearmanr(g["pred"], g["alpha_A"]).correlation if len(g) >= 5 else np.nan).dropna()


def day_boot_delta(d_series, n=3000):
    """day-clustered bootstrap 95% CI of the mean of a per-bar delta series (index=open_time)."""
    j = pd.DataFrame({"d": d_series.values}, index=pd.to_datetime(d_series.index, utc=True))
    j["day"] = j.index.floor("1D")
    gg = [x["d"].values for _, x in j.groupby("day")]
    if len(gg) < 4: return (np.nan, np.nan)
    boot = [np.concatenate([gg[k] for k in rng.integers(0, len(gg), len(gg))]).mean() for _ in range(n)]
    return tuple(np.percentile(boot, [2.5, 97.5]))


def flag(lo, up):
    return "ADDS (CI>0)" if lo > 0 else ("HURTS (CI<0)" if up < 0 else "within noise")


def full_panel_delta(base, var, mask):
    ib, iv = perbar_ic(base, mask), perbar_ic(var, mask)
    j = pd.concat([ib.rename("a"), iv.rename("b")], axis=1).dropna()
    d = j["b"] - j["a"]
    lo, up = day_boot_delta(d)
    return ib.mean(), iv.mean(), d.mean(), lo, up


def short_pool_metrics(base, var, mask, terc=1 / 3.0, kshort=2):
    """Align base/var pred per (bar,symbol) on covered names. Pool = bottom-tercile by BASE pred (incumbent shorts).
    (a) Delta short-pool rank-IC: within pool, rankIC(var_pred,alpha) - rankIC(base_pred,alpha).
    (b) Delta short-leg realized alpha: mean alpha of k most-short by var vs by base (negative Delta = better shorts).
    """
    b = base[base[mask]][["open_time", "symbol", "alpha_A", "pred"]].rename(columns={"pred": "pb"})
    v = var[var[mask]][["open_time", "symbol", "pred"]].rename(columns={"pred": "pv"})
    m = b.merge(v, on=["open_time", "symbol"], how="inner")
    dic, dsp = [], []
    idx = []
    for t, gbar in m.groupby("open_time"):
        n = len(gbar)
        if n < 9: continue
        k = max(3, int(np.ceil(n * terc)))
        pool = gbar.nsmallest(k, "pb")               # incumbent's short candidates
        if pool["alpha_A"].nunique() < 3: continue
        ic_b = spearmanr(pool["pb"], pool["alpha_A"]).correlation
        ic_v = spearmanr(pool["pv"], pool["alpha_A"]).correlation
        if np.isnan(ic_b) or np.isnan(ic_v): continue
        # short-leg realized alpha: k most-short by each model's pred over the FULL covered bar
        sb = gbar.nsmallest(kshort, "pb")["alpha_A"].mean()
        sv = gbar.nsmallest(kshort, "pv")["alpha_A"].mean()
        dic.append(ic_v - ic_b); dsp.append(sv - sb); idx.append(t)
    dic = pd.Series(dic, index=idx); dsp = pd.Series(dsp, index=idx)
    lo1, up1 = day_boot_delta(dic); lo2, up2 = day_boot_delta(dsp)
    return dic.mean(), lo1, up1, dsp.mean(), lo2, up2


# variant registry: (label, feature-add, coverage-mask, run-short-pool?)
VARIANTS = [
    ("C1 disp-inter  [imb1,imb1xdisp]", ["l2_imb1", "imb1_x_disp"], "cov_full", False),
    ("C1 regime-inter[imb1,imb1xrvol]", ["l2_imb1", "imb1_x_btcrvol"], "cov_full", False),
    ("C3 shape       [l2_slope]",       ["l2_slope"], "cov_full", False),
    ("C4 fragility   [asym,slope,istd]", ["l2_asym1", "l2_slope", "l2_imbstd"], "cov_full", True),
    ("C2 imb_ewma add(short-pool)",     ["imb_ewma"], "cov_full", True),
    ("C3 recent      [imb02,touch,tilt]", ["l2_imb02", "l2_touch", "micro_tilt"], "cov_rec", True),
]
GATE = {"RECENT": "+0.030", "OOS": "+0.017-0.024"}


def main():
    PAN = build_panel()
    print(f"panel rows {len(PAN)} | cov_full {int(PAN.cov_full.sum())} | cov_rec {int(PAN.cov_rec.sum())} "
          f"| V0_LEAN={len(V0_LEAN)}\n")
    for era, cuts in [("RECENT", RECENT_CUTS), ("OOS", OOS_CUTS)]:
        print(f"################  {era}  (gate baseline {GATE[era]})  ################")
        base = gen(PAN, V0_LEAN, cuts)
        for lab, add, mask, do_short in VARIANTS:
            if era == "OOS" and mask == "cov_rec":
                print(f"  {lab:34s} : RECENT-only feature -> SKIP OOS (auto-fails both-era gate)\n")
                continue
            var = gen(PAN, V0_LEAN + add, cuts)
            b_ic, v_ic, d, lo, up = full_panel_delta(base, var, mask)
            print(f"  {lab}")
            print(f"    baseline rank-IC {b_ic:+.4f}  (gate {GATE[era]})   variant {v_ic:+.4f}")
            print(f"    Delta full-panel rank-IC {d:+.4f} [{lo:+.4f},{up:+.4f}] -> {flag(lo, up)}")
            if do_short:
                dic, l1, u1, dsp, l2, u2 = short_pool_metrics(base, var, mask)
                print(f"    Delta short-pool rank-IC {dic:+.4f} [{l1:+.4f},{u1:+.4f}] -> {flag(l1, u1)}")
                print(f"    Delta short-leg alpha    {dsp:+.5f} [{l2:+.5f},{u2:+.5f}] "
                      f"-> {'BETTER shorts (CI<0)' if u2 < 0 else ('WORSE (CI>0)' if l2 > 0 else 'within noise')}")
            print()
        print()
    print("L2ALPHACONSTRUCTIONS_DONE")


if __name__ == "__main__":
    main()
