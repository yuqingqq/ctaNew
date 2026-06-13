"""Step 103 — FORENSICS (not a gated test): inspect the Step-102
LONG-consensus fired trades. Examples + honest descriptive breakdowns to
answer "show me the trades / is there a trackable pattern / what metrics".

Composite block REPLICATED VERBATIM from Step 102 (locked, deterministic)
so the forensic view == the tested signal exactly. Target = panel
alpha_beta (fwd-4h β-resid). DESCRIPTIVE ONLY — any subset that looks good
here is in-sample; per the arc (Step 88 etc.) such subsets fail
nested-OOS+placebo. Production LGBM unaffected.
"""
from __future__ import annotations
import importlib.util, sys, warnings
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
pd.set_option("display.width", 200)
REPO = Path("/home/yuqing/ctaNew")
sys.path.insert(0, str(REPO))


def _imp(n, r):
    s = importlib.util.spec_from_file_location(n, REPO / r)
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m


s94 = _imp("s94", "linear_model/scripts/94_info_ceiling_d1.py")


def main():
    print("=" * 100, flush=True)
    print("  STEP 103 — LONG-consensus FORENSICS (descriptive only)", flush=True)
    print("=" * 100, flush=True)
    dec, syms, btc, pan = s94.build(universe_oi=False)
    oi = pd.read_parquet(REPO / "outputs/vBTC_features_oi/oi_panel.parquet")
    of = pd.read_parquet(REPO / "outputs/vBTC_features_oflow/oflow_panel.parquet")
    for f in (oi, of):
        f["open_time"] = pd.to_datetime(f["open_time"], utc=True)
    d = (dec[dec.symbol.isin(set(oi.symbol) & set(of.symbol))]
         .merge(oi, on=["symbol", "open_time"], how="inner")
         .merge(of, on=["symbol", "open_time"], how="inner"))
    need = ["return_1d", "return_8h", "s_t", "funding_rate_z_7d", "oi_chg_1d",
            "oi_chg_4h", "oi_z_7d", "ls_taker_z_1d", "of_imb_1d",
            "vol_zscore_4h_over_7d", "obv_z_1d", "idio_vol_to_btc_1d",
            "corr_to_btc_1d", "alpha_beta", "fold"]
    d = d.dropna(subset=need).reset_index(drop=True)
    sgn = lambda x: np.sign(x).astype(float)
    r, r8, st = d["return_1d"], d["return_8h"], d["s_t"]
    oic, oiz = d["oi_chg_1d"], d["oi_z_7d"]
    fz, ls, ofi = d["funding_rate_z_7d"], d["ls_taker_z_1d"], d["of_imb_1d"]
    vz, obv, idv, cb = (d["vol_zscore_4h_over_7d"], d["obv_z_1d"],
                        d["idio_vol_to_btc_1d"], d["corr_to_btc_1d"])
    Z = 1.0
    C = {}
    C["O1"] = np.where((r > 0) & (oic > 0) & (fz > 0) & (ls > 0), -1.0, 0)
    C["O2"] = np.where((r < 0) & (oic > 0) & (fz < 0) & (ls < 0), 1.0, 0)
    C["O3"] = np.where((r < 0) & (oic < 0) & (vz > Z), 1.0, 0)
    C["O4"] = np.where((r > 0) & (oic < 0), -1.0, 0)
    C["O5"] = np.where(fz.abs() > Z, -sgn(fz), 0)
    C["P1"] = np.where(st.abs() > Z, -sgn(st), 0)
    C["P2"] = np.where((sgn(r) == sgn(r8)) & (r != 0), -sgn(r), 0)
    C["V1"] = np.where((vz > Z) & (r != 0), -sgn(r), 0)
    C["V2"] = np.where((vz < -0.5) & (r != 0), -sgn(r), 0)
    C["V3"] = np.where(sgn(obv) != sgn(r), sgn(obv), 0)
    C["V4"] = np.where((sgn(ofi) != sgn(st)) & (st.abs() > 0.5), -sgn(st), 0)
    C["R1"] = np.where((idv > Z) & (st.abs() > Z), -sgn(st), 0)
    C["R2"] = np.where((vz > Z) & (r != 0), sgn(r), 0)
    C["X1"] = np.where((cb < 0) & (st.abs() > Z), -sgn(st), 0)
    M = pd.DataFrame(C, index=d.index)
    d["V"] = M.sum(axis=1)
    d["ab"] = d["alpha_beta"] * 1e4
    L = d[d["V"] > 0].copy()
    Ml = M.loc[L.index]
    L["long_set"] = [",".join(sorted(Ml.columns[(Ml.loc[i] > 0).values]))
                     for i in L.index]
    L["win"] = (L["ab"] > 0)
    print(f"\n  LONG-consensus n={len(L)} ({len(L)/len(d)*100:.0f}% of "
          f"{len(d)})  mean αβ={L.ab.mean():+.2f}bps  hit={L.win.mean()*100:.1f}%",
          flush=True)

    print("\n  --- 15 example fired LONG trades (evenly sampled) ---",
          flush=True)
    ex = L.iloc[np.linspace(0, len(L)-1, 15).astype(int)]
    for _, x in ex.iterrows():
        print(f"   {x.symbol:10s} {str(x.open_time)[:16]}  V={int(x.V):+d}  "
              f"αβ={x.ab:+7.1f}bps  {'WIN ' if x.win else 'LOSS'}  "
              f"[{x.long_set}]", flush=True)

    print("\n  --- outcome distribution (is the +mean direction or tail?) ---",
          flush=True)
    a = L["ab"].to_numpy()
    tot = a.sum()
    top5 = np.sort(a)[-int(len(a)*0.05):].sum()
    print(f"   mean={a.mean():+.2f} median={np.median(a):+.2f} "
          f"std={a.std():.1f} skew={pd.Series(a).skew():+.2f}", flush=True)
    print(f"   hit={ (a>0).mean()*100:.1f}%  →  +mean despite ~50% hit is "
          f"TAIL-driven: top-5% trades = {top5/tot*100:.0f}% of total PnL "
          f"(direction is ~coin-flip; mild positive skew, not predictability)",
          flush=True)

    print("\n  --- per-composite WITHIN long-consensus (in-sample; needs "
          "nested+placebo to mean anything) ---", flush=True)
    rows = []
    for c in M.columns:
        m = (Ml[c] > 0).values
        if m.sum() < 50:
            continue
        rows.append((c, m.sum(), L["ab"].values[m].mean(),
                     (L["ab"].values[m] > 0).mean()*100))
    for c, n, mn, h in sorted(rows, key=lambda z: -z[2]):
        print(f"   {c}: n={n:5d}  mean αβ={mn:+6.2f}bps  hit={h:4.1f}%",
              flush=True)

    print("\n  --- per-symbol (long-consensus) ---", flush=True)
    g = L.groupby("symbol").agg(n=("ab", "size"), mean=("ab", "mean"),
                                hit=("win", "mean")).sort_values("mean")
    print(g.assign(hit=(g.hit*100).round(1)).round(2).to_string(), flush=True)

    print("\n  --- per-fold (long-consensus) ---", flush=True)
    gf = L.groupby("fold").agg(n=("ab", "size"), mean=("ab", "mean"),
                               hit=("win", "mean"))
    print(gf.assign(hit=(gf.hit*100).round(1)).round(2).to_string(),
          flush=True)

    # PIT regime: BTC trailing 1d trend sign (strictly past)
    bt = btc.pct_change(288).shift(1)               # btc = close Series (s94)
    L = L.merge(bt.rename("btc_tr").reset_index(), on="open_time", how="left")
    for nm, msk in [("BTC trail UP", L.btc_tr > 0),
                    ("BTC trail DOWN", L.btc_tr <= 0)]:
        s = L[msk]
        print(f"\n  regime {nm}: n={len(s)} mean αβ={s.ab.mean():+.2f}bps "
              f"hit={s.win.mean()*100:.1f}%", flush=True)

    print("\n  SUMMARY: the only 'pattern' in LONG-consensus trades is a mild "
          "positive SKEW (≈coin-flip direction, +mean carried by a few tail "
          "winners), not directional predictability; per-composite / "
          "per-symbol / per-fold / per-regime slices are noisy and "
          "in-sample. Any cell that looks good would have to clear the "
          "pre-registered nested-OOS + matched-placebo gate — the arc (Step "
          "88: −1.05; Step 90: −2.60 below random) shows such hindsight "
          "subsets do not. Forensic, not a strategy basis.", flush=True)


if __name__ == "__main__":
    main()
