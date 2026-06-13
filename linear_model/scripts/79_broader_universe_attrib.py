"""Step 79: broader-universe payoff + per-symbol attribution diagnostic.

Motivation (from Step 78): NNLS/pos-Ridge give a positive, non-inverting,
fold-robust (7/9) but SUB-COST payoff (~+6-7 bps K3 vs ~9 bps RT) on the
42-sym testbed. More cross-sectional breadth might push it past cost — OR any
widening could be a few illiquid/meme names (the Steps 55-60 trap, where a
Binance-110 "edge" was ~5 meme coins not on HL). This script answers that
WITHOUT a backtest, with the meme-tail detector built in.

Three universes, identical fold dates (derived once from the full panel so OOS
windows are apples-to-apples; sigma/feature standardization rebuilt per
universe = PIT-faithful):
  hl42        : on_hl & hl_vol>=$2M, drop BIO/VVV/BTC   (current baseline)
  hl_all      : on_hl (any vol), drop BTC               (EXECUTABLE, ~70)
  binance110  : all symbols, drop BTC                   (RESEARCH-ONLY)

Carried scores (from Step-78, the informative ones): ridge_xsz (best shape),
nnls_oriented (best positive monotone), signed_equal (monotone-but-flat ref).
signed_shrunk dropped (inverted/dead); posridge dropped (== nnls).

For each (universe, score):
  - Step-77 payoff: decile rho, K1/K3/K10 spread, K3 folds+, IC (context)
  - per-symbol gross attribution of the K=3 L/S book + HL status
  - top-5 share of positive gross; drop-top-2-and-rerank de-concentration

PRE-REGISTERED GATE (fixed before run):
  a score "rescues the line" on a universe iff
    decile rho >= +0.60  AND  K=3 spread >= +9.0 bps  AND  >=6/9 folds K3+
    AND top-5 symbols <= 60% of positive gross
    AND drop-top-2-and-rerank keeps K=3 spread > 0.
  DECISION is keyed off hl_all (executable). binance110 is context only —
  if its payoff is carried by non-HL / top-2 names it is the meme-tail
  artifact, NOT generalization. No backtest here.
"""
from __future__ import annotations

import importlib.util
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import nnls
from sklearn.linear_model import RidgeCV

warnings.filterwarnings("ignore")
REPO = Path("/home/yuqing/ctaNew")
sys.path.insert(0, str(REPO))


def _imp(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, REPO / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


s58 = _imp("s58", "linear_model/scripts/58_clean108_train.py")
s76 = _imp("s76", "linear_model/scripts/76_minimal_orientation.py")
s77 = _imp("s77", "linear_model/scripts/77_orientation_decile_diag.py")
s78 = _imp("s78", "linear_model/scripts/78_nnls_poscoef_payoff.py")
from ml.research.alpha_v4_xs_1d import _multi_oos_splits, _slice

OUT = REPO / "linear_model/results/step79_broader_universe"
OUT.mkdir(parents=True, exist_ok=True)
PANEL_111 = REPO / "outputs/vBTC_features_btc_only_111_full_pit/panel_btc_only_111.parquet"
HL_MAP = REPO / "outputs/vBTC_check_universe/panel110_hl_map.csv"
OOS = s76.OOS
BLOCK = s76.BLOCK
ALPHAS = s58.ALPHAS
SCORES = ["ridge_xsz", "nnls_oriented", "signed_equal"]
GATE_RHO, GATE_SPREAD, GATE_FOLDS, GATE_TOP5 = 0.60, 9.0, 6, 0.60


def universe_keep(hl: pd.DataFrame, all_syms: set, name: str):
    on_hl = set(hl[hl.on_hl]["symbol"])
    if name == "hl42":
        keep = set(hl[(hl.on_hl) & (hl.hl_day_vol_usd >= 2e6)]["symbol"])
        drop = {"BIOUSDT", "VVVUSDT", "BTCUSDT"}
    elif name == "hl_all":
        keep = on_hl
        drop = {"BTCUSDT"}
    else:  # binance110
        keep = set(all_syms)
        drop = {"BTCUSDT"}
    return keep, drop, on_hl


def build_universe(panel_raw: pd.DataFrame, hl: pd.DataFrame,
                   folds: list, name: str):
    keep, drop, on_hl = universe_keep(hl, set(panel_raw["symbol"].unique()), name)
    p = panel_raw[panel_raw["symbol"].isin(keep)
                  & ~panel_raw["symbol"].isin(drop)].copy()
    f0idx = _slice(p, folds[0])[0].index
    sg = p.loc[f0idx].groupby("symbol")["alpha_beta"].std()
    med = float(sg.dropna().median())
    p["sigma_idio"] = p["symbol"].map(sg).fillna(med).clip(lower=1e-6)
    p = s58.build_target_z(p, f0idx)
    tr0 = _slice(p, folds[0])[0]
    tm = p["open_time"].between(tr0["open_time"].min(), tr0["open_time"].max())
    X, fc = s58.build_v2_features(p, tm)
    px = p[["symbol", "open_time", "alpha_beta", "target_z",
            "autocorr_pctile_7d"]].merge(
        X.drop(columns=["alpha_beta", "target_z", "autocorr_pctile_7d"]),
        on=["symbol", "open_time"], how="left")
    px["open_time"] = pd.to_datetime(px["open_time"], utc=True)
    grid = sorted(px["open_time"].unique())[::BLOCK]
    dec = px[px["open_time"].isin(set(grid))].copy()
    dec = s76.assign_folds(dec, folds)
    return dec, fc, on_hl


def score_universe(dec: pd.DataFrame, fc: list, folds: list,
                   model: str) -> pd.DataFrame:
    rows = []
    for k in OOS:
        if k >= len(folds):
            continue
        tr = _slice(dec, folds[k])[0].dropna(subset=["alpha_beta"])
        if len(tr) < 500:
            continue
        w76 = s76.fit_weights(tr, fc, "alpha_beta")
        sign = np.array([np.sign(w76[f]) or 1.0 for f in fc], float)
        coef = None
        if model in ("ridge_xsz", "nnls_oriented"):
            Xtr, ytr = [], []
            for _, g in tr.dropna(subset=["target_z"]).groupby("open_time"):
                if len(g) < 5:
                    continue
                Z = s78._xsz(g, fc)
                yv = g["target_z"].to_numpy(float)
                if not np.isfinite(yv).all() or np.std(yv) <= 1e-12:
                    continue
                Xtr.append(Z if model == "ridge_xsz" else Z * sign)
                ytr.append(yv)
            if not Xtr:
                continue
            Xtr, ytr = np.vstack(Xtr), np.concatenate(ytr)
            if model == "ridge_xsz":
                coef = RidgeCV(alphas=ALPHAS, scoring="r2",
                               fit_intercept=False).fit(Xtr, ytr).coef_
            else:
                coef, _ = nnls(Xtr, ytr)
        te = dec[dec["fold"] == k].dropna(subset=["alpha_beta"]).copy()
        for t, g in te.groupby("open_time", sort=True):
            if len(g) < 5:
                continue
            Z = s78._xsz(g, fc)
            yv = g["alpha_beta"].to_numpy(float)
            if np.std(yv) <= 1e-12:
                continue
            if model == "signed_equal":
                sc = (Z * sign) @ np.ones(len(fc))
            elif model == "ridge_xsz":
                sc = Z @ coef
            else:
                sc = (Z * sign) @ coef
            if np.std(sc) <= 1e-12:
                continue
            rows.append(pd.DataFrame({"open_time": t, "fold": k,
                                      "symbol": g["symbol"].to_numpy(),
                                      "score": sc, "y": yv}))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def payoff(df: pd.DataFrame) -> dict:
    ic = df.groupby("open_time").apply(
        lambda g: g["score"].corr(g["y"], method="spearman")).dropna()
    dec = s77._binned(df, 10)
    rho = s77._mono(dec)
    ks = s77._ksweep(df, ks=(1, 3, 10))
    k3 = float(ks[ks["K"] == 3].iloc[0]["spread_bps"])
    fp = 0
    for _, g in df.groupby("fold"):
        sp = [s.head(3)["y"].mean() - s.tail(3)["y"].mean()
              for _, gg in g.groupby("open_time") if len(gg) >= 6
              for s in [gg.sort_values("score", ascending=False)]]
        if sp and np.mean(sp) > 0:
            fp += 1
    return dict(ic=float(ic.mean()), decile_rho=float(rho),
                k1=float(ks[ks["K"] == 1].iloc[0]["spread_bps"]), k3=k3,
                k10=float(ks[ks["K"] == 10].iloc[0]["spread_bps"]),
                k3_folds_pos=fp)


def attribution(df: pd.DataFrame, on_hl: set) -> tuple[pd.DataFrame, dict]:
    """Per-symbol gross of the K=3 L/S book (+1/3 top3, -1/3 bottom3)."""
    contrib = {}
    for _, g in df.groupby("open_time"):
        if len(g) < 6:
            continue
        gs = g.sort_values("score", ascending=False)
        for s, yy in zip(gs.head(3)["symbol"], gs.head(3)["y"]):
            contrib[s] = contrib.get(s, 0.0) + (1 / 3) * yy * 1e4
        for s, yy in zip(gs.tail(3)["symbol"], gs.tail(3)["y"]):
            contrib[s] = contrib.get(s, 0.0) + (-1 / 3) * yy * 1e4
    a = (pd.DataFrame({"symbol": list(contrib), "gross_bps": list(contrib.values())})
         .sort_values("gross_bps", ascending=False).reset_index(drop=True))
    a["on_hl"] = a["symbol"].isin(on_hl)
    pos_tot = a.loc[a["gross_bps"] > 0, "gross_bps"].sum()
    top5 = a.head(5)["gross_bps"].sum()
    top5_share = float(top5 / pos_tot) if pos_tot > 0 else np.nan
    top2 = list(a.head(2)["symbol"])
    top5_on_hl = bool(a.head(5)["on_hl"].all())
    return a, dict(top5_share=top5_share, top2=top2,
                   top5_all_hl=top5_on_hl,
                   total_gross=float(a["gross_bps"].sum()))


def main():
    print("=" * 100, flush=True)
    print("  STEP 79: broader-universe payoff + per-symbol attribution", flush=True)
    print(f"  GATE: rho>=+{GATE_RHO} & K3>=+{GATE_SPREAD}bps & >={GATE_FOLDS}/9 "
          f"& top5<= {GATE_TOP5:.0%} pos-gross & drop-top2 K3>0 | "
          f"decision keyed off hl_all", flush=True)
    print("=" * 100, flush=True)
    t0 = time.time()
    panel_raw = pd.read_parquet(PANEL_111)
    panel_raw["open_time"] = pd.to_datetime(panel_raw["open_time"], utc=True)
    hl = pd.read_csv(HL_MAP)
    full = panel_raw[panel_raw["symbol"] != "BTCUSDT"]
    folds = _multi_oos_splits(full)                     # shared fold dates
    print(f"  panel {len(panel_raw):,} rows, {panel_raw['symbol'].nunique()} "
          f"syms; {len(folds)} folds (shared)", flush=True)

    rows, attr_rows = [], []
    for uni in ["hl42", "hl_all", "binance110"]:
        dec, fc, on_hl = build_universe(panel_raw, hl, folds, uni)
        n_sym = dec["symbol"].nunique()
        executable = uni != "binance110"
        print(f"\n--- {uni}  ({n_sym} syms, "
              f"{'EXECUTABLE' if executable else 'RESEARCH-ONLY'}) ---",
              flush=True)
        for mdl in SCORES:
            df = score_universe(dec, fc, folds, mdl)
            if df.empty:
                print(f"  {mdl:14s} no scores", flush=True)
                continue
            p = payoff(df)
            a, ainfo = attribution(df, on_hl)
            a.to_csv(OUT / f"{uni}_{mdl}_attrib.csv", index=False)
            # drop-top-2-and-rerank de-concentration
            df2 = df[~df["symbol"].isin(ainfo["top2"])]
            p2 = payoff(df2) if not df2.empty else {"k3": np.nan,
                                                    "decile_rho": np.nan}
            gate = (p["decile_rho"] >= GATE_RHO and p["k3"] >= GATE_SPREAD
                    and p["k3_folds_pos"] >= GATE_FOLDS
                    and ainfo["top5_share"] <= GATE_TOP5
                    and p2["k3"] > 0)
            rows.append({"universe": uni, "executable": executable,
                         "n_sym": n_sym, "model": mdl, **p,
                         "top5_share": ainfo["top5_share"],
                         "top2": ",".join(ainfo["top2"]),
                         "top5_all_hl": ainfo["top5_all_hl"],
                         "k3_drop_top2": p2["k3"],
                         "rho_drop_top2": p2["decile_rho"],
                         "gate_pass": bool(gate)})
            print(f"  {mdl:14s} IC={p['ic']:+.4f} rho={p['decile_rho']:+.3f} "
                  f"K1={p['k1']:+6.2f} K3={p['k3']:+6.2f} K10={p['k10']:+6.2f} "
                  f"f+={p['k3_folds_pos']}/9 | top5={ainfo['top5_share']*100:4.0f}%"
                  f"({'HL' if ainfo['top5_all_hl'] else 'NON-HL'}) "
                  f"top2={ainfo['top2']} | dropT2 K3={p2['k3']:+6.2f} | "
                  f"{'PASS' if gate else 'FAIL'}", flush=True)
            head = a.head(5).to_dict("records")
            print("      top5 gross: " + " ".join(
                f"{r['symbol']}{'*' if not r['on_hl'] else ''}="
                f"{r['gross_bps']:+.0f}" for r in head), flush=True)

    out = pd.DataFrame(rows)
    out.to_csv(OUT / "summary.csv", index=False)
    ex = out[(out["universe"] == "hl_all")]
    rescued = bool(ex["gate_pass"].any())
    b110 = out[out["universe"] == "binance110"]
    meme = bool((b110["gate_pass"] | (b110["k3"] >= GATE_SPREAD)).any()
                and not (b110["top5_all_hl"].any()))
    print("\n" + "=" * 100, flush=True)
    print("  VERDICT", flush=True)
    print("=" * 100, flush=True)
    if rescued:
        v = ("EXECUTABLE hl_all clears the full gate (payoff + de-concentrated) "
             "-> broader universe rescues the linear line; next = robustness/"
             "audit of that (universe,score), still NO backtest.")
    elif meme:
        v = ("Binance-110 shows payoff but it is concentrated in non-HL / "
             "top-2 names = the Steps 55-60 MEME-TAIL ARTIFACT, NOT "
             "generalization. hl_all (executable) does NOT clear. Linear line "
             "stays closed; pivot to Step 80 Batch B.")
    else:
        v = ("No universe (incl. executable hl_all) clears the gate; breadth "
             "does not push the +6-7 bps constrained signal past cost and "
             "no meme-tail rescue either. Linear-on-current-features stays "
             "CLOSED; pivot to Step 80 Batch B (price x volume interactions).")
    print(f"  {v}", flush=True)
    pd.DataFrame([{"hl_all_rescued": rescued, "binance110_memetail": meme,
                   "verdict": v}]).to_csv(OUT / "verdict.csv", index=False)
    print(f"\nSaved under {OUT}\nTotal: {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
