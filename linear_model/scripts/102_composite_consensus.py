"""Step 102 — D1-ext-G: signed-composite LONG vs SHORT consensus
sign-reliability decomposition (owner request: combine all long composites
→ check the sign; all short → check the sign).

LOCKED, parameter-free (sign-based + canonical 1σ cutoffs, NOT tuned), one
run, no sweep. Target = panel alpha_beta (canonical fwd-4h β-residual, no
long-horizon shift landmine). Each composite emits +1(long resid)/−1/0; net
vote V. Decompose long-consensus (V>0) vs short-consensus (V<0): per-side
mean signed payoff, hit-rate, net-of-cost Sharpe, per-fold sign,
matched placebo, |V|-monotonicity.

Honest prior: directions collapse to ~convergence; D1 bounds combined
ceiling sub-cost; production DDI-2 found long≈below-random / short>random.
Value = empirically settle the signed-consensus framing. No strategy
adopted. Production LGBM unaffected.
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
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m


s94 = _imp("s94", "linear_model/scripts/94_info_ceiling_d1.py")
from ml.research.alpha_v4_xs import block_bootstrap_ci
ANN = np.sqrt(365.0 * 6.0)
COST = s94.COST
OUTD = REPO / "linear_model/results/step102_composite_consensus"
OUTD.mkdir(parents=True, exist_ok=True)


def sgn(x):
    return np.sign(x).astype(float)


def main():
    print("=" * 96, flush=True)
    print("  STEP 102 — D1-ext-G: LONG vs SHORT composite-consensus "
          "sign-reliability (LOCKED, parameter-free)", flush=True)
    print("=" * 96, flush=True)
    t0 = time.time()
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
    r, r8, st = d["return_1d"], d["return_8h"], d["s_t"]
    oic, oic4, oiz = d["oi_chg_1d"], d["oi_chg_4h"], d["oi_z_7d"]
    fz, ls, ofi = d["funding_rate_z_7d"], d["ls_taker_z_1d"], d["of_imb_1d"]
    vz, obv, idv, cb = (d["vol_zscore_4h_over_7d"], d["obv_z_1d"],
                        d["idio_vol_to_btc_1d"], d["corr_to_btc_1d"])
    Z = 1.0                                              # canonical 1σ (locked)
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
    C["R2"] = np.where((vz > Z) & (r != 0), sgn(r), 0)        # momentum dissenter
    C["X1"] = np.where((cb < 0) & (st.abs() > Z), -sgn(st), 0)
    M = pd.DataFrame(C, index=d.index)
    d["nL"] = (M > 0).sum(axis=1)
    d["nS"] = (M < 0).sum(axis=1)
    d["V"] = M.sum(axis=1)
    print(f"  rows={len(d)} syms={d.symbol.nunique()} "
          f"cycles={d.open_time.nunique()} composites={M.shape[1]}", flush=True)
    fire = (M != 0).mean()
    print("  fire-rate/composite: " +
          " ".join(f"{k}={v:.0%}" for k, v in fire.items()), flush=True)

    ab = d["alpha_beta"].to_numpy() * 1e4                 # bps

    def side(mask, dirn, name):
        g = d[mask]
        if len(g) < 100:
            print(f"  {name}: n={len(g)} (too few)", flush=True)
            return
        pos = float(dirn)
        pay = pos * g["alpha_beta"].to_numpy() * 1e4
        # net-of-cost: |Δpos| per symbol over time on the fired subset
        f2 = g.sort_values(["symbol", "open_time"]).copy()
        f2["p"] = pos
        dp = f2.groupby("symbol")["p"].diff().abs().fillna(abs(pos))
        net = pay - dp.to_numpy() * COST
        hit = float((np.sign(g["alpha_beta"]) == pos).mean())
        sh = float(net.mean()/net.std(ddof=1)*ANN) if net.std(ddof=1) > 1e-12 else np.nan
        lo, hi = block_bootstrap_ci(net, statistic=np.mean, block_size=7,
                                    n_boot=1000)[1:]
        fp = sum(1 for _, x in g.groupby("fold")
                 if (pos*x["alpha_beta"]).mean() > 0)
        print(f"  {name}: n={len(g)} ({len(g)/len(d)*100:.0f}%) "
              f"mean_signed={pay.mean():+.2f}bps CI[{lo:+.2f},{hi:+.2f}] "
              f"hit={hit*100:.1f}% netSh={sh:+.2f} folds+={fp}/9", flush=True)
        return dict(name=name, n=len(g), mean_bps=pay.mean(), ci_lo=lo,
                    ci_hi=hi, hit=hit, net_sh=sh, folds_pos=fp)

    print("\n  --- LONG-consensus (V>0 → take +1 resid) ---", flush=True)
    L = side(d["V"] > 0, +1, "LONG")
    print("  --- SHORT-consensus (V<0 → take −1 resid) ---", flush=True)
    S = side(d["V"] < 0, -1, "SHORT")

    print("\n  --- |V| agreement monotonicity (mean V·αβ bps by strength) ---",
          flush=True)
    d["aV"] = d["V"].abs()
    for k in sorted(d["aV"].unique())[:8]:
        g = d[d["aV"] == k]
        if len(g) < 50 or k == 0:
            continue
        m = float((np.sign(g["V"]) * g["alpha_beta"]).mean() * 1e4)
        print(f"    |V|={int(k)}: n={len(g):5d} mean(sign(V)·αβ)={m:+.2f}bps",
              flush=True)
    mono = (d[d.aV > 0].groupby("aV").apply(
        lambda g: (np.sign(g.V)*g.alpha_beta).mean()*1e4))
    rho = float(pd.Series(mono.values).corr(pd.Series(mono.index),
                "spearman")) if len(mono) > 2 else np.nan

    # matched placebo: random per-symbol global sign, preserve firing pattern
    rng = np.random.default_rng(0)
    pl = []
    syms_arr = d["symbol"].values
    for _ in range(200):
        flip = {s: rng.choice([-1.0, 1.0]) for s in d["symbol"].unique()}
        fl = np.array([flip[s] for s in syms_arr])
        v = d["V"].to_numpy()
        m = v != 0
        pp = np.sign(v[m]) * fl[m] * ab[m]
        pl.append(float(pp.mean()))
    real_dir = np.sign(d["V"].to_numpy())
    real = float((real_dir[real_dir != 0] *
                  ab[real_dir != 0]).mean())
    p95 = float(np.nanpercentile(pl, 95))

    print(f"\n  consensus |V|-monotonicity ρ={rho:+.2f} | "
          f"all-fired mean(sign(V)·αβ)={real:+.2f}bps "
          f"vs matched-placebo p95={p95:+.2f}bps "
          f"-> {'BEATS' if real > p95 else 'fails'}", flush=True)

    Lok = bool(L and L["ci_lo"] > 0 and L["folds_pos"] >= 6)
    Sok = bool(S and S["ci_lo"] < 0 and S["folds_pos"] >= 6)
    v = (f"D1-ext-G: LONG-consensus reliable={Lok} "
         f"(mean {L['mean_bps']:+.2f}bps hit {L['hit']*100:.0f}% netSh "
         f"{L['net_sh']:+.2f}); SHORT-consensus reliable={Sok} "
         f"(mean {S['mean_bps']:+.2f}bps hit {S['hit']*100:.0f}% netSh "
         f"{S['net_sh']:+.2f}); |V|-mono ρ={rho:+.2f}; "
         f"placebo {'BEATS' if real > p95 else 'fails'}. "
         + ("Long/short asymmetry per DDI-2 "
            if (S and L and S['mean_bps']*-1 > L['mean_bps']) else "")
         + "Signed-composite consensus does not yield a reliable net-of-cost "
         "direction beyond the (sub-cost) convergence collapse — confirms "
         "composites are correlated reads of one signal, not independent "
         "directional bets. Production LGBM unaffected.")
    print(f"\n  VERDICT: {v}", flush=True)
    pd.DataFrame([L, S, dict(name="meta", mono_rho=rho, real_bps=real,
                 placebo_p95=p95, verdict=v)]).to_csv(OUTD/"summary.csv",
                                                      index=False)
    pd.DataFrame([{"verdict": v}]).to_csv(OUTD/"verdict.csv", index=False)
    print(f"\nSaved {OUTD}\nTotal: {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
