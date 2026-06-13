"""Step 64: mean-reversion-exit v2 — user-locked (2026-05-16), corrected for the
pred_z scale finding (raw pred_z std≈0.07, NOT 1; lag-1 autocorr +0.73).

Signal standardized PER CYCLE, CROSS-SECTIONALLY within the rolling-IC subset:
    z = (pred_z − mean_subset) / std_subset
Entry  : among subset, |z| ≥ z_in (extreme vs peers) AND ranked top/bottom AND
         implied-bps gate |pred_z|·σ_idio·1e4 ≥ hurdle (σ_idio = fold-0 PIT std)
Exit   : |z| < z_out(=1.0)  [reverted toward middle]  OR  left the subset
         OR  age ≥ MAX_HOLD(72h)  OR  cum α_β ≤ −STOP_BPS(80)
Hedge  : underfilled side → fewer names + BTC leg = −Σwᵢβᵢ (turnover cost);
         β_pit recovered exactly via β=(return_pct−α_β)/btc_ret. PnL on α_β.

Grid (12): subset{top15, ic_pos} × z_in{1.5,2.0} × hurdle{0,5,9}, N=3.
Headline = nested-OOS. Plus fixed-24h baseline (same entries, time-only exit),
random-exit & random-pool placebos, 6 criteria, gate-block %, mean hold.

Honest prior: per-cycle IC ≈ −0.013 → a persistent but ~zero-skill signal;
expected to FAIL, but now behaves as intended (real multi-cycle holds) so the
refutation is valid rather than degenerate.
"""
from __future__ import annotations
import sys, time, importlib.util, warnings
from pathlib import Path
from itertools import product
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
REPO = Path("/home/yuqing/ctaNew")
sys.path.insert(0, str(REPO))


def _imp(n, r):
    sp = importlib.util.spec_from_file_location(n, REPO / r)
    m = importlib.util.module_from_spec(sp); sp.loader.exec_module(m); return m

psl = _imp("psl", "scripts/phase_ah_sleeve.py")
s59 = _imp("s59", "linear_model/scripts/59_clean108_funding.py")
from ml.research.alpha_v4_xs import block_bootstrap_ci

PREDS = REPO / "linear_model/results/step62_bluechip44/predictions.parquet"
BTC5M = REPO / "data/ml/test/parquet/klines/BTCUSDT/5m"
OUT = REPO / "linear_model/results/step64_meanrev_v2"
OUT.mkdir(parents=True, exist_ok=True)
OOS = list(range(1, 10))
COST = psl.COST_PER_UNIT_ABS_DELTA
BLOCK = psl.HORIZON_ENTRY                       # 48 bars = 4h
Z_OUT = 1.0                                    # exit when |z| reverts below this
MAX_HOLD = 18                                  # 72h backstop
STOP_BPS = 80.0
N_SLOTS = 3
GRID = [dict(sub=s, zin=zi, hurdle=h, N=N_SLOTS)
        for s, zi, h in product(["top15", "ic_pos"], [1.5, 2.0], [0, 5, 9])]
sig_med = 0.0127


def recover_beta(apd):
    fs = sorted(BTC5M.glob("*.parquet"))
    if not fs:
        return None
    b = pd.concat([pd.read_parquet(f, columns=["open_time", "close"]) for f in fs],
                  ignore_index=True)
    b["open_time"] = pd.to_datetime(b["open_time"], utc=True)
    b = b.drop_duplicates("open_time").sort_values("open_time").set_index("open_time")
    bret = (b["close"].shift(-BLOCK) / b["close"] - 1.0).rename("btc_ret")
    a = apd[["symbol", "open_time", "return_pct", "alpha_beta"]].copy()
    a = a.merge(bret, left_on="open_time", right_index=True, how="left")
    den = a["btc_ret"].where(a["btc_ret"].abs() > 1e-4)
    a["beta"] = ((a["return_pct"] - a["alpha_beta"]) / den).clip(-3, 3)
    return a.pivot_table(index="open_time", columns="symbol", values="beta",
                         aggfunc="first").sort_index().ffill()


def run(apd, alpha_w, fund_w, pz_w, tic_w, sig, beta_w, p,
        mode="meanrev", rng=None, rholds=None):
    times = sorted(apd[apd["fold"].isin(OOS)]["open_time"].unique())[::BLOCK]
    fold_of = apd.drop_duplicates("open_time").set_index("open_time")["fold"].to_dict()
    N, sub, zin, hur = p["N"], p["sub"], p["zin"], p["hurdle"]
    mh = p.get("mh", MAX_HOLD)
    opn = {}; prev_w = {}; rows = []; holds = []; gated = filled = 0
    for t in times:
        if t not in pz_w.index or t not in tic_w.index:
            continue
        pz = pz_w.loc[t]; ar = alpha_w.loc[t] if t in alpha_w.index else None
        fr = fund_w.loc[t] if t in fund_w.index else None
        tic = tic_w.loc[t]
        bt = beta_w.loc[t] if (beta_w is not None and t in beta_w.index) else None
        # rolling-IC subset
        if sub == "top15":
            subset = set(tic.dropna().sort_values(ascending=False).head(15).index)
        else:
            subset = set(tic.index[tic > 0])
        ss = [s for s in subset if pd.notna(pz.get(s))]
        if len(ss) < 4:
            # nothing tradeable; still mark/age/PnL existing book then continue
            z = pd.Series(dtype=float)
        else:
            v = pz[ss]; mu, sd = v.mean(), v.std()
            z = (pz - mu) / sd if sd > 1e-12 else pz * 0.0
        # CAUSAL ORDER (bugfix 2026-05-16): the old engine added alpha[t]
        # (forward [t,t+4h]) to cum and bumped age BEFORE this exit check,
        # letting stop/age exits see future return. Exit must be decided from
        # cum/age known strictly before alpha[t]; both are updated AFTER PnL
        # (below) for positions carried through cycle t.
        for s in [s for s in list(opn)]:
            st = opn[s]; zv = z.get(s, np.nan)
            if mode == "fixed":
                ex = st["age"] >= mh
            elif mode == "random":
                ex = st["age"] >= st["rh"]
            else:
                ex = ((s not in subset) or (pd.notna(zv) and abs(zv) < Z_OUT)
                      or st["age"] >= MAX_HOLD or st["cum"] <= -STOP_BPS)
            if ex:
                holds.append(st["age"]); del opn[s]
        # refill within subset, ranked by cross-sectional z, |z|>=zin + bps gate
        held = set(opn)
        cl = {s for s, st in opn.items() if st["side"] > 0}
        cs = {s for s, st in opn.items() if st["side"] < 0}
        nl, ns = N - len(cl), N - len(cs)
        if len(ss) >= 4:
            zr = z[ss].sort_values(ascending=False)

            def ok(s):
                return abs(pz.get(s, 0.0)) * float(sig.get(s, sig_med)) * 1e4 >= hur
            for s in zr.index:                       # longs: most positive z
                if nl <= 0 or zr[s] < zin:
                    break
                if s in held:
                    continue
                if not ok(s):
                    gated += 1; continue
                opn[s] = dict(side=1, cum=0.0, age=0,
                              rh=int(rng.choice(rholds)) if mode == "random"
                              and rholds is not None and len(rholds) else mh)
                held.add(s); nl -= 1; filled += 1
            for s in reversed(list(zr.index)):       # shorts: most negative z
                if ns <= 0 or zr[s] > -zin:
                    break
                if s in held:
                    continue
                if not ok(s):
                    gated += 1; continue
                opn[s] = dict(side=-1, cum=0.0, age=0,
                              rh=int(rng.choice(rholds)) if mode == "random"
                              and rholds is not None and len(rholds) else mh)
                held.add(s); ns -= 1; filled += 1
        # weights ±0.5/side + BTC hedge on residual net beta
        L = sum(1 for st in opn.values() if st["side"] > 0)
        S = sum(1 for st in opn.values() if st["side"] < 0)
        w = {}
        for s, st in opn.items():
            w[s] = (0.5 / L) if st["side"] > 0 and L else \
                   (-0.5 / S if st["side"] < 0 and S else 0.0)
        net_beta = 0.0
        if bt is not None:
            for s, wi in w.items():
                bv = bt.get(s)
                if bv is not None and not pd.isna(bv):
                    net_beta += wi * bv
        w_btc = -net_beta
        gross = fund = 0.0
        period_pnl = {}
        for s, wi in w.items():
            a = ar.get(s) if ar is not None else np.nan
            if a is not None and not pd.isna(a):
                gross += wi * a * 1e4
                period_pnl[s] = (a if opn[s]["side"] > 0 else -a) * 1e4
            fv = fr.get(s) if fr is not None else np.nan
            if fv is not None and not pd.isna(fv):
                fund += -wi * fv * 1e4
        allk = set(w) | set(prev_w)
        cost = sum(abs(w.get(k, 0) - prev_w.get(k, 0)) for k in allk) * COST
        cost += abs(w_btc - prev_w.get("__BTC__", 0.0)) * COST
        rows.append(dict(time=t, fold=fold_of.get(t, 0), gross=gross, funding=fund,
                         cost=cost, net=gross + fund - cost, n_open=len(opn),
                         net_beta=net_beta))
        # CAUSAL: alpha[t] realized — update carried positions' cum/age
        # (consumed only by the NEXT cycle's exit decision).
        for s, pnl in period_pnl.items():
            if s in opn:
                opn[s]["cum"] += pnl
                opn[s]["age"] += 1
        pw = dict(w); pw["__BTC__"] = w_btc; prev_w = pw
    return pd.DataFrame(rows), holds, (gated, filled)


def _summ(df):
    n = df["net"].to_numpy(); a = np.abs(n); o = np.argsort(-a)
    t5 = (n[o[:max(1, int(len(n) * .05))]].sum() / n.sum() * 100
          if n.sum() else 0.0)
    fp = sum(1 for _, g in df.groupby("fold") if s59._sharpe(g["net"]) > 0)
    return s59._sharpe(n), fp, t5


def main():
    print("=" * 100, flush=True)
    print("  STEP 64: mean-reversion-exit v2 (cross-sectional z, corrected scale)",
          flush=True)
    print("=" * 100, flush=True)
    t0 = time.time()
    if not PREDS.exists():
        print("  MISSING Step 62 predictions.", flush=True); return
    apd = pd.read_parquet(PREDS)
    apd["open_time"] = pd.to_datetime(apd["open_time"], utc=True)
    syms = sorted(apd["symbol"].unique())
    f0 = apd[apd["fold"] == 0]
    sg = f0.groupby("symbol")["alpha_beta"].std()
    global sig_med
    sig_med = float(sg.dropna().median())
    sig = sg.fillna(sig_med).to_dict()
    print(f"  {len(syms)} syms; σ_idio fold-0 median {sig_med:.5f}; "
          f"raw pred_z std {apd['pred_z'].std():.3f} (shrunk — using cross-sec z)",
          flush=True)

    sampled = sorted(apd[apd["fold"].isin(OOS)]["open_time"].unique())[::BLOCK]
    alpha_w = apd.pivot_table(index="open_time", columns="symbol",
                              values="alpha_beta", aggfunc="first").sort_index()
    pz_w = apd.pivot_table(index="open_time", columns="symbol",
                           values="pred_z", aggfunc="first").sort_index()
    tic_w = apd.pivot_table(index="open_time", columns="symbol",
                            values="trail_ic", aggfunc="first").sort_index()
    fund_w, _ = s59.infer_funding(syms, sampled)
    beta_w = recover_beta(apd)
    print(f"  β_pit recovered: {'yes' if beta_w is not None else 'NO'}", flush=True)

    print("\n  grid sweep (in-sample ceiling, NOT headline):", flush=True)
    pg = {}
    for i, p in enumerate(GRID):
        df, hd, gf = run(apd, alpha_w, fund_w, pz_w, tic_w, sig, beta_w, p)
        pg[i] = df
        sh, fp, t5 = _summ(df)
        mh_ = float(np.mean(hd)) if hd else 0.0
        br = gf[0] / max(gf[0] + gf[1], 1) * 100
        print(f"    [{i:2d}] {p} Sh={sh:+.2f} fp={fp}/9 meanHold={mh_:.1f} "
              f"gate-blk={br:.0f}%", flush=True)
    bi = max(pg, key=lambda k: s59._sharpe(pg[k]["net"].to_numpy()))
    sh_b, _, _ = _summ(pg[bi]); bp = GRID[bi]

    nested = []
    for k in range(1, 10):
        if k == 1:
            pick = sorted(pg, key=lambda j: s59._sharpe(pg[j]["net"]))[len(GRID)//2]
        else:
            pick = max(pg, key=lambda j: s59._sharpe(
                pg[j][pg[j]["fold"] < k]["net"].to_numpy()))
        nested.append(pg[pick][pg[pick]["fold"] == k])
    nd = pd.concat(nested).sort_values("time")
    nsh = s59._sharpe(nd["net"].to_numpy())
    lo, hi = block_bootstrap_ci(nd["net"].to_numpy(), statistic=s59._sharpe,
                                block_size=7, n_boot=1000)[1:]
    _, nfp, nt5 = _summ(nd)
    nd.to_csv(OUT / "nested_oos_per_cycle.csv", index=False)

    fx, _, _ = run(apd, alpha_w, fund_w, pz_w, tic_w, sig, beta_w,
                   dict(sub=bp["sub"], zin=bp["zin"], hurdle=bp["hurdle"],
                        N=bp["N"], mh=6), "fixed")
    sh_fx, _, _ = _summ(fx)
    _, hb, _ = run(apd, alpha_w, fund_w, pz_w, tic_w, sig, beta_w, bp)
    hb = np.array(hb) if len(hb) else np.array([MAX_HOLD])
    pe = []
    for sd in range(100):
        d, _, _ = run(apd, alpha_w, fund_w, pz_w, tic_w, sig, beta_w, bp,
                      "random", np.random.default_rng(sd), hb)
        pe.append(s59._sharpe(d["net"].to_numpy()))
    pe95 = float(np.percentile(pe, 95))
    pp = []
    for sd in range(100):
        rg = np.random.default_rng(900 + sd); shp = pz_w.copy()
        for tt in shp.index:
            vv = shp.loc[tt].values.copy(); m = ~pd.isna(vv); idx = np.where(m)[0]
            o = vv.copy(); o[idx] = vv[rg.permutation(idx)]; shp.loc[tt] = o
        d, _, _ = run(apd, alpha_w, fund_w, shp, tic_w, sig, beta_w, bp)
        pp.append(s59._sharpe(d["net"].to_numpy()))
    pp95 = float(np.percentile(pp, 95))

    print(f"\n{'='*100}\n  VERDICT\n{'='*100}", flush=True)
    print(f"  nested-OOS Sharpe : {nsh:+.2f} [{lo:+.2f},{hi:+.2f}] "
          f"fp={nfp}/9 top5%cyc={nt5:.0f}%", flush=True)
    print(f"  in-sample ceiling : {sh_b:+.2f} (NOT headline; best={bp})", flush=True)
    print(f"  fixed-24h baseline: {sh_fx:+.2f} -> beats? "
          f"{'YES' if nsh > sh_fx else 'NO'}", flush=True)
    print(f"  random-exit p95   : {pe95:+.2f} -> {'PASS' if nsh > pe95 else 'FAIL'}",
          flush=True)
    print(f"  random-pool p95   : {pp95:+.2f} -> {'PASS' if nsh > pp95 else 'FAIL'}",
          flush=True)
    crit = [lo > 0, nsh > sh_fx, nsh > pe95, nsh > pp95, nfp >= 6, nt5 < 35]
    print(f"\n  [CI>0,>base,>rexit,>rpool,fp>=6,not-tail] = {crit}", flush=True)
    print(f"  {'ALL PASS — executable candidate' if all(crit) else 'FAIL — close'}",
          flush=True)
    pd.DataFrame([dict(nested=nsh, lo=lo, hi=hi, fp=nfp, top5=nt5, ceiling=sh_b,
                       fixed=sh_fx, rexit=pe95, rpool=pp95, best=str(bp),
                       all_pass=all(crit))]).to_csv(OUT / "verdict.csv", index=False)
    print(f"\nTotal: {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
