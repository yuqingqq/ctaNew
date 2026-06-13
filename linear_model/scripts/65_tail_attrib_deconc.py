"""Step 65: reconstruct the nested-OOS mean-rev-v2 strategy with full position +
trade logging, then answer:

  (1) per-SYMBOL attribution of the top-5% tail cycles (blue-chip vs semi-meme)
  (2) per-TRADE stats (win rate, profit factor) — no annualization inflation
  (3) share of total edge coming from n_open==1 single-name cycles
  (4) robustness probe: does the nested edge / edge-vs-random survive
        - 'bothsides' : take exposure only when >=1 long AND >=1 short qualify
        - 'wtcap'     : cap |per-name weight| at 0.2 (no 0.5 single-name bet)

The 'design' variant is the user's actual strategy (single-name + BTC hedge
allowed). Probes (2)-(4) test whether the convexity is fundamentally single-
name-bet driven or survives de-concentration.
"""
from __future__ import annotations
import sys, time, importlib.util, warnings
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
REPO = Path("/home/yuqing/ctaNew")
sys.path.insert(0, str(REPO))
sp = importlib.util.spec_from_file_location("s64",
        REPO / "linear_model/scripts/64_meanrev_v2_backtest.py")
s64 = importlib.util.module_from_spec(sp); sp.loader.exec_module(s64)
s59 = s64.s59
from ml.research.alpha_v4_xs import block_bootstrap_ci

OUT = REPO / "linear_model/results/step65_tail_attrib"
OUT.mkdir(parents=True, exist_ok=True)
OOS, BLOCK = s64.OOS, s64.BLOCK
Z_OUT, MAX_HOLD, STOP_BPS, COST = s64.Z_OUT, s64.MAX_HOLD, s64.STOP_BPS, s64.COST
SEMI_MEME = {"PUMPUSDT", "WIFUSDT", "FARTCOINUSDT", "TRUMPUSDT", "SPXUSDT",
             "PENGUUSDT", "VIRTUALUSDT", "VVVUSDT", "BIOUSDT"}


def runL(apd, alpha_w, fund_w, pz_w, tic_w, sig, beta_w, p,
         constraint="design"):
    """Logged engine. constraint: design | bothsides | wtcap."""
    times = sorted(apd[apd["fold"].isin(OOS)]["open_time"].unique())[::BLOCK]
    fold_of = apd.drop_duplicates("open_time").set_index(
        "open_time")["fold"].to_dict()
    N, sub, zin, hur = p["N"], p["sub"], p["zin"], p["hurdle"]
    opn = {}; prev_w = {}; rows = []; trades = []; posrows = []
    for t in times:
        if t not in pz_w.index or t not in tic_w.index:
            continue
        pz = pz_w.loc[t]; ar = alpha_w.loc[t] if t in alpha_w.index else None
        fr = fund_w.loc[t] if t in fund_w.index else None
        tic = tic_w.loc[t]
        bt = beta_w.loc[t] if (beta_w is not None and t in beta_w.index) else None
        subset = (set(tic.dropna().sort_values(ascending=False).head(15).index)
                  if sub == "top15" else set(tic.index[tic > 0]))
        ss = [s for s in subset if pd.notna(pz.get(s))]
        if len(ss) >= 4:
            v = pz[ss]; mu, sd = v.mean(), v.std()
            z = (pz - mu) / sd if sd > 1e-12 else pz * 0.0
        else:
            z = pd.Series(dtype=float)
        # CAUSAL ORDER (bugfix 2026-05-16): the old engine added alpha[t]
        # (the forward [t,t+4h] return) to cum and bumped age BEFORE this exit
        # check, letting the stop/age exit see future return info. Exits must be
        # decided from cum/age known strictly before alpha[t]; cum/age are
        # updated only AFTER PnL, below, for positions carried through cycle t.
        for s in list(opn):
            st = opn[s]; zv = z.get(s, np.nan)
            ex = ((s not in subset) or (pd.notna(zv) and abs(zv) < Z_OUT)
                  or st["age"] >= MAX_HOLD or st["cum"] <= -STOP_BPS)
            if ex:
                trades.append(dict(symbol=s, side=st["side"], entry=st["e_t"],
                                   exit=t, age=st["age"], cum_bps=st["cum"]))
                del opn[s]
        held = set(opn)
        cl = {s for s, st in opn.items() if st["side"] > 0}
        cs = {s for s, st in opn.items() if st["side"] < 0}
        nl, ns = N - len(cl), N - len(cs)
        if len(ss) >= 4:
            zr = z[ss].sort_values(ascending=False)

            def ok(s):
                return abs(pz.get(s, 0.0)) * float(sig.get(s, s64.sig_med)) * 1e4 >= hur
            for s in zr.index:
                if nl <= 0 or zr[s] < zin:
                    break
                if s in held or not ok(s):
                    continue
                opn[s] = dict(side=1, cum=0.0, age=0, e_t=t); held.add(s); nl -= 1
            for s in reversed(list(zr.index)):
                if ns <= 0 or zr[s] > -zin:
                    break
                if s in held or not ok(s):
                    continue
                opn[s] = dict(side=-1, cum=0.0, age=0, e_t=t); held.add(s); ns -= 1
        L = sum(1 for st in opn.values() if st["side"] > 0)
        S = sum(1 for st in opn.values() if st["side"] < 0)
        flat = (constraint == "bothsides" and (L == 0 or S == 0))
        w = {}
        if not flat:
            for s, st in opn.items():
                wi = (0.5 / L) if st["side"] > 0 and L else \
                     (-0.5 / S if st["side"] < 0 and S else 0.0)
                if constraint == "wtcap":
                    wi = max(-0.2, min(0.2, wi))
                w[s] = wi
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
                posrows.append(dict(time=t, fold=fold_of.get(t, 0), symbol=s,
                                    side=opn[s]["side"], weight=wi,
                                    contrib_bps=wi * a * 1e4))
                period_pnl[s] = (a if opn[s]["side"] > 0 else -a) * 1e4
            fv = fr.get(s) if fr is not None else np.nan
            if fv is not None and not pd.isna(fv):
                fund += -wi * fv * 1e4
        allk = set(w) | set(prev_w)
        cost = sum(abs(w.get(k, 0) - prev_w.get(k, 0)) for k in allk) * COST
        cost += abs(w_btc - prev_w.get("__BTC__", 0.0)) * COST
        rows.append(dict(time=t, fold=fold_of.get(t, 0), gross=gross,
                         funding=fund, cost=cost, net=gross + fund - cost,
                         n_open=len(opn) if not flat else 0))
        # CAUSAL: alpha[t] now realized — update carried positions' cum/age
        # (consumed by the NEXT cycle's exit decision only).
        for s, pnl in period_pnl.items():
            if s in opn:
                opn[s]["cum"] += pnl
                opn[s]["age"] += 1
        pw = dict(w); pw["__BTC__"] = w_btc; prev_w = pw
    return (pd.DataFrame(rows), pd.DataFrame(trades),
            pd.DataFrame(posrows))


def nested(apd, aw, fw, pzw, tw, sig, bw, constraint):
    """Reconstruct the per-fold nested-OOS pick (same rule as Step 64) with
    full logging; return concatenated per-cycle/trade/pos for the nested path."""
    runs = {}
    for i, p in enumerate(s64.GRID):
        runs[i] = runL(apd, aw, fw, pzw, tw, sig, bw, p, constraint)
    nd, ntr, npo = [], [], []
    for k in range(1, 10):
        if k == 1:
            pick = sorted(runs, key=lambda j: s59._sharpe(
                runs[j][0]["net"]))[len(s64.GRID) // 2]
        else:
            pick = max(runs, key=lambda j: s59._sharpe(
                runs[j][0][runs[j][0]["fold"] < k]["net"].to_numpy()))
        df, tr, po = runs[pick]
        nd.append(df[df["fold"] == k])
        if len(tr):
            ntr.append(tr[tr["exit"].isin(df[df["fold"] == k]["time"])])
        if len(po):
            npo.append(po[po["fold"] == k])
    return (pd.concat(nd).sort_values("time"),
            pd.concat(ntr) if ntr else pd.DataFrame(),
            pd.concat(npo) if npo else pd.DataFrame())


def report(tag, df, tr, po):
    n = df["net"].to_numpy(); tot = n.sum()
    sh = s59._sharpe(n)
    lo, hi = block_bootstrap_ci(n, statistic=s59._sharpe, block_size=7,
                                n_boot=1000)[1:]
    fp = sum(1 for _, g in df.groupby("fold") if s59._sharpe(g["net"]) > 0)
    a = np.abs(n); thr = np.quantile(a, 0.95); tm = a >= thr
    body = n[~tm]
    one = df["n_open"] == 1
    one_pnl = df.loc[one, "net"].sum()
    sh_no1 = s59._sharpe(df.loc[~one, "net"].to_numpy())
    print(f"\n--- {tag} ---", flush=True)
    print(f"  nested Sharpe {sh:+.2f} [{lo:+.2f},{hi:+.2f}] fp={fp}/9 "
          f"total {tot:,.0f}bps", flush=True)
    print(f"  body(95%) {body.sum():+,.0f}bps Sh {s59._sharpe(body):+.2f} "
          f"| n_open==1 cycles {one.sum()} carry {one_pnl:,.0f}bps "
          f"({one_pnl/tot*100 if tot else 0:.0f}%); Sh excl n_open1 {sh_no1:+.2f}",
          flush=True)
    if len(tr):
        c = tr["cum_bps"]
        pf = c[c > 0].sum() / (-c[c < 0].sum()) if (c < 0).any() else float("inf")
        print(f"  trades {len(tr)} | win {(c>0).mean()*100:.0f}% | "
              f"profit-factor {pf:.2f} | mean +{c[c>0].mean():.0f}/"
              f"{c[c<0].mean():.0f} | best {c.max():+.0f} worst {c.min():+.0f} "
              f"| mean hold {tr['age'].mean():.1f} cyc", flush=True)
    if len(po):
        tcyc = set(df[tm]["time"])
        pt = po[po["time"].isin(tcyc)]
        att = pt.groupby("symbol")["contrib_bps"].agg(["sum", "count"]).sort_values(
            "sum", ascending=False)
        att["semi_meme"] = [s in SEMI_MEME for s in att.index]
        sm = att[att.semi_meme]["sum"].sum(); al = att["sum"].sum()
        print(f"  TAIL-cycle attribution (top {len(att)} syms, tail={tm.sum()} cyc):",
              flush=True)
        print(att.head(8).to_string(), flush=True)
        print(f"  semi-meme share of tail PnL: {sm:,.0f}/{al:,.0f} = "
              f"{sm/al*100 if al else 0:.0f}%  (semi-meme set hit: "
              f"{sorted(set(att[att.semi_meme].index))})", flush=True)
        att.to_csv(OUT / f"attrib_{tag}.csv")


def main():
    print("=" * 100, flush=True)
    print("  STEP 65: tail attribution + de-concentration probe", flush=True)
    print("=" * 100, flush=True)
    t0 = time.time()
    apd = pd.read_parquet(s64.PREDS)
    apd["open_time"] = pd.to_datetime(apd["open_time"], utc=True)
    syms = sorted(apd["symbol"].unique())
    f0 = apd[apd["fold"] == 0]
    sg = f0.groupby("symbol")["alpha_beta"].std()
    s64.sig_med = float(sg.dropna().median())
    sig = sg.fillna(s64.sig_med).to_dict()
    sampled = sorted(apd[apd["fold"].isin(OOS)]["open_time"].unique())[::BLOCK]
    aw = apd.pivot_table(index="open_time", columns="symbol",
                         values="alpha_beta", aggfunc="first").sort_index()
    pzw = apd.pivot_table(index="open_time", columns="symbol",
                          values="pred_z", aggfunc="first").sort_index()
    tw = apd.pivot_table(index="open_time", columns="symbol",
                         values="trail_ic", aggfunc="first").sort_index()
    fw, _ = s59.infer_funding(syms, sampled)
    bw = s64.recover_beta(apd)
    print(f"  {len(syms)} syms; β recovered {bw is not None}", flush=True)

    for c in ["design", "bothsides", "wtcap"]:
        nd, ntr, npo = nested(apd, aw, fw, pzw, tw, sig, bw, c)
        nd.to_csv(OUT / f"nested_{c}.csv", index=False)
        report(c, nd, ntr, npo)

    # random-pool edge reference under the user's design
    pp = []
    for sd in range(60):
        rg = np.random.default_rng(900 + sd); shp = pzw.copy()
        for tt in shp.index:
            vv = shp.loc[tt].values.copy(); m = ~pd.isna(vv); idx = np.where(m)[0]
            o = vv.copy(); o[idx] = vv[rg.permutation(idx)]; shp.loc[tt] = o
        d, _, _ = runL(apd, aw, fw, shp, tw, sig, bw,
                       dict(sub="ic_pos", zin=2.0, hurdle=0, N=3), "design")
        pp.append(s59._sharpe(d["net"].to_numpy()))
    print(f"\n  random-pool ref (design, single best cfg, 60 seeds): "
          f"p95={np.percentile(pp,95):+.2f} mean={np.mean(pp):+.2f}", flush=True)
    print(f"\nTotal: {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
