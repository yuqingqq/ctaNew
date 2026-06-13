"""Step 63: event-driven mean-reversion-exit / ranked-pool backtest + validation.

Architecture (user-locked 2026-05-15):
  - universe = Step 62 blue-chip 44 (HL & vol>=$2M)
  - market-neutral L/S, rank by pred_B, long top-N / short bottom-N
  - positions PERSIST; hybrid exit = first of {decay, target, time, stop}
  - refill freed slots each 4h from the ranked list
  - PnL on alpha_beta (already BTC-hedged at PIT β); causal convention
  - funding included (proven immaterial) for parity with Steps 59/60

Validation (must pass ALL — see docs/MEANREV_EXIT_PLAN.md):
  1. nested-OOS Sharpe CI > 0   (headline; NOT in-sample best)
  2. beats same-universe FIXED-24h-hold baseline
  3. beats RANDOM-EXIT placebo p95 (same entries, random holds)
  4. beats RANDOM-POOL placebo p95 (shuffled pred_B → random selection)
  5. per-fold >=6/9, per-cycle PnL not tail-concentrated
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


def _imp(name, rel):
    sp = importlib.util.spec_from_file_location(name, REPO / rel)
    m = importlib.util.module_from_spec(sp); sp.loader.exec_module(m); return m

psl = _imp("psl", "scripts/phase_ah_sleeve.py")
s59 = _imp("s59", "linear_model/scripts/59_clean108_funding.py")
from ml.research.alpha_v4_xs import block_bootstrap_ci

PREDS = REPO / "linear_model/results/step62_bluechip44/predictions.parquet"
OUT = REPO / "linear_model/results/step63_meanrev"
OUT.mkdir(parents=True, exist_ok=True)
OOS_FOLDS = list(range(1, 10))
MIN_HISTORY_DAYS = 60
COST = psl.COST_PER_UNIT_ABS_DELTA
BLOCK = psl.HORIZON_ENTRY                       # 48 bars = 4h cycle

# pre-registered SMALL grid (16 combos) — headline is nested-OOS over this grid
GRID = [dict(decay=d, tgt=tg, mh=mh, stop=80, N=n)
        for d, tg, mh, n in product([0.3, 0.5], [50, 100],
                                     [6, 12], [3, 5])]   # mh in 4h-cycles (24h,48h)


def _eligible(apd):
    """per cycle t -> set of symbols present with finite pred_B, PIT-listed."""
    lst = s59.get_listings()
    for s, t in apd.groupby("symbol")["open_time"].min().items():
        if s not in lst:
            lst[s] = t.tz_localize("UTC") if t.tz is None else t.tz_convert("UTC")
    elig = {}
    for t, g in apd.groupby("open_time"):
        cut = t - pd.Timedelta(days=MIN_HISTORY_DAYS)
        ok = g[g["pred_B"].notna()]
        elig[t] = set(s for s in ok["symbol"]
                      if lst.get(s) is not None and lst[s] <= cut)
    return elig


def run_engine(apd, alpha_w, fund_w, predB_w, p, mode="hybrid",
               rng=None, rand_holds=None):
    """Event-driven L/S. mode: hybrid | fixed (time-only) | random (rand hold).
    Returns per-cycle df + list of realized hold lengths (cycles)."""
    times = sorted(apd[apd["fold"].isin(OOS_FOLDS)]["open_time"].unique())
    times = times[::BLOCK]
    fold_of = apd.drop_duplicates("open_time").set_index("open_time")["fold"].to_dict()
    N, mh = p["N"], p["mh"]
    open_pos = {}   # sym -> dict(side, entry_pb_abs, cum, age, rhold)
    prev_w = {}
    rows, holds = [], []
    for t in times:
        pbrow = predB_w.loc[t] if t in predB_w.index else None
        arow = alpha_w.loc[t] if t in alpha_w.index else None
        frow = fund_w.loc[t] if t in fund_w.index else None
        # 1. mark + age open positions
        for s, st in open_pos.items():
            a = (arow.get(s) if arow is not None else np.nan)
            if a is not None and not pd.isna(a):
                st["cum"] += (a if st["side"] > 0 else -a) * 1e4
            st["age"] += 1
        # 2. exits
        to_close = []
        for s, st in open_pos.items():
            pb = (pbrow.get(s) if pbrow is not None else np.nan)
            ex = False
            if mode == "fixed":
                ex = st["age"] >= mh
            elif mode == "random":
                ex = st["age"] >= st["rhold"]
            else:  # hybrid
                decayed = (pd.notna(pb) and abs(pb) < p["decay"] * st["entry"]) or \
                          (pd.notna(pb) and np.sign(pb) == -st["side"])
                ex = (decayed or st["cum"] >= p["tgt"] or st["age"] >= mh
                      or st["cum"] <= -p["stop"])
            if ex:
                to_close.append(s)
        for s in to_close:
            holds.append(open_pos[s]["age"]); del open_pos[s]
        # 3. refill from ranked pred_B
        if pbrow is not None:
            r = pbrow.dropna().sort_values(ascending=False)
            cur_l = {s for s, st in open_pos.items() if st["side"] > 0}
            cur_s = {s for s, st in open_pos.items() if st["side"] < 0}
            need_l, need_s = N - len(cur_l), N - len(cur_s)
            held = set(open_pos)
            for s in r.index:
                if need_l <= 0: break
                if s in held: continue
                open_pos[s] = dict(side=1, entry=abs(r[s]), cum=0.0, age=0,
                                   rhold=(int(rng.choice(rand_holds))
                                          if mode == "random" and rand_holds is not None
                                          and len(rand_holds) else mh))
                held.add(s); need_l -= 1
            for s in reversed(list(r.index)):
                if need_s <= 0: break
                if s in held: continue
                open_pos[s] = dict(side=-1, entry=abs(r[s]), cum=0.0, age=0,
                                   rhold=(int(rng.choice(rand_holds))
                                          if mode == "random" and rand_holds is not None
                                          and len(rand_holds) else mh))
                held.add(s); need_s -= 1
        # 4. weights (balanced books, gross 1) + pnl/cost/funding
        nl = sum(1 for st in open_pos.values() if st["side"] > 0)
        ns = sum(1 for st in open_pos.values() if st["side"] < 0)
        w = {}
        for s, st in open_pos.items():
            w[s] = (0.5 / nl) if st["side"] > 0 and nl else \
                   (-0.5 / ns if st["side"] < 0 and ns else 0.0)
        gross = funding = 0.0
        for s, wi in w.items():
            a = (arow.get(s) if arow is not None else np.nan)
            if a is not None and not pd.isna(a):
                gross += wi * a * 1e4
            fv = (frow.get(s) if frow is not None else np.nan)
            if fv is not None and not pd.isna(fv):
                funding += -wi * fv * 1e4
        allk = set(w) | set(prev_w)
        cost = sum(abs(w.get(k, 0) - prev_w.get(k, 0)) for k in allk) * COST
        rows.append(dict(time=t, fold=fold_of.get(t, 0), gross=gross,
                          funding=funding, cost=cost,
                          net=gross + funding - cost, n_open=len(open_pos)))
        prev_w = w
    return pd.DataFrame(rows), holds


def _summ(df):
    n = df["net"].to_numpy()
    a = np.abs(n); o = np.argsort(-a)
    t5 = (n[o[:max(1, int(len(n) * .05))]].sum() / n.sum() * 100
          if n.sum() else 0.0)
    fp = sum(1 for _, g in df.groupby("fold") if s59._sharpe(g["net"]) > 0)
    return s59._sharpe(n), fp, t5


def main():
    print("=" * 100, flush=True)
    print("  STEP 63: mean-reversion-exit / ranked-pool backtest + validation", flush=True)
    print("=" * 100, flush=True)
    t0 = time.time()
    if not PREDS.exists():
        print(f"  MISSING {PREDS} — run Step 62 first.", flush=True); return
    apd = pd.read_parquet(PREDS)
    apd["open_time"] = pd.to_datetime(apd["open_time"], utc=True)
    syms = sorted(apd["symbol"].unique())
    print(f"  universe {len(syms)} syms, {len(apd):,} rows", flush=True)

    sampled = sorted(apd[apd["fold"].isin(OOS_FOLDS)]["open_time"].unique())[::BLOCK]
    alpha_w = apd.pivot_table(index="open_time", columns="symbol",
                               values="alpha_beta", aggfunc="first").sort_index()
    predB_w = apd.pivot_table(index="open_time", columns="symbol",
                               values="pred_B", aggfunc="first").sort_index()
    fund_w, _ = s59.infer_funding(syms, sampled)

    # ---- grid sweep (in-sample ceiling) ----
    print("\n  grid sweep (in-sample ceiling, NOT headline):", flush=True)
    per_grid = {}
    for i, p in enumerate(GRID):
        df, _ = run_engine(apd, alpha_w, fund_w, predB_w, p, "hybrid")
        per_grid[i] = df
        sh, fp, t5 = _summ(df)
        if i % 4 == 0 or sh > 1.0:
            print(f"    [{i:2d}] {p} -> Sh={sh:+.2f} fp={fp}/9 top5%cyc={t5:.0f}%",
                  flush=True)
    best_i = max(per_grid, key=lambda k: s59._sharpe(per_grid[k]["net"].to_numpy()))
    sh_b, fp_b, t5_b = _summ(per_grid[best_i])
    print(f"  in-sample BEST [{best_i}] {GRID[best_i]} Sh={sh_b:+.2f} "
          f"(ceiling only)", flush=True)

    # ---- nested-OOS (headline) ----
    nested = []
    for k in range(1, 10):
        if k == 1:
            pick = best_i  # fold1: no prior history -> use grid-median proxy
            pick = sorted(per_grid,
                          key=lambda j: s59._sharpe(per_grid[j]["net"]))[len(GRID)//2]
        else:
            pick = max(per_grid, key=lambda j: s59._sharpe(
                per_grid[j][per_grid[j]["fold"] < k]["net"].to_numpy()))
        nested.append(per_grid[pick][per_grid[pick]["fold"] == k])
    nd = pd.concat(nested).sort_values("time")
    nshe = s59._sharpe(nd["net"].to_numpy())
    lo, hi = block_bootstrap_ci(nd["net"].to_numpy(), statistic=s59._sharpe,
                                 block_size=7, n_boot=1000)[1:]
    _, nfp, nt5 = _summ(nd)   # _summ -> (sharpe, folds_pos, top5%); fixed unpack
    nd.to_csv(OUT / "nested_oos_per_cycle.csv", index=False)

    # ---- fixed-24h-hold baseline (same universe) ----
    fx, _ = run_engine(apd, alpha_w, fund_w, predB_w,
                        dict(decay=0, tgt=1e9, mh=6, stop=1e9, N=GRID[best_i]["N"]),
                        "fixed")
    sh_fx, fp_fx, _ = _summ(fx)

    # ---- placebos under the nested-chosen behaviour ----
    best_p = GRID[best_i]
    _, hb = run_engine(apd, alpha_w, fund_w, predB_w, best_p, "hybrid")
    hb = np.array(hb) if len(hb) else np.array([best_p["mh"]])
    rng = np.random.default_rng(0)
    pe = []  # random-exit
    for sd in range(100):
        d, _ = run_engine(apd, alpha_w, fund_w, predB_w, best_p, "random",
                          np.random.default_rng(sd), hb)
        pe.append(s59._sharpe(d["net"].to_numpy()))
    pe = np.array(pe); pe95 = float(np.percentile(pe, 95))
    pp = []  # random-pool (shuffle pred_B within cycle)
    for sd in range(100):
        rg = np.random.default_rng(1000 + sd)
        shuf = predB_w.copy()
        for t in shuf.index:
            v = shuf.loc[t].values.copy(); m = ~pd.isna(v)
            idx = np.where(m)[0]; perm = rg.permutation(idx)
            vv = v.copy(); vv[idx] = v[perm]; shuf.loc[t] = vv
        d, _ = run_engine(apd, alpha_w, fund_w, shuf, best_p, "hybrid")
        pp.append(s59._sharpe(d["net"].to_numpy()))
    pp = np.array(pp); pp95 = float(np.percentile(pp, 95))

    print(f"\n{'='*100}\n  VERDICT\n{'='*100}", flush=True)
    print(f"  nested-OOS Sharpe   : {nshe:+.2f} [{lo:+.2f},{hi:+.2f}]  "
          f"folds+={nfp}/9  top5%cyc={nt5:.0f}%", flush=True)
    print(f"  in-sample ceiling   : {sh_b:+.2f} (NOT headline)", flush=True)
    print(f"  fixed-24h baseline  : {sh_fx:+.2f}  -> nested beats it? "
          f"{'YES' if nshe > sh_fx else 'NO'}", flush=True)
    print(f"  random-exit  p95    : {pe95:+.2f}  -> {'PASS' if nshe > pe95 else 'FAIL'}",
          flush=True)
    print(f"  random-pool  p95    : {pp95:+.2f}  -> {'PASS' if nshe > pp95 else 'FAIL'}",
          flush=True)
    crit = [lo > 0, nshe > sh_fx, nshe > pe95, nshe > pp95, nfp >= 6, nt5 < 35]
    print(f"\n  criteria [CI>0, >baseline, >randexit, >randpool, fp>=6, "
          f"not-tail]: {crit}", flush=True)
    print(f"  {'ALL PASS — executable candidate' if all(crit) else 'FAIL — document & close'}",
          flush=True)
    pd.DataFrame([dict(nested=nshe, ci_lo=lo, ci_hi=hi, fp=nfp, top5pct=nt5,
                       ceiling=sh_b, fixed=sh_fx, randexit_p95=pe95,
                       randpool_p95=pp95, all_pass=all(crit))]).to_csv(
        OUT / "verdict.csv", index=False)
    print(f"\nTotal: {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
