"""X128 — iter-014: STRUCTURAL K x hold/sleeve sweep (discrete architecture, NOT a tuned parameter).

Question: are the INHERITED structural choices K=5 legs/side and HOLD=6 sleeves (24h hold) robust-
optimal on this system, or does a different DISCRETE config improve Sharpe/Calmar robustly across
universes (HL70 prod + EXT 2021-26 + S44)? The sister vBTC line found K=3 beat K=4 (+0.82 Sharpe,
p100 matched placebo) — K matters and 5 may be suboptimal here.

K and hold are DISCRETE structural choices. Per the contract, G3 (nested-OOS of a tuned parameter) is
WAIVABLE for a discrete arch choice IF it is not picked to maximize in-sample Sharpe — but we MUST
still show robustness via:
  - G7 cross-universe consistency (one K + one hold ranks well on HL70 AND EXT AND S44; the trap is
    picking the HL70-best K that is universe-overfit, cf the 51-panel K=4 vs K=3 lesson),
  - G5 fold-robustness + a nested-OOS-style "choose-K-on-past-folds, apply-forward" check (does the
    chosen K generalize or churn fold-to-fold),
  - G6 paired block-bootstrap CI of the best candidate vs the current K=5.

This evaluates the ALPHA CHAMPION's structure CLEAN — base regime-hybrid held book, NO reactive
overlay (iter-012 stop) — so the structural effect is isolated.

Reuses X123 verbatim for: load_close, betas/mom/alt-index/regime construction, metrics helpers. The
ONLY things parameterized here are K (tail-selection width + beta-neutral sizing) and HOLD (sleeve
count in the held-book averaging). Everything else = baseline. Modifies NOTHING prior.

PIT: identical to X123 — features trailing/shifted, regime from BTC-30d lagged, label by return_pct
at the cycle (already 4h-fwd in the cached preds). No look-ahead introduced by varying K/HOLD.
"""
from __future__ import annotations
import time
from pathlib import Path
import numpy as np
import pandas as pd
import importlib.util as _ilu

REPO = Path("/home/yuqing/ctaNew")
SCRIPTS = REPO/"research/convexity_portable_2026-05-20/scripts"
OUT = REPO/"research/convexity_portable_2026-05-20/results"

_spec = _ilu.spec_from_file_location("x123", SCRIPTS/"X123_altbear_short_probe.py")
x123 = _ilu.module_from_spec(_spec); _spec.loader.exec_module(x123)
load_close = x123.load_close
HL70_PREDS, EXT_PREDS, S44_PREDS = x123.HL70_PREDS, x123.EXT_PREDS, x123.S44_PREDS
WIN = x123.WIN

ANN = 6*365
COSTS_BPS = [1.0, 3.0, 4.5]
PRIMARY_COST = 4.5e-4
SEED = 12345
N_BOOT = 2000

K_GRID = [2, 3, 4, 5, 6, 7]
HOLD_GRID = [3, 6, 9, 12]
K_BASE, HOLD_BASE = 5, 6


def metrics(pnl_bps):
    pb = np.asarray(pnl_bps, dtype=np.float64); pb = pb[np.isfinite(pb)]
    if len(pb) < 3:
        return dict(n=len(pb), tot=np.nan, maxDD=np.nan, Sharpe=np.nan, Calmar=np.nan, pctpos=np.nan)
    eq = np.cumsum(pb); dd = eq - np.maximum.accumulate(eq); mdd = float(dd.min())
    sd = pb.std()
    sh = float(pb.mean()/sd*np.sqrt(ANN)) if sd > 0 else np.nan
    cal = float(pb.mean()*ANN/abs(mdd)) if (mdd < 0 and np.isfinite(mdd)) else np.nan
    return dict(n=len(pb), tot=float(eq[-1]), maxDD=mdd, Sharpe=sh, Calmar=cal, pctpos=float((pb > 0).mean()*100))


# ---------------------------------------------------------------- build the cross-section ONCE per universe
def build_panel(preds_path, label):
    """Build the per-cycle ranking inputs (key-sorted symbols, betas, returns, regime) ONCE; K is
    applied later at book-construction time so we don't reload klines per K. Returns the raw per-cycle
    sorted frames so build_cyc_w(K) is cheap."""
    print(f"\n--- building {label} ({preds_path.name}) ---", flush=True)
    cols = ["symbol", "open_time", "pred", "return_pct", "fold"]
    d = pd.read_parquet(preds_path, columns=cols)
    d["open_time"] = pd.to_datetime(d["open_time"], utc=True)
    d = d[(d["open_time"].dt.hour % 4 == 0) & (d["open_time"].dt.minute == 0)].copy()

    btc = load_close("BTCUSDT"); b4 = btc[(btc.index.hour % 4 == 0) & (btc.index.minute == 0)]
    br = np.log(b4/b4.shift(1)); bvar = br.rolling(WIN, min_periods=42).var()
    syms = sorted(d["symbol"].unique())
    mom_rows = []; beta_map = {}
    for sym in syms:
        c = load_close(sym)
        if c is None: continue
        c4 = c[(c.index.hour % 4 == 0) & (c.index.minute == 0)]
        mom_rows.append(pd.DataFrame({"symbol": sym, "open_time": c4.index,
                                      "mom30": (c4/c4.shift(WIN)-1).shift(1).values}))
        r = np.log(c4/c4.shift(1)); ri, bi = r.align(br, join="inner")
        beta_map[sym] = (ri.rolling(WIN, min_periods=42).cov(bi)/bvar.reindex(ri.index).replace(0, np.nan)).shift(1)
    mom = pd.concat(mom_rows, ignore_index=True); mom["open_time"] = pd.to_datetime(mom["open_time"], utc=True)
    betas = pd.concat([s.rename(k) for k, s in beta_map.items()], axis=1)
    d = d.merge(mom, on=["symbol", "open_time"], how="left")
    btc30 = (b4/b4.shift(WIN)-1).to_frame("b30").reset_index(); btc30["open_time"] = pd.to_datetime(btc30["open_time"], utc=True)
    d = d.merge(btc30, on="open_time", how="left").dropna(subset=["b30"])
    d["regime"] = np.where(d["b30"] > 0.10, "bull", np.where(d["b30"] < -0.10, "bear", "side"))

    times = sorted(d["open_time"].unique()); by_t = {ot: g for ot, g in d.groupby("open_time")}
    fold_by_time = {ot: int(g["fold"].iloc[0]) for ot, g in by_t.items() if "fold" in g.columns}

    # Pre-extract per-cycle: regime, the key-sorted symbol list (full), returns dict, beta row.
    cyc_meta = []; rs = []; regimes = []
    for ot in times:
        g = by_t[ot]; rg = g["regime"].iloc[0]; regimes.append(rg)
        rl = dict(zip(g["symbol"], g["return_pct"])); rs.append(rl)
        key = "mom30" if rg == "bull" else "pred"
        gg = g.dropna(subset=[key]).sort_values(key)
        sorted_syms = gg["symbol"].tolist()                  # ascending by key
        brow = betas.loc[ot].to_dict() if ot in betas.index else {}
        cyc_meta.append(dict(rg=rg, sorted_syms=sorted_syms, brow=brow))
    print(f"  {len(syms)} syms, {len(times)} cycles, {pd.Timestamp(times[0]).date()}->{pd.Timestamp(times[-1]).date()}",
          flush=True)
    return dict(times=list(times), cyc_meta=cyc_meta, rs=rs, regimes=regimes, fold_by_time=fold_by_time,
                n_sym=len(syms))


def build_cyc_w(panel, K):
    """Construct the per-cycle base book at a given K (verbatim X123 base logic, K parameterized).
    bear -> {}; else long top-K / short bottom-K by key, beta-neutral leg sizing in side."""
    cyc_w = []
    for meta in panel["cyc_meta"]:
        rg = meta["rg"]; ss = meta["sorted_syms"]; brow = meta["brow"]
        if rg == "bear" or len(ss) < 2*K:
            cyc_w.append({}); continue
        L = ss[-K:]; S = ss[:K]
        a = b = 1.0
        if rg == "side":
            mbL = np.nanmean([brow.get(s, np.nan) for s in L]); mbS = np.nanmean([brow.get(s, np.nan) for s in S])
            if np.isfinite(mbL) and np.isfinite(mbS) and mbL > 0 and mbS > 0:
                a = 2*mbS/(mbL+mbS); b = 2*mbL/(mbL+mbS)
        w = {}
        for s in L: w[s] = w.get(s, 0.0)+a/K
        for s in S: w[s] = w.get(s, 0.0)-b/K
        cyc_w.append(w)
    return cyc_w


def heldbook(cyc_w, rs, cost, HOLD):
    """X123 held-book engine, HOLD parameterized. Equal-weight overlapping sleeves."""
    prev = {}; pnl = np.empty(len(cyc_w), dtype=np.float64)
    for t in range(len(cyc_w)):
        active = cyc_w[max(0, t-HOLD+1):t+1]
        net = {}
        for w in active:
            for s, wt in w.items(): net[s] = net.get(s, 0.0)+wt/HOLD
        alls = set(net) | set(prev)
        turn = sum(abs(net.get(s, 0.0)-prev.get(s, 0.0)) for s in alls)
        rl = rs[t]
        c = sum(net.get(s, 0.0)*rl.get(s, 0.0) for s in net if np.isfinite(rl.get(s, 0.0)))
        if not np.isfinite(c): c = 0.0
        pnl[t] = c - turn*0.5*cost
        prev = net
    return pnl


def pnl_for(panel, K, HOLD, cost):
    return heldbook(build_cyc_w(panel, K), panel["rs"], cost, HOLD)*1e4


# ---------------------------------------------------------------- block bootstrap paired CI
def paired_ci(diff, fold_arr, n_boot=N_BOOT, rng=None):
    """Block-bootstrap by fold the mean paired per-cycle diff. Returns (mean, lo, hi)."""
    rng = rng or np.random.default_rng(SEED)
    folds = [f for f in pd.unique(fold_arr) if f >= 0]
    blocks = [diff[fold_arr == f] for f in folds]
    blocks = [b for b in blocks if len(b) > 0]
    means = np.empty(n_boot)
    for i in range(n_boot):
        pick = rng.integers(0, len(blocks), size=len(blocks))
        means[i] = np.concatenate([blocks[j] for j in pick]).mean()
    return float(diff.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main():
    t0 = time.time()
    print("="*120, flush=True)
    print("X128 — iter-014 STRUCTURAL K x hold/sleeve SWEEP (alpha champion, NO reactive overlay)", flush=True)
    print(f"  K in {K_GRID}  x  HOLD/sleeves in {HOLD_GRID}  | baseline K={K_BASE} HOLD={HOLD_BASE}", flush=True)
    print(f"  universes HL70(prod)+EXT(2021-26)+S44 | costs {COSTS_BPS}bps | clean structure (no stop)", flush=True)
    print("="*120, flush=True)

    panels = {}
    for nm, pp in (("HL70", HL70_PREDS), ("EXT", EXT_PREDS), ("S44", S44_PREDS)):
        panels[nm] = build_panel(pp, nm)

    # ============================================================ STEP 2: full K x HOLD grid per universe @4.5bps
    print("\n" + "="*120, flush=True)
    print("STEP 2 — K x HOLD GRID per universe @4.5bps (Sharpe / maxDD / Calmar / totPnL / %pos)", flush=True)
    print("="*120, flush=True)
    grid_rows = []
    base_metrics = {}
    for nm in ("HL70", "EXT", "S44"):
        panel = panels[nm]
        fold_arr = np.array([panel["fold_by_time"].get(t, -1) for t in panel["times"]])
        print(f"\n--- {nm} ({panel['n_sym']} syms) ---", flush=True)
        print(f"{'K':>3}{'HOLD':>6}{'Sharpe':>8}{'maxDD':>9}{'Calmar':>8}{'totPnL':>9}{'%pos':>7}  flag", flush=True)
        for K in K_GRID:
            for HOLD in HOLD_GRID:
                pnl = pnl_for(panel, K, HOLD, PRIMARY_COST)
                m = metrics(pnl)
                is_base = (K == K_BASE and HOLD == HOLD_BASE)
                if is_base:
                    base_metrics[nm] = m
                tag = "  <- BASELINE" if is_base else ""
                print(f"{K:>3}{HOLD:>6}{m['Sharpe']:>+8.2f}{m['maxDD']:>+9.0f}{m['Calmar']:>+8.2f}"
                      f"{m['tot']:>+9.0f}{m['pctpos']:>7.1f}{tag}", flush=True)
                grid_rows.append(dict(universe=nm, K=K, HOLD=HOLD, **{k: m[k] for k in
                                  ("Sharpe", "maxDD", "Calmar", "tot", "pctpos")}))
    grid = pd.DataFrame(grid_rows)
    grid.to_parquet(OUT/"X128_K_hold_grid.parquet", index=False)

    # cost sensitivity at the baseline hold for each K (G8) -------------------------------------------
    print("\n" + "-"*120, flush=True)
    print("G8 — cost sensitivity: Sharpe by K at HOLD=6, costs {1,3,4.5}bps per universe", flush=True)
    print("-"*120, flush=True)
    for nm in ("HL70", "EXT", "S44"):
        panel = panels[nm]
        print(f"  {nm}:", flush=True)
        for cb in COSTS_BPS:
            row = "    @%.1fbps  " % cb + "  ".join(
                f"K{K}:{metrics(pnl_for(panel, K, HOLD_BASE, cb*1e-4))['Sharpe']:+.2f}" for K in K_GRID)
            print(row, flush=True)

    # ============================================================ STEP 3a: cross-universe consistency (G7)
    print("\n" + "="*120, flush=True)
    print("STEP 3a (G7) — CROSS-UNIVERSE CONSISTENCY: rank each (K,HOLD) within each universe by Sharpe &", flush=True)
    print("  Calmar; a robust config ranks well on ALL THREE. Avg-rank across universes (lower=better).", flush=True)
    print("="*120, flush=True)
    # rank within universe (1 = best Sharpe)
    grid["sh_rank"] = grid.groupby("universe")["Sharpe"].rank(ascending=False)
    grid["cal_rank"] = grid.groupby("universe")["Calmar"].rank(ascending=False)
    pivot_sh = grid.pivot_table(index=["K", "HOLD"], columns="universe", values="sh_rank")
    pivot_cal = grid.pivot_table(index=["K", "HOLD"], columns="universe", values="cal_rank")
    agg = pd.DataFrame({
        "avg_sh_rank": pivot_sh.mean(axis=1),
        "worst_sh_rank": pivot_sh.max(axis=1),
        "avg_cal_rank": pivot_cal.mean(axis=1),
        "worst_cal_rank": pivot_cal.max(axis=1),
    })
    agg = agg.join(pivot_sh.rename(columns=lambda c: f"shR_{c}"))
    agg = agg.sort_values("avg_sh_rank")
    print("\n  Top configs by AVG Sharpe-rank across universes (n_configs = %d):" % len(grid["K"].unique() * 1), flush=True)
    print(f"  {'K':>3}{'HOLD':>6}{'avgShR':>8}{'worstShR':>9}{'avgCalR':>9}{'shR_HL70':>9}{'shR_EXT':>9}{'shR_S44':>9}", flush=True)
    for (K, HOLD), r in agg.head(12).iterrows():
        base = "  <- BASELINE" if (K == K_BASE and HOLD == HOLD_BASE) else ""
        print(f"  {K:>3}{HOLD:>6}{r['avg_sh_rank']:>8.1f}{r['worst_sh_rank']:>9.0f}{r['avg_cal_rank']:>9.1f}"
              f"{r['shR_HL70']:>9.0f}{r['shR_EXT']:>9.0f}{r['shR_S44']:>9.0f}{base}", flush=True)
    base_row = agg.loc[(K_BASE, HOLD_BASE)]
    print(f"\n  BASELINE (K=5,HOLD=6): avg Sharpe-rank {base_row['avg_sh_rank']:.1f}, "
          f"worst {base_row['worst_sh_rank']:.0f}, avg Calmar-rank {base_row['avg_cal_rank']:.1f} "
          f"(HL70 {base_row['shR_HL70']:.0f} / EXT {base_row['shR_EXT']:.0f} / S44 {base_row['shR_S44']:.0f})",
          flush=True)

    # also: best-per-universe and whether they agree -------------------------------------------------
    print("\n  Best (K,HOLD) by Sharpe PER universe (do they agree?):", flush=True)
    for nm in ("HL70", "EXT", "S44"):
        sub = grid[grid.universe == nm].sort_values("Sharpe", ascending=False).iloc[0]
        subc = grid[grid.universe == nm].sort_values("Calmar", ascending=False).iloc[0]
        print(f"    {nm}: best-Sharpe K={int(sub.K)} HOLD={int(sub.HOLD)} ({sub.Sharpe:+.2f}); "
              f"best-Calmar K={int(subc.K)} HOLD={int(subc.HOLD)} ({subc.Calmar:+.2f})", flush=True)

    # K-marginal at baseline HOLD: how does Sharpe move with K alone? --------------------------------
    print("\n  K-marginal at HOLD=6 (isolate the K=3-vs-K=5 question, cf vBTC):", flush=True)
    for nm in ("HL70", "EXT", "S44"):
        sub = grid[(grid.universe == nm) & (grid.HOLD == HOLD_BASE)].sort_values("K")
        print(f"    {nm}:  " + "  ".join(f"K{int(r.K)} Sh{r.Sharpe:+.2f}/Cal{r.Calmar:+.2f}"
              for _, r in sub.iterrows()), flush=True)
    print("\n  HOLD-marginal at K=5 (isolate the sleeve-count question):", flush=True)
    for nm in ("HL70", "EXT", "S44"):
        sub = grid[(grid.universe == nm) & (grid.K == K_BASE)].sort_values("HOLD")
        print(f"    {nm}:  " + "  ".join(f"H{int(r.HOLD)} Sh{r.Sharpe:+.2f}/Cal{r.Calmar:+.2f}"
              for _, r in sub.iterrows()), flush=True)

    # ============================================================ STEP 3b: nested-OOS choose-K (and hold) on past folds
    print("\n" + "="*120, flush=True)
    print("STEP 3b (G3/G5) — NESTED-OOS: choose the structural config on PAST folds (max Sharpe), apply", flush=True)
    print("  to the NEXT fold; concat the forward slices. Does the chosen config generalize or churn?", flush=True)
    print("  Tested separately: choose-K-only (HOLD=6 fixed) and choose-(K,HOLD).  HL70 + EXT + S44 @4.5bps", flush=True)
    print("="*120, flush=True)
    # precompute per-(K,HOLD) full pnl per universe to slice by fold cheaply
    full_pnl = {nm: {} for nm in ("HL70", "EXT", "S44")}
    for nm in ("HL70", "EXT", "S44"):
        panel = panels[nm]
        for K in K_GRID:
            for HOLD in HOLD_GRID:
                full_pnl[nm][(K, HOLD)] = pnl_for(panel, K, HOLD, PRIMARY_COST)

    def nested_oos(nm, candidate_configs, fixed_hold=None):
        panel = panels[nm]
        fold_arr = np.array([panel["fold_by_time"].get(t, -1) for t in panel["times"]])
        folds = sorted(f for f in pd.unique(fold_arr) if f >= 0)
        oos_pnl = []; oos_base = []; chosen = []
        for i in range(1, len(folds)):
            past = np.isin(fold_arr, folds[:i]); fut = fold_arr == folds[i]
            best_cfg, best_sh = None, -1e18
            for cfg in candidate_configs:
                sh = metrics(full_pnl[nm][cfg][past])["Sharpe"]
                if np.isfinite(sh) and sh > best_sh:
                    best_sh, best_cfg = sh, cfg
            oos_pnl.append(full_pnl[nm][best_cfg][fut])
            oos_base.append(full_pnl[nm][(K_BASE, HOLD_BASE)][fut])
            chosen.append(best_cfg)
        ob = metrics(np.concatenate(oos_base)); oo = metrics(np.concatenate(oos_pnl))
        return ob, oo, chosen

    k_only = [(K, HOLD_BASE) for K in K_GRID]
    kh_all = [(K, HOLD) for K in K_GRID for HOLD in HOLD_GRID]
    for nm in ("HL70", "EXT", "S44"):
        ob, oo, chosen = nested_oos(nm, k_only)
        print(f"\n  {nm} choose-K-only (HOLD=6): chosen K per fold {[c[0] for c in chosen]}", flush=True)
        print(f"    OOS baseline(K5H6) Sharpe {ob['Sharpe']:+.2f} Calmar {ob['Calmar']:+.2f} maxDD {ob['maxDD']:+.0f}",
              flush=True)
        print(f"    OOS nested-chosen  Sharpe {oo['Sharpe']:+.2f} Calmar {oo['Calmar']:+.2f} maxDD {oo['maxDD']:+.0f}"
              f"  -> Δsh {oo['Sharpe']-ob['Sharpe']:+.2f} Δcal {oo['Calmar']-ob['Calmar']:+.2f} "
              f"{'GENERALIZES' if oo['Calmar'] >= ob['Calmar'] else 'CHURNS/loses'}", flush=True)
        obh, ooh, chosenh = nested_oos(nm, kh_all)
        print(f"  {nm} choose-(K,HOLD): chosen per fold {chosenh}", flush=True)
        print(f"    OOS nested-(K,HOLD) Sharpe {ooh['Sharpe']:+.2f} Calmar {ooh['Calmar']:+.2f} "
              f"-> Δcal vs base {ooh['Calmar']-obh['Calmar']:+.2f} "
              f"{'GENERALIZES' if ooh['Calmar'] >= obh['Calmar'] else 'CHURNS/loses'}", flush=True)

    # ============================================================ STEP 3c: per-fold + paired CI of best candidate vs K=5
    print("\n" + "="*120, flush=True)
    print("STEP 3c (G5/G6) — pick the cross-universe-robust CANDIDATE (best avg Sharpe-rank that is not", flush=True)
    print("  baseline), then per-fold win-count + block-bootstrap paired CI vs K=5,HOLD=6 on each universe.", flush=True)
    print("="*120, flush=True)
    # candidate = best avg_sh_rank config that isn't baseline
    cand = None
    for (K, HOLD), _ in agg.iterrows():
        if not (K == K_BASE and HOLD == HOLD_BASE):
            cand = (int(K), int(HOLD)); break
    print(f"  cross-universe-robust candidate (best non-baseline avg Sharpe-rank): K={cand[0]} HOLD={cand[1]}",
          flush=True)
    rng = np.random.default_rng(SEED)
    for nm in ("HL70", "EXT", "S44"):
        panel = panels[nm]
        fold_arr = np.array([panel["fold_by_time"].get(t, -1) for t in panel["times"]])
        base_pnl = full_pnl[nm][(K_BASE, HOLD_BASE)]
        cand_pnl = full_pnl[nm][cand]
        folds = sorted(f for f in pd.unique(fold_arr) if f >= 0)
        wins = 0; nf = 0
        for f in folds:
            m = fold_arr == f
            if m.sum() < 3: continue
            nf += 1
            if metrics(cand_pnl[m])["Calmar"] >= metrics(base_pnl[m])["Calmar"]:
                wins += 1
        diff = cand_pnl - base_pnl
        mean, lo, hi = paired_ci(diff, fold_arr, rng=rng)
        bm = metrics(base_pnl); cm = metrics(cand_pnl)
        clears = (lo > 0)
        print(f"\n  {nm}: base(K5H6) Sh{bm['Sharpe']:+.2f}/Cal{bm['Calmar']:+.2f}/DD{bm['maxDD']:+.0f}  ->  "
              f"cand(K{cand[0]}H{cand[1]}) Sh{cm['Sharpe']:+.2f}/Cal{cm['Calmar']:+.2f}/DD{cm['maxDD']:+.0f}",
              flush=True)
        print(f"    G5 Calmar-wins {wins}/{nf} folds; G6 paired diff mean {mean:+.2f}bps/cyc CI[{lo:+.2f},{hi:+.2f}] "
              f"{'clears 0' if clears else 'CROSSES 0'}", flush=True)

    print(f"\nartifacts: X128_K_hold_grid.parquet", flush=True)
    print(f"Done [{time.time()-t0:.0f}s]", flush=True)


if __name__ == "__main__":
    main()
