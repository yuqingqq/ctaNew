"""L2 book-fragility as a SHORT-LEG SQUEEZE veto (convexity v4, limitation #4).

RISK/tail question (NOT mean-alpha): does L2 book-fragility predict which SHORT
legs will SQUEEZE (blow up), so a veto/downsize overlay cuts the short-leg tail
without killing its edge? A prior agent already showed L2 fragility adds NO mean
short-selection edge, so this is ONLY about tail/variance reduction.

Book: convexity v4 = 1-long / 2-short per 4h bar. SHORT = the 2 lowest-pred names.
A short's PnL = -alpha_A (loses when alpha_A>0 => name went UP vs BTC = squeeze).
The "squeeze tail" = the large-positive-alpha_A shorts.

Fragility features (data/ml/cache/l2_<SYM>.parquet), directions pre-committed from
bookdepth_loader.py economic definitions (HIGH signed value = MORE squeeze-prone):
  l2_asym1  = log(bidN1/askN1)             HIGH = thin asks  -> +1  (limitation #4)
  l2_imbstd = std(imb1) within bar         HIGH = unstable   -> +1
  l2_slope  = (bidN5+askN5)/(bidN1+askN1)  HIGH = thin near  -> +1
  l2_liq1   = log(bidN1+askN1)             LOW  = thin book  -> -1
  l2_touch  = (bidN02+askN02)/(bidN1+askN1) LOW = thin touch -> -1

PIT: L2 obs_bar + 4h == decision open_time (bar known AT the decision).
Book-level, path-independent, BOTH eras, threshold-swept. Run:
    python3 -m live.l2_squeeze_veto
"""
import os
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ERAS = {
    "RECENT": "live/state/convexity/hl_tgt_res_base_honest/v0full_hl60.parquet",
    "OOS":    "live/state/convexity/hl_v4base_oos_honest/v0full_hl60.parquet",
}
FEATS = ["l2_asym1", "l2_imbstd", "l2_slope", "l2_liq1", "l2_touch"]
FRAG_SIGN = {"l2_asym1": +1, "l2_imbstd": +1, "l2_slope": +1, "l2_liq1": -1, "l2_touch": -1}
KS = [0.10, 0.20, 0.30]
ANN = np.sqrt(365.0)
RNG = np.random.default_rng(0)


# ----------------------------------------------------------------------------- data
def load_l2_panel(symbols):
    """Concat all l2_<SYM> caches into one long panel keyed by (symbol, decision_time)
    where decision_time = obs_bar + 4h == the book open_time it is known at (PIT)."""
    frames = []
    for s in symbols:
        f = f"data/ml/cache/l2_{s}.parquet"
        if not os.path.exists(f):
            continue
        d = pd.read_parquet(f)[FEATS].copy()
        d["symbol"] = s
        d["open_time"] = (d.index + pd.Timedelta("4h")).tz_convert("UTC")
        frames.append(d.reset_index(drop=True))
    L2 = pd.concat(frames, ignore_index=True)
    L2["open_time"] = L2["open_time"].astype("datetime64[ns, UTC]")
    return L2


def build_era(path, L2):
    bk = pd.read_parquet(path)
    bk["open_time"] = bk["open_time"].astype("datetime64[ns, UTC]")
    bk["alpha_A"] = bk["alpha_A"].astype("float64")
    m = bk.merge(L2, on=["symbol", "open_time"], how="left")
    # signed fragility (HIGH = squeeze-prone) + per-bar cross-sectional percentile
    # among the tradeable UNIVERSE that bar (PIT, path-independent, no threshold look-ahead)
    for f in FEATS:
        sg = FRAG_SIGN[f] * m[f]
        m[f + "_sg"] = sg
        m[f + "_pct"] = m.groupby("open_time")[f + "_sg"].rank(pct=True)
    # within-bar pred rank: 0,1 = shorts (2 lowest pred); 2 = next-best short (replace target)
    m = m.sort_values(["open_time", "pred"], kind="mergesort")
    m["rk"] = m.groupby("open_time").cumcount()
    return m


# ----------------------------------------------------------------------------- test 1
def test1_ic_tail(m, era):
    print(f"\n=== TEST 1  IC + right-tail (squeeze) among SHORT legs — {era} ===")
    sh = m[m.rk <= 1].copy()  # the two short legs
    sh["pnl"] = -sh["alpha_A"]
    print(f"short legs: {len(sh)}  bars: {sh.open_time.nunique()}  "
          f"dates {sh.open_time.min().date()}..{sh.open_time.max().date()}")
    print(f"  short PnL(-alpha_A) mean {sh.pnl.mean():+.5f}  median {sh.pnl.median():+.5f}  "
          f"(mean<median => left tail drag) | alpha_A p95 {sh.alpha_A.quantile(.95):.4f} "
          f"max {sh.alpha_A.max():.3f}")
    rows = []
    for f in FEATS:
        d = sh[[f + "_sg", "alpha_A", "open_time"]].dropna().reset_index(drop=True)
        if len(d) < 200:
            print(f"  {f:>9}  SKIP (only {len(d)} valid short legs in this era)")
            continue
        xv, yv = d[f + "_sg"].values, d["alpha_A"].values
        ic, _ = spearmanr(xv, yv)
        # day-clustered bootstrap CI on pooled IC (resample whole days)
        day_groups = list(d.groupby(d["open_time"].dt.floor("1D")).indices.values())
        bs = []
        for _ in range(400):
            pick = [day_groups[i] for i in RNG.integers(0, len(day_groups), len(day_groups))]
            idx = np.concatenate(pick)
            bs.append(spearmanr(xv[idx], yv[idx])[0])
        lo, hi = np.percentile(bs, [2.5, 97.5])
        # right tail: top-30% fragile vs rest -> fatter positive-alpha_A tail?
        pc = sh[f + "_pct"]
        hi_m = sh.loc[pc >= 0.70, "alpha_A"]
        lo_m = sh.loc[pc < 0.70, "alpha_A"]
        rows.append((f, ic, lo, hi, hi_m.quantile(.90), lo_m.quantile(.90),
                     hi_m.quantile(.95), lo_m.quantile(.95), hi_m.max(), lo_m.max(),
                     hi_m.mean(), lo_m.mean()))
    hdr = ("feat", "poolIC", "ic_lo", "ic_hi", "hiF_p90", "loF_p90",
           "hiF_p95", "loF_p95", "hiF_max", "loF_max", "hiF_mean", "loF_mean")
    print("  " + "".join(f"{h:>9}" for h in hdr))
    for r in rows:
        print("  " + f"{r[0]:>9}" + "".join(f"{x:>9.4f}" for x in r[1:]))
    print("  [right-tail = top-30%-fragile (hiF) vs rest (loF); squeeze => hiF tail should be FATTER/higher]")


# ----------------------------------------------------------------------------- test 2
def bar_frame(m):
    """One row per bar: the two short legs' pnl + fragility pct (per feature) + next-best short pnl."""
    top3 = m[m.rk <= 2].copy()
    top3["pnl"] = -top3["alpha_A"]
    wide = top3.pivot_table(index="open_time", columns="rk",
                            values=["pnl"] + [f + "_pct" for f in FEATS],
                            aggfunc="first", dropna=False)
    wide.columns = [f"{a}_{b}" for a, b in wide.columns]
    wide = wide.dropna(subset=["pnl_0", "pnl_1"])  # need both shorts
    wide["day"] = wide.index.floor("1D")
    return wide


def feat_subset(wide, feat):
    """Bars where BOTH short legs have a valid fragility percentile for `feat`
    (the L2-covered bars). Baseline & overlay are both evaluated here => paired &
    apples-to-apples, not diluted by bars the overlay cannot act on (OOS is ~28% L2-covered)."""
    return wide[wide[f"{feat}_pct_0"].notna() & wide[f"{feat}_pct_1"].notna()]


def daily_series(wide, feat, K, mode):
    """Fixed capital PER BAR (1 unit split across surviving short legs). Returns a daily
    short-PnL series (sum of bar short-returns per day). Baseline/overlay share the same
    bars => paired & exposure-fair (dropping a fragile leg concentrates into the other,
    it does NOT just reduce gross exposure)."""
    p0, p1, p2 = wide["pnl_0"].values, wide["pnl_1"].values, wide["pnl_2"].values
    if mode == "base":
        ret = 0.5 * p0 + 0.5 * p1
    else:
        v0 = wide[f"{feat}_pct_0"].values >= (1 - K)
        v1 = wide[f"{feat}_pct_1"].values >= (1 - K)
        if mode == "drop":            # veto fragile legs; split 1 unit across survivors
            w0 = np.where(v0, 0.0, 1.0)
            w1 = np.where(v1, 0.0, 1.0)
            tot = w0 + w1
            ret = np.where(tot > 0, (w0 * p0 + w1 * p1) / np.where(tot > 0, tot, 1), 0.0)
        elif mode == "halve":         # tilt weight of fragile legs to 0.5, renormalize
            w0 = np.where(v0, 0.5, 1.0)
            w1 = np.where(v1, 0.5, 1.0)
            ret = (w0 * p0 + w1 * p1) / (w0 + w1)
        elif mode == "replace":       # swap fragile leg for next-best-pred short (constant 2 legs)
            a = np.where(v0, p2, p0)
            b = np.where(v1, p2, p1)
            ret = 0.5 * a + 0.5 * b
    s = pd.Series(ret, index=wide.index)
    return s.groupby(wide["day"]).sum()


def stats(daily):
    mu, sd = daily.mean(), daily.std()
    return dict(sharpe=(mu / sd * ANN) if sd > 0 else np.nan, mean=mu, std=sd,
                p5=daily.quantile(.05), p1=daily.quantile(.01), worst=daily.min())


def veto_frac(wide, feat, K):
    v0 = wide[f"{feat}_pct_0"].values >= (1 - K)
    v1 = wide[f"{feat}_pct_1"].values >= (1 - K)
    return (v0.sum() + v1.sum()) / (2 * len(wide))


def test2_veto(m, era):
    print(f"\n=== TEST 2  veto/downsize overlay — short-leg daily Sharpe & tail — {era} ===")
    wide = bar_frame(m)
    gbase = stats(daily_series(wide, None, None, "base"))
    print(f"GLOBAL baseline (all {len(wide)} bars): Sharpe {gbase['sharpe']:+.2f}  mean {gbase['mean']:+.5f}"
          f"  p5 {gbase['p5']:+.4f}  p1 {gbase['p1']:+.4f}  worst {gbase['worst']:+.4f}")
    print(f"  [dSharpe/dtail below are vs the FEATURE-MATCHED baseline on that feature's L2-covered bars]")
    print(f"  {'feat':>9} {'K':>4} {'mode':>7} {'vfrac':>6} {'Sharpe':>7} {'dShrp':>7} "
          f"{'mean':>8} {'p5':>8} {'dp5':>7} {'p1':>8} {'dp1':>7} {'worst':>8} {'dworst':>7}")
    results, base_by_feat, sub_by_feat = {}, {}, {}
    for f in FEATS:
        w = feat_subset(wide, f)
        if len(w) < 100:
            print(f"  {f:>9}  SKIP (only {len(w)} L2-covered bars in this era)")
            continue
        base = stats(daily_series(w, None, None, "base"))
        base_by_feat[f], sub_by_feat[f] = base, w
        span = f"{w.index.min().date()}..{w.index.max().date()}"
        print(f"  {f} matched-baseline: {len(w)} bars ({span})  Sharpe {base['sharpe']:+.2f}"
              f"  p5 {base['p5']:+.4f}  p1 {base['p1']:+.4f}  worst {base['worst']:+.4f}")
        for K in KS:
            vf = veto_frac(w, f, K)
            for mode in ("drop", "halve", "replace"):
                st = stats(daily_series(w, f, K, mode))
                results[(f, K, mode)] = st
                print(f"  {f:>9} {K:>4.2f} {mode:>7} {vf:>6.3f} "
                      f"{st['sharpe']:>+7.2f} {st['sharpe']-base['sharpe']:>+7.2f} "
                      f"{st['mean']:>+8.5f} {st['p5']:>+8.4f} {st['p5']-base['p5']:>+7.4f} "
                      f"{st['p1']:>+8.4f} {st['p1']-base['p1']:>+7.4f} "
                      f"{st['worst']:>+8.4f} {st['worst']-base['worst']:>+7.4f}")
    return wide, base_by_feat, sub_by_feat, results


# ----------------------------------------------------------------------------- test 3
def boot_sharpe_delta(wide, feat, K, mode):
    """Day-clustered bootstrap of (overlay - base) daily Sharpe delta."""
    b = daily_series(wide, None, None, "base")
    o = daily_series(wide, feat, K, mode)
    df = pd.DataFrame({"b": b, "o": o}).dropna()
    idx = np.arange(len(df))
    deltas = []
    for _ in range(1000):
        s = RNG.choice(idx, len(idx), replace=True)
        bb, oo = df["b"].values[s], df["o"].values[s]
        sb = bb.mean() / bb.std() * ANN if bb.std() > 0 else np.nan
        so = oo.mean() / oo.std() * ANN if oo.std() > 0 else np.nan
        deltas.append(so - sb)
    return np.percentile(deltas, [2.5, 50, 97.5])


def main():
    print("L2 BOOK-FRAGILITY -> SHORT-LEG SQUEEZE VETO  (convexity v4, limitation #4)")
    print("fragility dirs (HIGH=squeeze-prone):", FRAG_SIGN)
    # union of symbols across both eras
    syms = set()
    for p in ERAS.values():
        syms |= set(pd.read_parquet(p, columns=["symbol"])["symbol"].unique())
    L2 = load_l2_panel(sorted(syms))
    print(f"L2 panel: {len(L2):,} rows, {L2.symbol.nunique()} symbols, "
          f"{L2.open_time.min().date()}..{L2.open_time.max().date()}")

    store = {}
    for era, path in ERAS.items():
        m = build_era(path, L2)
        test1_ic_tail(m, era)
        wide, base_by_feat, sub_by_feat, results = test2_veto(m, era)
        store[era] = (base_by_feat, sub_by_feat, results)

    # ---- test 3: adversarial both-era robustness -----------------------------
    print("\n=== TEST 3  adversarial: is any overlay BOTH-era, robust across K & feature? ===")
    er = list(ERAS)
    # a cell "helps" if it improves Sharpe AND cuts the worst-day vs its feature-matched baseline
    def helps(era, k):
        base_by_feat, _, results = store[era]
        st, base = results[k], base_by_feat[k[0]]
        return (st["sharpe"] > base["sharpe"] + 0.05) and (st["worst"] > base["worst"] + 1e-6)
    common = set(store[er[0]][2]) & set(store[er[1]][2])  # cells testable in BOTH eras
    print(f"  cells testable in both eras (feature has L2 both): {len(common)} "
          f"(l2_touch is RECENT-only -> excluded)")
    good = {era: {k for k in common if helps(era, k)} for era in er}
    for era in er:
        print(f"  {era}: {len(good[era])}/{len(common)} cells improve Sharpe(+>.05) AND cut worst-day")
    both = good[er[0]] & good[er[1]]
    print(f"  BOTH-era cells (Sharpe up & worst-day cut in both): {len(both)}")
    for k in sorted(both):
        print("     ", k)
    # also: cells that merely CUT the worst-day in both (ignore Sharpe) — pure tail question
    tail_both = {k for k in common
                 if all(store[e][2][k]["worst"] > store[e][0][k[0]]["worst"] + 1e-6 for e in er)}
    print(f"  BOTH-era cells that cut worst-day (Sharpe-agnostic): {len(tail_both)}")
    # headline: best both-era min-era Sharpe delta, with day-clustered bootstrap CI each era
    def min_delta(k):
        return min(store[e][2][k]["sharpe"] - store[e][0][k[0]]["sharpe"] for e in er)
    best = max(common, key=min_delta)
    print(f"\n  headline cell (max of min-era Sharpe delta among both-era-testable): {best}")
    for era in er:
        base_by_feat, sub_by_feat, results = store[era]
        st, base, w = results[best], base_by_feat[best[0]], sub_by_feat[best[0]]
        lo, md, hi = boot_sharpe_delta(w, best[0], best[1], best[2])
        print(f"   {era}: base Sh {base['sharpe']:+.2f} -> {st['sharpe']:+.2f}  "
              f"dSharpe {st['sharpe']-base['sharpe']:+.2f} (boot95 [{lo:+.2f},{hi:+.2f}]) | "
              f"worst {base['worst']:+.4f} -> {st['worst']:+.4f} ({st['worst']-base['worst']:+.4f}) | "
              f"p1 {base['p1']:+.4f} -> {st['p1']:+.4f}")


if __name__ == "__main__":
    main()
