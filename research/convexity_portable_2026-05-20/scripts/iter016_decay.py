"""iter-016 — SIGNAL DECAY + HETEROGENEITY diagnostic (event-driven / variable-horizon pre-check).

Question: does the 4h-trained mean-rev `pred` signal have edge BEYOND 4h (justifying the 24h
held-book hold), or does it decay fast (=> sleeves 2-6 hold STALE signal, and an event-driven
exit-on-decay could trade less while preserving edge)? And critically for VARIABLE horizon:
is the decay rate HOMOGENEOUS (fixed hold already optimal) or HETEROGENEOUS by something
observable-at-entry (=> variable hold has real potential)?

Measured (computed, not speculated) on HL70 (prod) + EXT (2021-26), in SIDE regime (where the
mean-rev pred operates) and bull/all for reference:

  1. SIGNAL DECAY CURVE: cross-sectional IC of pred vs FORWARD return at h in
     {4,8,12,16,24,36,48,72}h. Where does IC peak / cross zero (signal life)?
  2. HETEROGENEITY: does the decay rate vary by |pred| strength, by symbol vol, by regime?
     (the decisive variable-horizon test)
  3. EVENT-DRIVEN ENTRY value: IC/edge per unit turnover when trading only strong per-symbol signal.
  4. TURNOVER framing: would a decay-based variable hold trade LESS than the 24h sleeve book?

All forward returns computed from klines at each horizon. PIT for trading decision; forward
returns are the LABEL we are measuring the signal against (this is signal characterization, not
a trade with look-ahead).
"""
from __future__ import annotations
import time
from pathlib import Path
import pandas as pd, numpy as np

REPO = Path("/home/yuqing/ctaNew")
RC = REPO/"research/convexity_portable_2026-05-20/results/_cache"
OUT = REPO/"research/convexity_portable_2026-05-20/results"
KLINES = REPO/"data/ml/test/parquet/klines"

HORIZONS_H = [4, 8, 12, 16, 24, 36, 48, 72]   # hours
BARS_PER_4H = 48   # 5m bars in 4h
# in 4h-grid terms (each step = 4h): horizon in 4h-steps
H_STEPS = {h: h//4 for h in HORIZONS_H}

PANELS = {
    "HL70": RC/"x64_HL70_V5mv3_7cx_filter_t0.3_preds.parquet",
    "EXT":  RC/"x113_ext_v0_preds.parquet",
}


def load_close(sym):
    sd = KLINES/sym/"5m"
    if not sd.exists():
        return None
    dfs = [pd.read_parquet(f, columns=["open_time", "close"]) for f in sorted(sd.glob("*.parquet"))]
    df = pd.concat(dfs, ignore_index=True).drop_duplicates("open_time").sort_values("open_time")
    df["open_time"] = pd.to_datetime(df["open_time"], utc=True)
    return df.set_index("open_time")["close"].astype(np.float64)


def build(panel_path):
    d = pd.read_parquet(panel_path, columns=["symbol", "open_time", "pred", "alpha_A", "return_pct", "fold"])
    d["open_time"] = pd.to_datetime(d["open_time"], utc=True)
    # 4h grid entries
    d = d[(d["open_time"].dt.hour % 4 == 0) & (d["open_time"].dt.minute == 0)].copy()
    d = d.sort_values(["symbol", "open_time"])

    syms = sorted(d["symbol"].unique())
    # 4h-grid close panel + BTC + per-symbol fwd returns at each horizon + trailing vol
    closes = {}
    for sym in syms + (["BTCUSDT"] if "BTCUSDT" not in syms else []):
        c = load_close(sym)
        if c is None:
            continue
        c4 = c[(c.index.hour % 4 == 0) & (c.index.minute == 0)]
        closes[sym] = c4
    cl = pd.concat([s.rename(k) for k, s in closes.items()], axis=1).sort_index()

    # forward simple returns at each horizon (in 4h-steps)
    fwd = {}
    for h, st in H_STEPS.items():
        fwd[h] = (cl.shift(-st) / cl - 1.0)   # fwd return over h hours, index=entry time

    # trailing 7d (42-bar) realized vol of 4h log returns per symbol, shift(1) PIT
    logr = np.log(cl / cl.shift(1))
    tvol = logr.rolling(42, min_periods=20).std().shift(1)

    # BTC 30d regime
    btc = cl["BTCUSDT"]
    btc30 = (btc / btc.shift(180) - 1.0)
    reg = pd.Series(np.where(btc30 > 0.10, "bull", np.where(btc30 < -0.10, "bear", "side")), index=btc.index)

    return d, cl, fwd, tvol, reg, syms


def cs_ic_curve(d, fwd, reg, regime_filter, min_names=8):
    """Cross-sectional spearman IC of pred vs forward return at each horizon, averaged over cycles.
    Returns dict h -> (mean_ic, n_cyc, t_stat) within the regime filter."""
    # attach regime to each row by open_time
    d = d.copy()
    d["regime"] = d["open_time"].map(reg)
    if regime_filter is not None:
        d = d[d["regime"] == regime_filter]
    out = {}
    for h, fwh in fwd.items():
        ics = []
        for ot, g in d.groupby("open_time"):
            if ot not in fwh.index:
                continue
            fr = fwh.loc[ot]
            gg = g.dropna(subset=["pred"]).copy()
            gg["fr"] = gg["symbol"].map(fr)
            gg = gg.dropna(subset=["fr"])
            if len(gg) >= min_names:
                ic = gg["pred"].corr(gg["fr"], method="spearman")
                if np.isfinite(ic):
                    ics.append(ic)
        ics = np.array(ics)
        if len(ics) > 2:
            out[h] = (ics.mean(), len(ics), ics.mean() / (ics.std() / np.sqrt(len(ics))) if ics.std() > 0 else np.nan)
        else:
            out[h] = (np.nan, len(ics), np.nan)
    return out


def cs_ic_curve_subset(d, fwd, reg, regime_filter, row_mask_col, min_names=6):
    """Same as cs_ic_curve but restrict to rows where row_mask_col is True (per-row strong-signal subset)."""
    d = d.copy()
    d["regime"] = d["open_time"].map(reg)
    if regime_filter is not None:
        d = d[d["regime"] == regime_filter]
    d = d[d[row_mask_col]]
    out = {}
    for h, fwh in fwd.items():
        ics = []
        for ot, g in d.groupby("open_time"):
            if ot not in fwh.index:
                continue
            fr = fwh.loc[ot]
            gg = g.dropna(subset=["pred"]).copy()
            gg["fr"] = gg["symbol"].map(fr)
            gg = gg.dropna(subset=["fr"])
            if len(gg) >= min_names:
                ic = gg["pred"].corr(gg["fr"], method="spearman")
                if np.isfinite(ic):
                    ics.append(ic)
        ics = np.array(ics)
        out[h] = (ics.mean() if len(ics) > 2 else np.nan, len(ics))
    return out


def decile_decay(d, fwd, reg, regime_filter):
    """HETEROGENEITY by |pred|: split each cycle into per-cycle |pred| terciles, measure how the
    K=5 long-short spread return evolves with horizon for STRONG vs WEAK signal names.
    We use the top-K/bottom-K LS spread (the actual book selection) and track its forward return
    decay at each horizon, separately conditioned on cross-cycle signal strength (|pred| of the
    selected extremes)."""
    d = d.copy()
    d["regime"] = d["open_time"].map(reg)
    if regime_filter is not None:
        d = d[d["regime"] == regime_filter]
    K = 5
    # for each cycle: long = top-K pred, short = bottom-K pred; spread fwd ret at each horizon.
    # signal-strength of the cycle = mean(|pred| of the 2K selected extremes)
    rows = []
    for ot, g in d.groupby("open_time"):
        gg = g.dropna(subset=["pred"]).sort_values("pred")
        if len(gg) < 2 * K:
            continue
        Lsym = gg.tail(K)["symbol"].tolist()
        Ssym = gg.head(K)["symbol"].tolist()
        strength = np.abs(pd.concat([gg.tail(K)["pred"], gg.head(K)["pred"]])).mean()
        row = {"open_time": ot, "strength": strength}
        for h, fwh in fwd.items():
            if ot not in fwh.index:
                row[f"sp_{h}"] = np.nan
                continue
            fr = fwh.loc[ot]
            lr = fr.reindex(Lsym).mean()
            sr = fr.reindex(Ssym).mean()
            row[f"sp_{h}"] = lr - sr   # long-short spread fwd ret at horizon h
        rows.append(row)
    sp = pd.DataFrame(rows)
    return sp


def main():
    t0 = time.time()
    print("=== iter-016 SIGNAL DECAY + HETEROGENEITY (variable-horizon pre-check) ===\n", flush=True)
    summary = {}
    for pname, ppath in PANELS.items():
        print(f"\n########## PANEL {pname} ##########", flush=True)
        d, cl, fwd, tvol, reg, syms = build(ppath)
        print(f"  {len(syms)} syms, {d['open_time'].nunique()} 4h cycles, "
              f"{reg.value_counts().to_dict()} regime-bar counts", flush=True)

        # ---- 1. DECAY CURVE per regime ----
        for rf in ["side", "bull", None]:
            lab = rf if rf else "all"
            curve = cs_ic_curve(d, fwd, reg, rf)
            print(f"\n  [DECAY] regime={lab}: cross-sectional IC(pred, fwd_ret) by horizon")
            hdr = "    h(h):    " + "  ".join(f"{h:>6}" for h in HORIZONS_H)
            icl = "    IC:      " + "  ".join(f"{curve[h][0]:+.4f}" if np.isfinite(curve[h][0]) else "   nan" for h in HORIZONS_H)
            tl  = "    t:       " + "  ".join(f"{curve[h][2]:+.2f}" if np.isfinite(curve[h][2]) else "  nan" for h in HORIZONS_H)
            print(hdr); print(icl); print(tl)
            summary[f"{pname}_{lab}_decay"] = {h: curve[h][0] for h in HORIZONS_H}

        # ---- 2. HETEROGENEITY by signal strength (SIDE) ----
        print(f"\n  [HETEROGENEITY] SIDE regime — LS-spread fwd-ret decay by cycle signal-strength tercile")
        sp = decile_decay(d, fwd, reg, "side")
        if len(sp) > 30:
            sp["sgrp"] = pd.qcut(sp["strength"], 3, labels=["weak", "mid", "strong"])
            print(f"    n_cyc={len(sp)}")
            print("    grp     " + "  ".join(f"{h:>7}h" for h in HORIZONS_H))
            for grp in ["weak", "mid", "strong"]:
                sub = sp[sp["sgrp"] == grp]
                vals = [sub[f"sp_{h}"].mean() * 1e4 for h in HORIZONS_H]  # bps
                print(f"    {grp:<6}  " + "  ".join(f"{v:+7.1f}" for v in vals))
            # also: per-tercile peak horizon (argmax of spread)
            print("    (values = mean LS-spread forward return in bps at each horizon)")
            for grp in ["weak", "mid", "strong"]:
                sub = sp[sp["sgrp"] == grp]
                vals = np.array([sub[f"sp_{h}"].mean() for h in HORIZONS_H])
                if np.all(np.isnan(vals)):
                    continue
                peak_h = HORIZONS_H[int(np.nanargmax(vals))]
                # zero-cross: first horizon where spread per-step turns negative (marginal)
                print(f"      {grp}: peak cumulative spread at h={peak_h}h")
            summary[f"{pname}_side_strength_decay"] = {
                grp: {h: float(sp[sp['sgrp']==grp][f'sp_{h}'].mean()*1e4) for h in HORIZONS_H}
                for grp in ["weak","mid","strong"]}

        # ---- 2b. MARGINAL per-step spread (is later holding adding edge?) ----
        if len(sp) > 30:
            print(f"\n  [MARGINAL] SIDE — incremental LS-spread return earned in each 4h sub-window (bps)")
            # marginal at step covering (h-4 -> h)
            prev_h = None
            marg = {}
            cum_prev = {grp: 0.0 for grp in ["weak", "mid", "strong"]}
            print("    grp     " + "  ".join(f"{a}->{b}h".rjust(8) for a,b in zip([0]+HORIZONS_H[:-1],HORIZONS_H)))
            for grp in ["weak", "mid", "strong"]:
                sub = sp[sp["sgrp"] == grp]
                cum = [0.0] + [sub[f"sp_{h}"].mean() * 1e4 for h in HORIZONS_H]
                margs = [cum[i+1] - cum[i] for i in range(len(HORIZONS_H))]
                print(f"    {grp:<6}  " + "  ".join(f"{m:+8.1f}" for m in margs))
            print("    (marginal bps added in each window; if it goes ~0/negative, holding longer is stale)")

        # ---- 3. EVENT-DRIVEN ENTRY value: IC at h=4 for strong-|pred| per-symbol rows vs all ----
        print(f"\n  [EVENT-ENTRY] SIDE — IC by horizon, ALL rows vs |pred|>=per-cycle p70 (strong-signal subset)")
        # mark strong rows: within each cycle, |pred| in top 30%
        dd = d.copy()
        dd["regime"] = dd["open_time"].map(reg)
        dd["abspred"] = dd["pred"].abs()
        dd["thr70"] = dd.groupby("open_time")["abspred"].transform(lambda x: x.quantile(0.70))
        dd["strong"] = dd["abspred"] >= dd["thr70"]
        allc = cs_ic_curve(dd, fwd, reg, "side")
        # build subset-aware: reuse cs_ic_curve_subset on dd
        strc = cs_ic_curve_subset(dd, fwd, reg, "side", "strong")
        print("    h(h):    " + "  ".join(f"{h:>6}" for h in HORIZONS_H))
        print("    all:     " + "  ".join(f"{allc[h][0]:+.4f}" if np.isfinite(allc[h][0]) else "   nan" for h in HORIZONS_H))
        print("    strong:  " + "  ".join(f"{strc[h][0]:+.4f}" if np.isfinite(strc[h][0]) else "   nan" for h in HORIZONS_H))

    print(f"\n[{time.time()-t0:.0f}s] done", flush=True)
    import json
    (OUT/"iter016_decay_summary.json").write_text(json.dumps(summary, indent=2, default=float))
    print(f"Saved -> {OUT/'iter016_decay_summary.json'}")


if __name__ == "__main__":
    main()
