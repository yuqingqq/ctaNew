"""X129 — iter-015 (ALPHA track): does a BETTER MOMENTUM formulation have genuine BULL-regime
cross-sectional IC? (the PRE-CHECK that gates any selection-signal change to the bull sleeve.)

Context (iter-004): the bull regime is 100% of HL70 PnL (+6.03 Sharpe) but the cross-sectional
rank-IC of the CURRENT selection signal `mom30` in bull is ~0 (−0.0016, t=−0.17). iter-004 showed
the bull PnL is net-long-BETA capture, not stock selection, and that REPLACING mom30 with `pred`
is catastrophic (the beta engine dies). UNTESTED question: does a DIFFERENT momentum formulation
have real bull XS-IC (vs mom30's ~0) → improvable selection → worth proposing a swap?

This is STEP 2 of the directive: a CHEAP pre-check BEFORE building. For BULL cycles only, compute the
per-cycle cross-sectional rank-corr (Spearman) of each momentum variant vs the forward target, then
average across cycles, with a fold-level breakdown and a t-stat (cycle-level). Two targets:
  - return_pct  (RAW 4h-fwd return — what the bull long/short book actually monetizes)
  - alpha_A     (4h-fwd alpha-residual ret − β·BTC — the beta-neutral selection skill)

Battery of momentum formulations (all PIT — built from 4h closes, shifted by 1 bar):
  multi-TF       : mom_7d, mom_14d, mom_30d(current), mom_90d, mom_180d
  vol-adjusted   : mom_30d / trailing-vol  (Sharpe-like)      ; mom_30d * inv-vol (Barroso-scaled)
  rank-composite : mean rank across {7,14,30,90}d momenta
  residual-mom   : trailing-30d cumulative alpha-residual (ret − β·BTC), shifted (momentum of the
                   beta-stripped return — the "momentum that isn't just beta")
  trend-quality  : mom_30d * trend_strength, trend_strength = |net move| / sum|bar moves| over 30d
                   (Kaufman efficiency ratio — momentum weighted by low-choppiness)

A variant is "interesting" only if its BULL XS-IC is meaningfully > mom30's ~0 AND consistent across
folds (not one fold). If EVERY variant is ~0 in bull, the bull engine is irreducible beta capture
and a better SELECTION signal cannot help (the iter-004 wall) → NO-CANDIDATE.

Reuses X123 load_close / WIN verbatim. Reads only cached preds + klines. Modifies nothing.
PIT: every momentum variant is .shift(1) on 4h closes (no current-bar). Targets are forward (label).
"""
from __future__ import annotations
import time
from pathlib import Path
import numpy as np
import pandas as pd
import importlib.util as _ilu
from scipy.stats import spearmanr

REPO = Path("/home/yuqing/ctaNew")
SCRIPTS = REPO/"research/convexity_portable_2026-05-20/scripts"
OUT = REPO/"research/convexity_portable_2026-05-20/results"

_spec = _ilu.spec_from_file_location("x123", SCRIPTS/"X123_altbear_short_probe.py")
x123 = _ilu.module_from_spec(_spec); _spec.loader.exec_module(x123)
load_close = x123.load_close
HL70_PREDS, EXT_PREDS, S44_PREDS = x123.HL70_PREDS, x123.EXT_PREDS, x123.S44_PREDS
WIN = x123.WIN  # 180 bars = 30d at 4h

# bar windows at 4h
W7, W14, W30, W90, W180 = 42, 84, 180, 540, 1080
VOLWIN = 180  # trailing vol window for vol-adjusted momentum


def build_signals(preds_path, label):
    """Build a long frame: rows = (symbol, open_time) at 4h cadence, with all momentum variants
    (PIT, shifted), the regime tag (bull/side/bear from BTC-30d lagged), fold, and forward targets
    return_pct + alpha_A. Returns DataFrame restricted to BULL cycles for the IC computation, plus
    the full frame for sanity."""
    print(f"\n--- building signals {label} ({preds_path.name}) ---", flush=True)
    cols = ["symbol", "open_time", "return_pct", "alpha_A", "fold"]
    d = pd.read_parquet(preds_path, columns=cols)
    d["open_time"] = pd.to_datetime(d["open_time"], utc=True)
    d = d[(d["open_time"].dt.hour % 4 == 0) & (d["open_time"].dt.minute == 0)].copy()

    btc = load_close("BTCUSDT"); b4 = btc[(btc.index.hour % 4 == 0) & (btc.index.minute == 0)]
    br = np.log(b4/b4.shift(1)); bvar = br.rolling(WIN, min_periods=42).var()
    syms = sorted(d["symbol"].unique())

    sig_rows = []
    for sym in syms:
        c = load_close(sym)
        if c is None:
            continue
        c4 = c[(c.index.hour % 4 == 0) & (c.index.minute == 0)]
        lr = np.log(c4/c4.shift(1))                         # 4h log returns
        # multi-TF simple momentum (price ratio), shifted 1 bar (PIT)
        m7 = (c4/c4.shift(W7) - 1).shift(1)
        m14 = (c4/c4.shift(W14) - 1).shift(1)
        m30 = (c4/c4.shift(W30) - 1).shift(1)
        m90 = (c4/c4.shift(W90) - 1).shift(1)
        m180 = (c4/c4.shift(W180) - 1).shift(1)
        # trailing realized vol over 30d (PIT)
        vol30 = lr.rolling(VOLWIN, min_periods=42).std().shift(1)
        mom_voladj = (m30 / vol30.replace(0, np.nan))       # Sharpe-like momentum
        mom_barroso = (m30 * (vol30.median(skipna=True) / vol30.replace(0, np.nan)))  # vol-scaled to const target
        # residual momentum: trailing-30d cumulative (ret - beta*BTC), shifted
        ri, bi = lr.align(br, join="inner")
        beta = (ri.rolling(WIN, min_periods=42).cov(bi)/bvar.reindex(ri.index).replace(0, np.nan)).shift(1)
        # residual return per bar = own ret - beta(t)*btc ret ; cumulate trailing 30d
        beta_al = beta.reindex(lr.index)
        br_al = br.reindex(lr.index)
        resid_r = lr - beta_al*br_al
        res_mom30 = resid_r.rolling(W30, min_periods=60).sum().shift(1)
        # trend-quality: Kaufman efficiency ratio over 30d * mom30
        net_move = (c4 - c4.shift(W30)).abs()
        path = c4.diff().abs().rolling(W30, min_periods=60).sum()
        eff = (net_move / path.replace(0, np.nan)).shift(1)   # 0..1, 1=clean trend
        mom_trendq = (m30 * eff)
        sub = pd.DataFrame({
            "symbol": sym, "open_time": c4.index,
            "mom_7d": m7.values, "mom_14d": m14.values, "mom_30d": m30.values,
            "mom_90d": m90.values, "mom_180d": m180.values,
            "mom_voladj": mom_voladj.values, "mom_barroso": mom_barroso.values,
            "res_mom_30d": res_mom30.values, "mom_trendq": mom_trendq.values,
        })
        sig_rows.append(sub)
    sig = pd.concat(sig_rows, ignore_index=True)
    sig["open_time"] = pd.to_datetime(sig["open_time"], utc=True)

    d = d.merge(sig, on=["symbol", "open_time"], how="left")
    # rank-composite momentum: per-cycle mean of cross-sectional ranks across {7,14,30,90}d
    rank_cols = ["mom_7d", "mom_14d", "mom_30d", "mom_90d"]
    rk = d.groupby("open_time")[rank_cols].rank(pct=True)
    d["mom_rankcomp"] = rk.mean(axis=1)

    btc30 = (b4/b4.shift(WIN)-1).to_frame("b30").reset_index()
    btc30["open_time"] = pd.to_datetime(btc30["open_time"], utc=True)
    d = d.merge(btc30, on="open_time", how="left").dropna(subset=["b30"])
    d["regime"] = np.where(d["b30"] > 0.10, "bull", np.where(d["b30"] < -0.10, "bear", "side"))

    n_bull_cyc = d[d.regime == "bull"]["open_time"].nunique()
    print(f"  {len(syms)} syms; bull cycles {n_bull_cyc}; "
          f"{d['open_time'].min().date()}->{d['open_time'].max().date()}", flush=True)
    return d


VARIANTS = ["mom_7d", "mom_14d", "mom_30d", "mom_90d", "mom_180d",
            "mom_voladj", "mom_barroso", "mom_rankcomp", "res_mom_30d", "mom_trendq"]


def xs_ic_table(d, target, regime="bull", min_names=6):
    """Per-cycle cross-sectional Spearman rank-corr of each variant vs `target`, restricted to
    `regime` cycles. Returns dict variant -> (mean_ic, t_stat, n_cyc, fold_means dict)."""
    sub = d[d.regime == regime].copy()
    out = {}
    folds = sorted(f for f in sub["fold"].unique() if f >= 0)
    for v in VARIANTS:
        ics = []
        fold_ics = {f: [] for f in folds}
        for (ot, fold), g in sub.groupby(["open_time", "fold"]):
            gg = g[[v, target]].dropna()
            if len(gg) < min_names or gg[v].nunique() < 3:
                continue
            ic = spearmanr(gg[v], gg[target]).correlation
            if np.isfinite(ic):
                ics.append(ic)
                if fold in fold_ics:
                    fold_ics[fold].append(ic)
        ics = np.array(ics)
        if len(ics) < 3:
            out[v] = (np.nan, np.nan, len(ics), {})
            continue
        mean_ic = ics.mean()
        t = mean_ic / (ics.std(ddof=1)/np.sqrt(len(ics))) if ics.std(ddof=1) > 0 else np.nan
        fm = {f: (np.mean(x) if len(x) else np.nan) for f, x in fold_ics.items()}
        out[v] = (mean_ic, t, len(ics), fm)
    return out


def print_table(label, target, tbl):
    print(f"\n  [BULL XS-IC vs {target} — {label}]  (per-cycle Spearman, mean ± t; current = mom_30d)",
          flush=True)
    print(f"  {'variant':>14}{'mean_IC':>10}{'t':>8}{'n_cyc':>7}  fold-mean ICs", flush=True)
    base_ic = tbl.get("mom_30d", (np.nan,))[0]
    for v in VARIANTS:
        mic, t, n, fm = tbl[v]
        folds = sorted(fm.keys())
        fold_str = " ".join(f"{fm[f]:+.3f}" for f in folds)
        delta = (mic - base_ic) if (np.isfinite(mic) and np.isfinite(base_ic)) else np.nan
        flag = "  <-current" if v == "mom_30d" else (f"  Δ{delta:+.3f}" if np.isfinite(delta) else "")
        print(f"  {v:>14}{mic:>+10.4f}{t:>+8.2f}{n:>7}  [{fold_str}]{flag}", flush=True)


def main():
    t0 = time.time()
    print("="*120, flush=True)
    print("X129 — iter-015 PRE-CHECK: BULL-regime cross-sectional IC of a momentum battery", flush=True)
    print(f"  variants: {VARIANTS}", flush=True)
    print("  targets: return_pct (raw fwd — what bull book monetizes) + alpha_A (beta-resid fwd)", flush=True)
    print("  universes: HL70(prod) + EXT(2021-26) + S44 | bull = BTC-30d>+10% (PIT)", flush=True)
    print("="*120, flush=True)

    summary = {}
    for nm, pp in (("HL70", HL70_PREDS), ("EXT", EXT_PREDS), ("S44", S44_PREDS)):
        d = build_signals(pp, nm)
        tbl_ret = xs_ic_table(d, "return_pct", "bull")
        tbl_alpha = xs_ic_table(d, "alpha_A", "bull")
        print_table(nm, "return_pct", tbl_ret)
        print_table(nm, "alpha_A", tbl_alpha)
        summary[nm] = dict(ret=tbl_ret, alpha=tbl_alpha)
        # also: sideways-regime IC of pred (sanity vs known mean-rev sleeve) and bull pctpos
        rows = []
        for tgt, tbl in (("return_pct", tbl_ret), ("alpha_A", tbl_alpha)):
            for v in VARIANTS:
                mic, t, n, fm = tbl[v]
                rows.append(dict(universe=nm, target=tgt, variant=v, mean_ic=mic, t=t, n_cyc=n))
        pd.DataFrame(rows).to_parquet(OUT/f"X129_bull_xsic_{nm}.parquet", index=False)

    # ----- cross-universe synthesis: which variant (if any) is robustly > mom30 in bull -----
    print("\n" + "="*120, flush=True)
    print("CROSS-UNIVERSE SYNTHESIS — bull XS-IC vs return_pct (the bull book's actual target)", flush=True)
    print("  A variant is a CANDIDATE only if mean_IC meaningfully > mom_30d AND |t|>=2 AND same-signed", flush=True)
    print("  across HL70+EXT+S44 (not one fold/universe).", flush=True)
    print("="*120, flush=True)
    print(f"  {'variant':>14}{'HL70_IC':>10}{'HL70_t':>9}{'EXT_IC':>10}{'EXT_t':>9}{'S44_IC':>10}{'S44_t':>9}",
          flush=True)
    for v in VARIANTS:
        h = summary["HL70"]["ret"][v]; e = summary["EXT"]["ret"][v]; s = summary["S44"]["ret"][v]
        cur = "  <-current" if v == "mom_30d" else ""
        print(f"  {v:>14}{h[0]:>+10.4f}{h[1]:>+9.2f}{e[0]:>+10.4f}{e[1]:>+9.2f}{s[0]:>+10.4f}{s[1]:>+9.2f}{cur}",
              flush=True)
    print("\n  (same table for alpha_A target)", flush=True)
    print(f"  {'variant':>14}{'HL70_IC':>10}{'HL70_t':>9}{'EXT_IC':>10}{'EXT_t':>9}{'S44_IC':>10}{'S44_t':>9}",
          flush=True)
    for v in VARIANTS:
        h = summary["HL70"]["alpha"][v]; e = summary["EXT"]["alpha"][v]; s = summary["S44"]["alpha"][v]
        cur = "  <-current" if v == "mom_30d" else ""
        print(f"  {v:>14}{h[0]:>+10.4f}{h[1]:>+9.2f}{e[0]:>+10.4f}{e[1]:>+9.2f}{s[0]:>+10.4f}{s[1]:>+9.2f}{cur}",
              flush=True)

    print(f"\nDone [{time.time()-t0:.0f}s]", flush=True)


if __name__ == "__main__":
    main()
