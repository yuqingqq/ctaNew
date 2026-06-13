"""iter-035 — EX-ANTE STRUCTURAL panel-selection STANDARDS, validated nested-OOS.

GOAL: a maintainable, rule-based "good panel" selector that beats naive full-156 and at least
matches the curated established-70 (iter-034), validated HONESTLY (the RULE nested-OOS), so the live
universe can be refreshed by rule as listings drift.

HARD CONSTRAINT (#1 lesson): standards must be EX-ANTE STRUCTURAL, computed from TRAILING/PIT data
ONLY — NEVER select by trailing Sharpe/IC/performance (proven value-negative: IC-selector,
high-IC-names carry non-repeating alpha). Any rule that ranks/keeps symbols by realized PnL/IC is
forbidden here. The honest test of a selection RULE is NESTED-OOS: build the panel by the rule using
ONLY past data, trade it forward, repeat per fold; concatenate.

Reuses the iter-032 fast engine VERBATIM (which imports iter-031 X117 held-book + X125 vol-norm stop,
verified == slow engine to 1e-13). All numbers @4.5bps with the iter-012 vol-norm stop (deploy config).

Standards tested (each PIT, structural, NO performance):
  - maturity floor: include a symbol only after >=N days of history AS OF FOLD START (sweep N).
  - liquidity floor: trailing median $vol >= threshold (execution floor; test it doesn't hurt).
  - hygiene: drop stables/wrapped/pegged/non-crypto-beta (PAXG = tokenized gold).
  - de-duplication: among trailing-corr>0.9 pairs, keep one (maximize effective breadth).
  - dispersion floor (skeptical): trailing idio-vol-to-BTC above a floor (structural, ex-ante).

Controls:
  (a) naive full-156
  (b) established-70 (HL68-in-x132)
  (c) RANDOM-panel-of-same-size (panel-construction placebo): per fold, draw random subsets of the
      SAME count as the rule selects, from the maturity+hygiene-eligible pool. Does the RULE beat
      random selection of the same count, nested-OOS?
"""
from __future__ import annotations
import time, pickle, importlib.util as _ilu
from pathlib import Path
import numpy as np
import pandas as pd

REPO = Path("/home/yuqing/ctaNew")
SCR = REPO/"agents_system/research/scratch"
RC = REPO/"research/convexity_portable_2026-05-20/results/_cache"
KLINES = REPO/"data/ml/test/parquet/klines"
OUT = REPO/"outputs/iter035"; OUT.mkdir(parents=True, exist_ok=True)

X132_PREDS = RC/"x132_expanded_v0_preds.parquet"
HL70_PREDS = RC/"x64_HL70_V5mv3_7cx_filter_t0.3_preds.parquet"
ANN = 6*365
SEED = 35035
N_RAND = 60  # random-same-size placebo draws

# hygiene: tokenized-gold / non-crypto-beta / stable-proxy. (no USD stables in this perp universe;
# PAXG/XAUT = gold; check for any others structurally below.)
HYGIENE = {"PAXGUSDT"}

# reuse the iter-032 fast engine (imports iter-031 verbatim)
_s = _ilu.spec_from_file_location("i32", SCR/"iter032_expanded_universe.py")
i32 = _ilu.module_from_spec(_s); _s.loader.exec_module(i32)
i31 = i32.i31
precompute_cycles = i32.precompute_cycles
fast_cyc_weights = i32.fast_cyc_weights
PRIMARY_COST = i31.PRIMARY_COST
load_close = i31.x123.load_close


def metrics(pnl):
    return i31.metrics(pnl)


def run_panel_pnl(PC, subset_ids, with_stop=True, hist_ok=None):
    cyc_w, rs = fast_cyc_weights(PC, subset_ids, hist_ok=hist_ok)
    pnl = i31.volnorm_stop(cyc_w, rs, PRIMARY_COST) if with_stop else i31.heldbook(cyc_w, rs, PRIMARY_COST)*1e4
    return pnl


# ============================================================================================
# STRUCTURAL PIT METADATA — per-symbol features computable from data up to ANY cutoff time.
# We precompute per-symbol trailing series so a per-fold-cutoff snapshot is a cheap lookup.
# ============================================================================================
def build_struct_meta(syms):
    """For each symbol: 4h close series, first-seen, daily-$vol series, idio-vol-to-BTC series.
    All series indexed by time so a PIT cutoff = take values strictly before cutoff."""
    WIN = 180
    btc = load_close("BTCUSDT"); b4 = btc[(btc.index.hour % 4 == 0) & (btc.index.minute == 0)]
    br = np.log(b4/b4.shift(1)); bvar = br.rolling(WIN, min_periods=42).var()
    meta = {}
    closes_4h = {}
    for sym in syms:
        c = load_close(sym)
        if c is None:
            continue
        c4 = c[(c.index.hour % 4 == 0) & (c.index.minute == 0)]
        closes_4h[sym] = c4
        first_seen = c4.index.min()
        # trailing idio-vol-to-BTC: std of (sym 4h logret - beta*btc 4h logret) over WIN, rolling
        r = np.log(c4/c4.shift(1)); ri, bi = r.align(br, join="inner")
        beta = (ri.rolling(WIN, min_periods=42).cov(bi)/bvar.reindex(ri.index).replace(0, np.nan))
        resid = ri - beta*bi
        idiovol = resid.rolling(WIN, min_periods=42).std()  # 4h idio vol, PIT
        # daily $vol from 5m quote_volume
        sd = KLINES/sym/"5m"
        if sd.exists():
            dfs = [pd.read_parquet(f, columns=["open_time", "quote_volume"]) for f in sorted(sd.glob("*.parquet"))]
            q = pd.concat(dfs, ignore_index=True).drop_duplicates("open_time")
            q["open_time"] = pd.to_datetime(q["open_time"], utc=True)
            daily = q.set_index("open_time")["quote_volume"].resample("1D").sum()
            daily = daily[daily > 0]
        else:
            daily = pd.Series(dtype=float)
        meta[sym] = dict(first_seen=first_seen, idiovol=idiovol, daily_vol=daily, ret4h=ri)
    return meta, closes_4h


def pit_snapshot(meta, syms, cutoff, liq_win_days=90):
    """PIT structural snapshot AS OF cutoff (use only data strictly before cutoff).
    Returns DataFrame index=sym with: age_days, med_dvol (trailing liq_win_days), idiovol (last PIT)."""
    rows = {}
    for s in syms:
        m = meta.get(s)
        if m is None:
            continue
        fs = m["first_seen"]
        if pd.isna(fs) or fs >= cutoff:
            continue  # not yet listed at cutoff
        age_days = (cutoff - fs).days
        # trailing median daily $vol over the liq window before cutoff
        dv = m["daily_vol"]
        dv = dv[dv.index < cutoff]
        if len(dv) == 0:
            med_dvol = np.nan
        else:
            med_dvol = float(dv.iloc[-liq_win_days:].median())
        # last PIT idio-vol strictly before cutoff
        iv = m["idiovol"]; iv = iv[iv.index < cutoff].dropna()
        idiovol = float(iv.iloc[-1]) if len(iv) else np.nan
        rows[s] = dict(age_days=age_days, med_dvol=med_dvol, idiovol=idiovol)
    return pd.DataFrame(rows).T


def pit_corr_dedup(meta, cand_syms, cutoff, corr_thr=0.9, corr_win_days=120):
    """PIT de-duplication: among symbols whose trailing 4h-return correlation > corr_thr, keep ONE.
    Tie-break by LONGEST history (most mature = keep), structural, no performance.
    Returns the kept subset (list)."""
    win_bars = int(corr_win_days*6)  # 4h bars per day = 6
    # build trailing 4h return matrix up to cutoff
    series = {}
    for s in cand_syms:
        r = meta[s]["ret4h"]
        r = r[r.index < cutoff]
        if len(r) >= 60:
            series[s] = r.iloc[-win_bars:]
    if len(series) < 2:
        return list(cand_syms)
    mat = pd.DataFrame(series).dropna(how="all")
    # require reasonable overlap
    R = mat.corr(min_periods=60)
    # greedy dedup: process by descending maturity (keep most mature), drop later highly-corr names
    age = {s: (cutoff - meta[s]["first_seen"]).days for s in series}
    order = sorted(series.keys(), key=lambda s: -age[s])
    kept = []
    dropped = set()
    for s in order:
        if s in dropped:
            continue
        kept.append(s)
        for t in order:
            if t == s or t in dropped or t in kept:
                continue
            c = R.loc[s, t] if (s in R.index and t in R.columns) else np.nan
            if np.isfinite(c) and c > corr_thr:
                dropped.add(t)
    # names without enough series to correlate: keep (can't judge)
    for s in cand_syms:
        if s not in series and s not in kept:
            kept.append(s)
    return kept


def main():
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    print("="*100)
    print("iter-035 — EX-ANTE STRUCTURAL panel-selection STANDARDS, nested-OOS")
    print("="*100, flush=True)

    # ---- load cached x132 panel + precompute fast cycles ----
    PX = pickle.loads((REPO/"outputs/iter032/panel_x132.pkl").read_bytes())
    print(f"  x132 panel: {len(PX['syms'])} syms, {len(PX['times'])} 4h-cycles", flush=True)
    PCX = precompute_cycles(PX)
    sidx = PCX["sidx"]; syms_by_id = PCX["syms"]
    full156 = [s for s in PX["syms"] if s != "BTCUSDT"]
    hl70 = sorted(pd.read_parquet(HL70_PREDS, columns=["symbol"])["symbol"].unique())
    hl_in = [s for s in hl70 if s in set(full156)]
    print(f"  full-156 tradable={len(full156)}, established-70 (HL in x132)={len(hl_in)}", flush=True)

    def ids(names):
        return [sidx[s] for s in names if s in sidx]

    # fold boundaries (each fold start = cutoff for PIT selection using ONLY past data)
    fbt = PCX["fold_by_time"]; times = PCX["times"]
    folds_arr = np.array([fbt.get(t, -1) for t in times])
    fold_ids = sorted(set(int(f) for f in folds_arr if f >= 0))
    fold_start = {f: min(times[i] for i in range(len(times)) if folds_arr[i] == f) for f in fold_ids}
    print(f"  folds: {fold_ids}; fold starts: " +
          ", ".join(f"f{f}={pd.Timestamp(fold_start[f]).date()}" for f in fold_ids), flush=True)

    # ---- structural PIT metadata (cached to disk; heavy kline I/O) ----
    meta_cache = OUT/"struct_meta.pkl"
    if meta_cache.exists():
        print("  loading cached structural PIT metadata...", flush=True)
        meta = pickle.loads(meta_cache.read_bytes())
    else:
        print("  building structural PIT metadata (first-seen, $vol, idio-vol-to-BTC)...", flush=True)
        meta, closes = build_struct_meta(full156)
        meta_cache.write_bytes(pickle.dumps(meta))
    print(f"  metadata ready for {len(meta)} syms [{time.time()-t0:.0f}s]", flush=True)

    # convenience: per-fold concatenation of a panel's PnL restricted to that fold's cycles, given a
    # per-fold symbol subset. We build full-length cyc weights for a FIXED subset then slice by fold;
    # for NESTED-OOS we vary the subset per fold and stitch.
    def fold_mask(f):
        return folds_arr == f

    def panel_pnl_fixed(subset_names, with_stop=True):
        return run_panel_pnl(PCX, ids(subset_names), with_stop=with_stop)

    def nested_pnl(select_fn, with_stop=True):
        """select_fn(cutoff)->list of names. Build that panel's full pnl, keep only THIS fold's cycles.
        Concatenate across folds = honest nested-OOS (panel chosen from PAST data, traded forward)."""
        out = np.full(len(times), np.nan)
        per_fold_names = {}
        for f in fold_ids:
            cutoff = pd.Timestamp(fold_start[f])
            names = select_fn(cutoff)
            per_fold_names[f] = names
            pnl_f = run_panel_pnl(PCX, ids(names), with_stop=with_stop)
            m = fold_mask(f)
            out[m] = pnl_f[m]
        return out, per_fold_names

    def perfold_pos(pnl):
        fp = 0; nf = 0
        for f in fold_ids:
            seg = pnl[fold_mask(f)]; seg = seg[np.isfinite(seg)]
            if len(seg) >= 3:
                nf += 1
                sh = i31.ann(pd.Series(seg)/1e4)
                if np.isfinite(sh) and sh > 0:
                    fp += 1
        return fp, nf

    def summ(pnl, label):
        m = metrics(pnl); fp, nf = perfold_pos(pnl)
        print(f"  {label:<46} Sh {m['Sharpe']:>+6.2f}  mDD {m['maxDD']:>+8.0f}  "
              f"Cal {m['Calmar']:>+6.2f}  tot {m['tot']:>+9.0f}  %pos {m['pct_pos']:>5.1f}  fp {fp}/{nf}",
              flush=True)
        return dict(label=label, Sharpe=m["Sharpe"], maxDD=m["maxDD"], Calmar=m["Calmar"],
                    tot=m["tot"], pct_pos=m["pct_pos"], folds_positive=fp, n_folds=nf)

    rows = []

    # ============================================================ A. REFERENCE (fixed panels)
    print("\n" + "="*100)
    print("A. REFERENCE PANELS (fixed across folds), +stop @4.5bps")
    print("="*100, flush=True)
    pnl_full = panel_pnl_fixed(full156)
    rows.append(summ(pnl_full, "full-156 (naive)"))
    pnl_70 = panel_pnl_fixed(hl_in)
    rows.append(summ(pnl_70, "established-70 (HL68 in x132)"))

    # ============================================================ B. PIT SNAPSHOT diagnostics
    print("\n" + "="*100)
    print("B. PIT SNAPSHOTS at each fold start — how many names clear each structural floor?")
    print("="*100, flush=True)
    snaps = {}
    for f in fold_ids:
        cutoff = pd.Timestamp(fold_start[f])
        snap = pit_snapshot(meta, full156, cutoff)
        snaps[f] = snap
        listed = len(snap)
        n_mat180 = int((snap["age_days"] >= 180).sum())
        n_mat365 = int((snap["age_days"] >= 365).sum())
        n_liq3m = int((snap["med_dvol"] >= 3e6).sum())
        print(f"  f{f} cutoff {cutoff.date()}: listed={listed:>3}  age>=180d={n_mat180:>3}  "
              f"age>=365d={n_mat365:>3}  $vol>=3M={n_liq3m:>3}", flush=True)

    # ============================================================ C. MATURITY FLOOR sweep (nested-OOS)
    print("\n" + "="*100)
    print("C. MATURITY FLOOR sweep (nested-OOS): include sym only if age>=N days AS OF FOLD START")
    print("   (+ hygiene drop PAXG). Panel re-selected each fold from PAST data only.")
    print("="*100, flush=True)
    def sel_maturity(N_days, extra_liq=None, dispersion_q=None, dedup=False, hygiene=True):
        def fn(cutoff):
            snap = pit_snapshot(meta, full156, cutoff)
            keep = snap[snap["age_days"] >= N_days].index.tolist()
            if hygiene:
                keep = [s for s in keep if s not in HYGIENE]
            if extra_liq is not None:
                snk = snap.loc[keep]
                keep = snk[snk["med_dvol"] >= extra_liq].index.tolist()
            if dispersion_q is not None:
                snk = snap.loc[keep]
                iv = snk["idiovol"].dropna()
                if len(iv) > 5:
                    thr = iv.quantile(dispersion_q)
                    keep = snk[snk["idiovol"] >= thr].index.tolist()
            if dedup:
                keep = pit_corr_dedup(meta, keep, cutoff)
            return keep
        return fn

    mat_panels = {}
    for N in (60, 120, 180, 365):
        pnl, pf = nested_pnl(sel_maturity(N))
        mat_panels[N] = pf
        szs = [len(v) for v in pf.values()]
        rows.append(summ(pnl, f"maturity>={N}d +hygiene (nested-OOS, sz {min(szs)}-{max(szs)})"))

    # ============================================================ D. + LIQUIDITY FLOOR (execution, test no-hurt)
    print("\n" + "="*100)
    print("D. MATURITY(180) + LIQUIDITY FLOOR (execution floor; test it doesn't HURT). nested-OOS")
    print("="*100, flush=True)
    for liq in (1e6, 3e6, 5e6):
        pnl, pf = nested_pnl(sel_maturity(180, extra_liq=liq))
        szs = [len(v) for v in pf.values()]
        rows.append(summ(pnl, f"mat>=180d + $vol>=${liq/1e6:.0f}M (sz {min(szs)}-{max(szs)})"))

    # ============================================================ E. + DEDUP (corr>0.9 keep most-mature)
    print("\n" + "="*100)
    print("E. MATURITY(180) + HYGIENE + DEDUP (trailing corr>0.9 -> keep most-mature). nested-OOS")
    print("="*100, flush=True)
    pnl_dedup, pf_dedup = nested_pnl(sel_maturity(180, dedup=True))
    szs = [len(v) for v in pf_dedup.values()]
    rows.append(summ(pnl_dedup, f"mat>=180d + dedup0.9 (sz {min(szs)}-{max(szs)})"))

    # ============================================================ F. + DISPERSION FLOOR (skeptical)
    print("\n" + "="*100)
    print("F. MATURITY(180) + DISPERSION FLOOR (trailing idio-vol-to-BTC >= quantile). SKEPTICAL. nested-OOS")
    print("="*100, flush=True)
    disp_panels = {}
    for q in (0.25, 0.50):
        pnl, pf = nested_pnl(sel_maturity(180, dispersion_q=q))
        disp_panels[q] = pf
        szs = [len(v) for v in pf.values()]
        rows.append(summ(pnl, f"mat>=180d + idiovol>=q{int(q*100)} (sz {min(szs)}-{max(szs)})"))

    # ============================================================ G. THE CANDIDATE STANDARD (combined)
    print("\n" + "="*100)
    print("G. CANDIDATE STANDARD = mat>=180d + hygiene + $vol>=$3M + dedup0.9 (nested-OOS)")
    print("="*100, flush=True)
    pnl_std, pf_std = nested_pnl(sel_maturity(180, extra_liq=3e6, dedup=True))
    szs_std = [len(v) for v in pf_std.values()]
    row_std = summ(pnl_std, f"STANDARD mat180+liq3M+dedup (sz {min(szs_std)}-{max(szs_std)})")
    rows.append(row_std)

    # save core panel results BEFORE the expensive placebo (survives a timeout)
    pd.DataFrame(rows).to_csv(OUT/"iter035_panels.csv", index=False)
    np.savez(OUT/"iter035_pnls.npz", full=pnl_full, estab70=pnl_70, std=pnl_std,
             folds=folds_arr)
    print(f"  [core panel results saved @ {time.time()-t0:.0f}s]", flush=True)

    # ============================================================ H. RANDOM-SAME-SIZE PLACEBO (per fold)
    print("\n" + "="*100)
    print("H. RANDOM-SAME-SIZE PLACEBO — does the RULE beat random selection of the SAME count?")
    print("   Per fold: the rule selects n_f names; draw N_RAND random subsets of size n_f from the")
    print("   maturity+hygiene-ELIGIBLE pool (same ex-ante feasibility), trade forward, stitch, repeat.")
    print("="*100, flush=True)
    def random_same_size_nested(per_fold_names, eligible_fn, n_draws=N_RAND):
        """eligible_fn(cutoff)->pool to draw from. Returns array of Sharpes (one per draw),
        each draw uses an independent random subset PER FOLD of the rule's per-fold size."""
        shs = []
        for d in range(n_draws):
            out = np.full(len(times), np.nan)
            for f in fold_ids:
                cutoff = pd.Timestamp(fold_start[f])
                pool = eligible_fn(cutoff)
                k = len(per_fold_names[f])
                k = min(k, len(pool))
                if k < 2*i31.K:
                    continue
                sub = list(rng.choice(pool, size=k, replace=False))
                pnl_f = run_panel_pnl(PCX, ids(sub), with_stop=True)
                out[folds_arr == f] = pnl_f[folds_arr == f]
            shs.append(metrics(out)["Sharpe"])
        return np.array(shs, float)

    # eligible pool = maturity>=180 + hygiene (the structural feasibility set the rule draws within)
    elig_fn = lambda cutoff: [s for s in pit_snapshot(meta, full156, cutoff).query("age_days>=180").index
                              if s not in HYGIENE]
    print("  random-same-size placebo for the CANDIDATE STANDARD (size-matched per fold)...", flush=True)
    rand_sh = random_same_size_nested(pf_std, elig_fn)
    real_sh = row_std["Sharpe"]
    rank = (rand_sh < real_sh).mean()*100
    print(f"  STANDARD nested-OOS Sharpe {real_sh:+.3f}", flush=True)
    print(f"  random-same-size ({len(rand_sh)} draws): mean {np.nanmean(rand_sh):+.3f}  "
          f"p5 {np.nanpercentile(rand_sh,5):+.3f}  p50 {np.nanpercentile(rand_sh,50):+.3f}  "
          f"p95 {np.nanpercentile(rand_sh,95):+.3f}  max {np.nanmax(rand_sh):+.3f}", flush=True)
    print(f"  => STANDARD ranks p{rank:.0f} of random-same-size "
          f"({'BEATS random p95 (RULE adds value)' if rank>=95 else 'does NOT beat random p95'})", flush=True)

    # also placebo for the simplest mat-only-180 rule (does even maturity beat random-same-size?)
    print("\n  random-same-size placebo for MATURITY>=180d-ONLY rule...", flush=True)
    pnl_mat180, pf_mat180 = nested_pnl(sel_maturity(180))
    rand_sh_m = random_same_size_nested(pf_mat180, elig_fn)
    real_m = metrics(pnl_mat180)["Sharpe"]
    rank_m = (rand_sh_m < real_m).mean()*100
    print(f"  MAT>=180d nested-OOS Sharpe {real_m:+.3f}; random-same-size mean {np.nanmean(rand_sh_m):+.3f} "
          f"p95 {np.nanpercentile(rand_sh_m,95):+.3f} -> ranks p{rank_m:.0f}", flush=True)

    # ============================================================ I. PAIRED CI: STANDARD vs full-156
    print("\n" + "="*100)
    print("I. PAIRED block-bootstrap CI: STANDARD nested-OOS minus full-156 (per-cycle, by fold)")
    print("="*100, flush=True)
    diff = pnl_std - pnl_full
    uf = fold_ids
    boots = []
    for _ in range(2000):
        chosen = rng.choice(uf, size=len(uf), replace=True)
        seg = np.concatenate([diff[folds_arr == f] for f in chosen])
        boots.append(np.nanmean(seg))
    lo, hi = np.nanpercentile(boots, [2.5, 97.5])
    print(f"  mean per-cycle diff (STANDARD - full156) {np.nanmean(diff):+.3f} bps; "
          f"95% CI [{lo:+.3f}, {hi:+.3f}] ({'clears 0' if lo>0 or hi<0 else 'CROSSES 0'})", flush=True)
    # vs established-70
    diff70 = pnl_std - pnl_70
    boots70 = []
    for _ in range(2000):
        chosen = rng.choice(uf, size=len(uf), replace=True)
        seg = np.concatenate([diff70[folds_arr == f] for f in chosen])
        boots70.append(np.nanmean(seg))
    lo7, hi7 = np.nanpercentile(boots70, [2.5, 97.5])
    print(f"  mean per-cycle diff (STANDARD - estab70)  {np.nanmean(diff70):+.3f} bps; "
          f"95% CI [{lo7:+.3f}, {hi7:+.3f}] ({'clears 0' if lo7>0 or hi7<0 else 'CROSSES 0'})", flush=True)

    # save
    pd.DataFrame(rows).to_csv(OUT/"iter035_panels.csv", index=False)
    np.savez(OUT/"iter035_placebo.npz", rand_std=rand_sh, real_std=real_sh,
             rand_mat=rand_sh_m, real_mat=real_m)
    # per-fold panel sizes for the standard
    pf_log = {f"f{f}": len(v) for f, v in pf_std.items()}
    print(f"\n  STANDARD per-fold panel sizes: {pf_log}", flush=True)
    print(f"\nDone [{time.time()-t0:.0f}s]  -> {OUT}", flush=True)
    return rows


if __name__ == "__main__":
    main()
