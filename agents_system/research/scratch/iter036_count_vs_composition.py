"""iter-036 — WHY is >70 (wide) worse than established-70? COUNT vs COMPOSITION decomposition.

Reuses iter-035 engine + cached struct_meta (NO rebuild). All @4.5bps + iter-012 stop, full 2021-26.

STEP 2 plan:
 1. COUNT within MATURE: random-N draws (40 seeds) from maturity>=180d+hygiene eligible pool at
    N in {40,70,100,full~140} per fold (nested-OOS, fixed pool feasibility). Sharpe vs N?
 2. COMPOSITION: established-70 percentile vs random-70-from-mature distribution.
 3. EX-ANTE capture: top-N by trailing cumulative-$volume (structural, PIT) and by listing-age.
    Nested-OOS vs (a) established-70 (b) mature-wide (c) random-70-from-mature.
 4. NOISE: paired block-bootstrap CI of (best ex-ante - mature-wide) and (established-70 - mature-wide).
"""
from __future__ import annotations
import time, pickle, importlib.util as _ilu
from pathlib import Path
import numpy as np
import pandas as pd

REPO = Path("/home/yuqing/ctaNew")
SCR = REPO/"agents_system/research/scratch"
RC = REPO/"research/convexity_portable_2026-05-20/results/_cache"
OUT = REPO/"outputs/iter036"; OUT.mkdir(parents=True, exist_ok=True)
HL70_PREDS = RC/"x64_HL70_V5mv3_7cx_filter_t0.3_preds.parquet"
HYGIENE = {"PAXGUSDT"}
SEED = 36036
N_RAND = 20

_s = _ilu.spec_from_file_location("i32", SCR/"iter032_expanded_universe.py")
i32 = _ilu.module_from_spec(_s); _s.loader.exec_module(i32)
i31 = i32.i31
precompute_cycles = i32.precompute_cycles
fast_cyc_weights = i32.fast_cyc_weights
PRIMARY_COST = i31.PRIMARY_COST


def metrics(pnl):
    return i31.metrics(pnl)


def run_panel_pnl(PC, subset_ids, with_stop=True):
    cyc_w, rs = fast_cyc_weights(PC, subset_ids)
    return i31.volnorm_stop(cyc_w, rs, PRIMARY_COST) if with_stop else i31.heldbook(cyc_w, rs, PRIMARY_COST)*1e4


def main():
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    print("="*100, flush=True)
    print("iter-036 — COUNT vs COMPOSITION decomposition (why >70 worse)", flush=True)
    print("="*100, flush=True)

    PX = pickle.loads((REPO/"outputs/iter032/panel_x132.pkl").read_bytes())
    PCX = precompute_cycles(PX)
    sidx = PCX["sidx"]
    full156 = [s for s in PX["syms"] if s != "BTCUSDT"]
    hl70 = sorted(pd.read_parquet(HL70_PREDS, columns=["symbol"])["symbol"].unique())
    hl_in = [s for s in hl70 if s in set(full156)]
    print(f"  full-156={len(full156)}, established-70={len(hl_in)}", flush=True)

    def ids(names):
        return [sidx[s] for s in names if s in sidx]

    fbt = PCX["fold_by_time"]; times = PCX["times"]
    folds_arr = np.array([fbt.get(t, -1) for t in times])
    fold_ids = sorted(set(int(f) for f in folds_arr if f >= 0))
    fold_start = {f: min(times[i] for i in range(len(times)) if folds_arr[i] == f) for f in fold_ids}

    meta = pickle.loads((REPO/"outputs/iter035/struct_meta.pkl").read_bytes())
    print(f"  cached struct_meta: {len(meta)} syms", flush=True)

    # PIT eligible pool = maturity>=180d + hygiene (the established mature pool)
    def elig_pool(cutoff):
        pool = []
        for s in full156:
            m = meta.get(s)
            if m is None:
                continue
            fs = m["first_seen"]
            if pd.isna(fs) or fs >= cutoff:
                continue
            if (cutoff - fs).days < 180:
                continue
            if s in HYGIENE:
                continue
            pool.append(s)
        return pool

    # PIT trailing cumulative-$volume (sum of daily $vol over trailing 180d before cutoff)
    def cum_dvol(s, cutoff, win=180):
        dv = meta[s]["daily_vol"]; dv = dv[dv.index < cutoff]
        if len(dv) == 0:
            return np.nan
        return float(dv.iloc[-win:].sum())

    def age_days(s, cutoff):
        fs = meta[s]["first_seen"]
        return (cutoff - fs).days if (not pd.isna(fs) and fs < cutoff) else -1

    def fold_mask(f):
        return folds_arr == f

    def nested_pnl(select_fn, with_stop=True):
        out = np.full(len(times), np.nan)
        pf = {}
        for f in fold_ids:
            cutoff = pd.Timestamp(fold_start[f])
            names = select_fn(cutoff); pf[f] = names
            pnl_f = run_panel_pnl(PCX, ids(names), with_stop=with_stop)
            out[fold_mask(f)] = pnl_f[fold_mask(f)]
        return out, pf

    def panel_pnl_fixed(names, with_stop=True):
        return run_panel_pnl(PCX, ids(names), with_stop=with_stop)

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
        print(f"  {label:<48} Sh {m['Sharpe']:>+6.2f}  mDD {m['maxDD']:>+8.0f}  "
              f"Cal {m['Calmar']:>+6.2f}  tot {m['tot']:>+9.0f}  fp {fp}/{nf}", flush=True)
        return dict(label=label, Sharpe=m["Sharpe"], maxDD=m["maxDD"], Calmar=m["Calmar"],
                    tot=m["tot"], folds_positive=fp, n_folds=nf)

    rows = []
    # ---- references ----
    print("\nA. REFERENCES", flush=True)
    pnl_full = panel_pnl_fixed(full156); rows.append(summ(pnl_full, "full-156 (naive)"))
    pnl_70 = panel_pnl_fixed(hl_in); rows.append(summ(pnl_70, "established-70 (curated)"))
    pnl_mat, pf_mat = nested_pnl(elig_pool)
    szs = [len(v) for v in pf_mat.values()]
    rows.append(summ(pnl_mat, f"mature-wide (>=180d, nested sz {min(szs)}-{max(szs)})"))

    # ============================================================ 1. COUNT within MATURE
    print("\n1. COUNT EFFECT within MATURE pool: random-N draws (nested-OOS), N in {40,70,100,full}", flush=True)
    def random_N_nested(N, n_draws=N_RAND):
        shs = []; tots = []
        for d in range(n_draws):
            out = np.full(len(times), np.nan)
            for f in fold_ids:
                cutoff = pd.Timestamp(fold_start[f])
                pool = elig_pool(cutoff)
                k = N if N is not None else len(pool)
                k = min(k, len(pool))
                if k < 2*i31.K:
                    continue
                sub = list(rng.choice(pool, size=k, replace=False))
                pnl_f = run_panel_pnl(PCX, ids(sub), with_stop=True)
                out[folds_arr == f] = pnl_f[folds_arr == f]
            m = metrics(out); shs.append(m["Sharpe"]); tots.append(m["tot"])
        return np.array(shs, float), np.array(tots, float)

    count_res = {}
    for N in (40, 70, 100, None):
        sh, tot = random_N_nested(N)
        lbl = "full(~140)" if N is None else str(N)
        count_res[lbl] = sh
        print(f"  N={lbl:<10} Sh mean {np.nanmean(sh):+.3f}  p25 {np.nanpercentile(sh,25):+.3f}  "
              f"p50 {np.nanpercentile(sh,50):+.3f}  p75 {np.nanpercentile(sh,75):+.3f}  "
              f"p95 {np.nanpercentile(sh,95):+.3f}  max {np.nanmax(sh):+.3f}", flush=True)
    # incremental save (survive a timeout)
    pd.DataFrame(rows).to_csv(OUT/"iter036_panels.csv", index=False)
    np.savez(OUT/"iter036_count.npz", **{f"count_{k}": v for k, v in count_res.items()})
    print(f"  [count sweep saved @ {time.time()-t0:.0f}s]", flush=True)

    # ============================================================ 2. COMPOSITION: established-70 percentile
    print("\n2. COMPOSITION: established-70 (+1.34) vs random-70-from-mature distribution", flush=True)
    rand70 = count_res["70"]
    estab_sh = rows[1]["Sharpe"]
    rank70 = (rand70 < estab_sh).mean()*100
    print(f"  established-70 fixed Sharpe {estab_sh:+.3f}", flush=True)
    print(f"  random-70-from-mature: mean {np.nanmean(rand70):+.3f}  p50 {np.nanpercentile(rand70,50):+.3f}  "
          f"p95 {np.nanpercentile(rand70,95):+.3f}  max {np.nanmax(rand70):+.3f}", flush=True)
    print(f"  => established-70 ranks p{rank70:.0f} of random-70 "
          f"({'SPECIAL (p>=90)' if rank70>=90 else 'TYPICAL mature-70 (composition NOT special)'})", flush=True)
    # also: % of mature-70 names that are IN established-70 (overlap)
    pool_last = elig_pool(pd.Timestamp(fold_start[fold_ids[-1]]))
    ov = len(set(hl_in) & set(pool_last))
    print(f"  overlap: {ov}/{len(hl_in)} established-70 names are in last-fold mature pool "
          f"({len(pool_last)} eligible)", flush=True)

    # ============================================================ 3. EX-ANTE structural capture
    print("\n3. EX-ANTE STRUCTURAL rules (nested-OOS): top-N by trailing cum-$vol / by listing-age", flush=True)
    def sel_top_dvol(N):
        def fn(cutoff):
            pool = elig_pool(cutoff)
            scored = [(s, cum_dvol(s, cutoff)) for s in pool]
            scored = [(s, v) for s, v in scored if np.isfinite(v)]
            scored.sort(key=lambda x: -x[1])
            return [s for s, _ in scored[:N]]
        return fn

    def sel_top_age(N):
        def fn(cutoff):
            pool = elig_pool(cutoff)
            scored = sorted(pool, key=lambda s: -age_days(s, cutoff))
            return scored[:N]
        return fn

    exante = {}
    for N in (40, 70, 100):
        pnl_dv, pf_dv = nested_pnl(sel_top_dvol(N))
        exante[("dvol", N)] = (pnl_dv, pf_dv)
        rows.append(summ(pnl_dv, f"EXANTE top-{N} by cum-$vol (nested-OOS)"))
    for N in (70,):
        pnl_ag, pf_ag = nested_pnl(sel_top_age(N))
        exante[("age", N)] = (pnl_ag, pf_ag)
        rows.append(summ(pnl_ag, f"EXANTE top-{N} by listing-age (nested-OOS)"))

    # pick best ex-ante by Sharpe among the candidates
    cand = [("dvol", 40), ("dvol", 70), ("dvol", 100), ("age", 70)]
    best_key = max(cand, key=lambda k: metrics(exante[k][0])["Sharpe"])
    pnl_best, pf_best = exante[best_key]
    best_sh = metrics(pnl_best)["Sharpe"]
    print(f"  BEST ex-ante rule: top-{best_key[1]} by {best_key[0]} -> Sharpe {best_sh:+.3f}", flush=True)

    # does best ex-ante beat random-of-same-size-from-mature? (use the matched N's random distribution)
    rand_match = count_res.get(str(best_key[1]), count_res["70"])
    rank_best = (rand_match < best_sh).mean()*100
    print(f"  best ex-ante ranks p{rank_best:.0f} of random-{best_key[1]}-from-mature "
          f"({'BEATS random p95' if rank_best>=95 else 'does NOT beat random'})", flush=True)

    # ============================================================ 4. NOISE: paired block-bootstrap CIs
    print("\n4. PAIRED block-bootstrap CI (per-fold resample)", flush=True)
    def paired_ci(a, b, label):
        diff = a - b
        boots = []
        for _ in range(2000):
            chosen = rng.choice(fold_ids, size=len(fold_ids), replace=True)
            seg = np.concatenate([diff[folds_arr == f] for f in chosen])
            boots.append(np.nanmean(seg))
        lo, hi = np.nanpercentile(boots, [2.5, 97.5])
        verdict = "clears 0 (REAL)" if (lo > 0 or hi < 0) else "CROSSES 0 (within noise)"
        print(f"  {label:<42} mean diff {np.nanmean(diff):+.3f} bps  CI [{lo:+.3f},{hi:+.3f}]  {verdict}", flush=True)
        return float(np.nanmean(diff)), float(lo), float(hi)

    ci_best_mat = paired_ci(pnl_best, pnl_mat, "best-exante - mature-wide")
    ci_estab_mat = paired_ci(pnl_70, pnl_mat, "established-70 - mature-wide")
    ci_estab_full = paired_ci(pnl_70, pnl_full, "established-70 - full-156")
    ci_best_full = paired_ci(pnl_best, pnl_full, "best-exante - full-156")

    pd.DataFrame(rows).to_csv(OUT/"iter036_panels.csv", index=False)
    np.savez(OUT/"iter036_results.npz",
             **{f"count_{k}": v for k, v in count_res.items()},
             rand70=rand70, estab_sh=estab_sh, rank70=rank70,
             best_sh=best_sh, rank_best=rank_best,
             best_key=str(best_key),
             ci_best_mat=ci_best_mat, ci_estab_mat=ci_estab_mat,
             ci_estab_full=ci_estab_full, ci_best_full=ci_best_full)
    print(f"\nDone [{time.time()-t0:.0f}s] -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
