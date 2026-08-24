"""Adversarial verification before believing the +9.0% positioning-driven walk-forward (it reverses a prior 'closed'
verdict, and a positioning LOOK-AHEAD would fake exactly this). Two checks:
 (A) PIT spot-check: for each decile-basket name, confirm the tt_ls/oi it was fed equals the LAST metrics snapshot
     STRICTLY BEFORE entry (re-derive from raw metrics parquet, compare to enriched value; flag any that used a
     future snapshot). Also print metrics timestamp semantics + staleness (entry_t - snapshot_t >= 0 always).
 (B) per-fold breakdown of the walk-forward decile: is +9% spread across folds or 1-2 folds carrying it?
"""
from pathlib import Path
import numpy as np, pandas as pd
import warnings; warnings.filterwarnings("ignore")
from lightgbm import LGBMRegressor
KD = Path("/home/yuqing/ctaNew/data/ml/cache")
SD = Path("/tmp/claude-1001/-home-yuqing-ctaNew/ecbd8f4c-236c-426c-85e5-e1f6b6edd11d/scratchpad")
rng = np.random.default_rng(7); N_FUND = 21; COST = 0.0040
PF = ["climax", "climax_build", "runup_3d", "runup_1d", "parab", "rvol_7d", "dist_ath", "taker", "age_d",
      "funding", "funding_chg", "funding_z"]; POS = ["oi_chg", "tt_ls", "ls", "taker_ls"]

def mk(s=0): return LGBMRegressor(n_estimators=250, num_leaves=7, learning_rate=0.03, min_child_samples=30,
                                  subsample=0.8, colsample_bytree=0.7, reg_lambda=5.0, random_state=s, verbose=-1)

def raw_metric(sym, col):
    p = KD / f"metrics_{sym}.parquet"
    if not p.exists(): return None
    d = pd.read_parquet(p)
    if not isinstance(d.index, pd.DatetimeIndex):
        tc = "create_time" if "create_time" in d.columns else "calc_time"
        d = d.set_index(pd.to_datetime(d[tc], utc=True))
    if col not in d.columns: return None
    s = d[col].sort_index(); return s[~s.index.duplicated()]

def main():
    e = pd.read_csv(SD / "pump_enriched.csv"); e["t"] = pd.to_datetime(e["t"], utc=True)
    e = e.dropna(subset=["fwd_ret", "funding"]); ec = e[e["tt_ls"].notna()].copy().sort_values("t").reset_index(drop=True)

    # rebuild the walk-forward, keep per-fold + basket
    feats = PF + POS; warm = 250; step = pd.Timedelta("42D")
    t0 = ec["t"].iloc[warm]; cur = t0; rows = []; fold = 0
    while cur <= ec["t"].max():
        tr = ec[ec.t < cur]; te = ec[(ec.t >= cur) & (ec.t < cur + step)]
        if len(tr) >= warm and len(te) >= 10:
            med = tr[feats].median()
            P = np.array([mk(s).fit(tr[feats].fillna(med), tr["fwd_ret"].clip(-0.9, 2.0).values).predict(te[feats].fillna(med)) for s in range(3)])
            te = te.copy(); te["pred"] = P.mean(0)
            te["ct"] = pd.qcut(te["pred"].rank(method="first"), 10, labels=False, duplicates="drop")
            s = te[te.ct == 0].copy(); s["net"] = -s["fwd_ret"] + s["funding"] * N_FUND - COST; s["fold"] = fold
            rows.append(s); fold += 1
        cur = cur + step
    S = pd.concat(rows)

    print("### (B) per-fold walk-forward decile (is +9% spread or 1-2 folds?) ###")
    for f, g in S.groupby("fold"):
        print(f"    fold {f:2d} {str(g.t.min())[:10]}..{str(g.t.max())[:10]} n={len(g)} net {g.net.mean()*100:+6.1f}% (names {','.join(g.sym.str.replace('USDT','').tolist())})")
    pos_folds = S.groupby("fold").net.mean().gt(0).sum(); nf = S.fold.nunique()
    print(f"    -> {pos_folds}/{nf} folds net-positive | pooled mean {S.net.mean()*100:+.1f}% median {S.net.median()*100:+.1f}%")

    print("\n### (A) PIT spot-check on the decile basket (fed value == last snapshot STRICTLY before entry?) ###")
    bad = 0
    for _, r in S.sort_values("t").iterrows():
        s = raw_metric(r["sym"], "sum_toptrader_long_short_ratio")
        if s is None: continue
        before = s[s.index < r["t"]]                       # STRICTLY before entry
        after_ok = s[s.index <= r["t"]]
        pit = before.iloc[-1] if len(before) else np.nan
        fed = r["tt_ls"]; snap_t = before.index[-1] if len(before) else None
        stale_h = (r["t"] - snap_t).total_seconds()/3600 if snap_t is not None else np.nan
        # look-ahead would show fed == a snapshot AT/AFTER entry that differs from the last-before
        leak = (not np.isnan(fed)) and (not np.isnan(pit)) and abs(fed - pit) > 1e-6
        if leak: bad += 1
        flag = "  <-- MISMATCH(recheck)" if leak else ""
        print(f"    {str(r['t'])[:10]} {r['sym']:12s} fed_tt_ls {fed:.4f} last-before {pit if not np.isnan(pit) else float('nan'):.4f} staleness {stale_h:4.1f}h{flag}")
    print(f"    -> {bad} names used a value != last-strictly-before snapshot (0 = clean PIT, ffill staleness>=0 always)")
    print("PROBE4DONE")

if __name__ == "__main__":
    main()
