"""Decisive test of the +12.8% positioning-driven decile short: is it a real WALK-FORWARD signal or a fixed-split
artifact? Expanding-window walk-forward over positioning-covered pumps: step through time in 6-week blocks; at each
step train on ALL strictly-prior covered entries, short the predicted-worst decile of the block, accumulate. Pool all
walk-forward decile shorts -> week-clustered CI. This is the honest forward simulation (train past, trade next block).
Controls:
  - POS-SHUFFLE placebo: shuffle positioning columns across rows before fit -> if decile still wins, positioning is
    NOT the driver (concentration/price is). If it collapses, positioning carries the signal.
  - price+funding-only (no positioning) walk-forward, same folds.
Warmup: start stepping once train >= 250 covered entries (need enough dumps to learn).
"""
from pathlib import Path
import numpy as np, pandas as pd
import warnings; warnings.filterwarnings("ignore")
from lightgbm import LGBMRegressor
SD = Path("/tmp/claude-1001/-home-yuqing-ctaNew/ecbd8f4c-236c-426c-85e5-e1f6b6edd11d/scratchpad")
rng = np.random.default_rng(7); N_FUND = 21; COST = 0.0040
PF = ["climax", "climax_build", "runup_3d", "runup_1d", "parab", "rvol_7d", "dist_ath", "taker", "age_d",
      "funding", "funding_chg", "funding_z"]
POS = ["oi_chg", "tt_ls", "ls", "taker_ls"]

def wk_boot(t, x):
    x = np.asarray(x, float); t = pd.to_datetime(np.asarray(t), utc=True)
    wk = pd.Series(t).dt.to_period("W").astype(str).values
    grps = [x[wk == w] for w in pd.unique(wk)]
    if len(grps) < 4: return (np.nan, np.nan)
    out = [np.concatenate([grps[i] for i in rng.integers(0, len(grps), len(grps))]).mean() for _ in range(4000)]
    return tuple(np.percentile(out, [2.5, 97.5]))

def mk(seed=0): return LGBMRegressor(n_estimators=250, num_leaves=7, learning_rate=0.03, min_child_samples=30,
                                     subsample=0.8, colsample_bytree=0.7, reg_lambda=5.0, random_state=seed, verbose=-1)

def walk(e, feats, q=10, shuffle_pos=False, warm=250, step="42D"):
    e = e.sort_values("t").reset_index(drop=True)
    feats = [f for f in feats if e[f].notna().mean() > 0.5]
    t0 = e["t"].iloc[warm]; tmax = e["t"].max(); shorts = []; nblk = 0
    step_td = pd.Timedelta(step); cur = t0
    while cur <= tmax:
        tr = e[e.t < cur]; te = e[(e.t >= cur) & (e.t < cur + step_td)]
        if len(tr) >= warm and len(te) >= q:
            trf = tr.copy()
            if shuffle_pos:
                for c in POS:
                    if c in trf: trf[c] = rng.permutation(trf[c].values)
            med = trf[feats].median()
            P = np.array([mk(s).fit(trf[feats].fillna(med), trf["fwd_ret"].clip(-0.9, 2.0).values).predict(te[feats].fillna(med)) for s in range(3)])
            te = te.copy(); te["pred"] = P.mean(0)
            te["ct"] = pd.qcut(te["pred"].rank(method="first"), q, labels=False, duplicates="drop")
            s = te[te.ct == 0].copy(); s["net"] = -s["fwd_ret"] + s["funding"] * N_FUND - COST
            shorts.append(s); nblk += 1
        cur = cur + step_td
    if not shorts: return None
    S = pd.concat(shorts); lo, up = wk_boot(S["t"], S["net"].values)
    return S, nblk, (S["net"].mean(), np.median(S["net"]), (S["net"] > 0).mean(), lo, up)

def line(tag, r):
    if r is None: print(f"    {tag:36s} (insufficient folds)"); return
    S, nblk, (mean, medn, win, lo, up) = r
    f = "NET>0 (CI>0)" if lo > 0 else ("NET<0" if up < 0 else "CI~0")
    print(f"    {tag:36s} folds={nblk} n={len(S):3d} | mean {mean*100:+5.1f}% median {medn*100:+5.1f}% win {win*100:.0f}% [wkCI {lo*100:+.1f},{up*100:+.1f}] -> {f}")

def main():
    e = pd.read_csv(SD / "pump_enriched.csv"); e["t"] = pd.to_datetime(e["t"], utc=True)
    e = e.dropna(subset=["fwd_ret", "funding"]); ec = e[e["tt_ls"].notna()].copy()
    print(f"positioning-covered pumps: {len(ec)} | span {str(ec.t.min())[:10]} -> {str(ec.t.max())[:10]}\n")
    print("### EXPANDING WALK-FORWARD (train all-prior, short next-6wk block's predicted decile) ###")
    line("M3 +positioning   decile", walk(ec, PF + POS, q=10))
    line("M3 +positioning   tercile", walk(ec, PF + POS, q=3))
    line("M2 price+funding  decile", walk(ec, PF, q=10))
    print("\n### PLACEBO — positioning shuffled (breaks smart$/crowd alignment) ###")
    for i in range(3):
        line(f"M3 POS-SHUFFLED   decile #{i+1}", walk(ec, PF + POS, q=10, shuffle_pos=True))
    print("\nread: real if M3 decile CI>0 AND shuffled-POS collapses AND M2 (no-pos) is weaker. PROBE3DONE")

if __name__ == "__main__":
    main()
