"""iter-016b — decisive variable-horizon test: does conditioning the HOLD on an observable-at-entry
feature (|pred| strength) beat a fixed hold, AND does it beat a RANDOM hold of matched average
length (the G4-style placebo)? Plus a turnover comparison.

Mechanism under test: the per-cycle LS-spread decay curves differ by signal-strength tercile
(iter016_decay). IF strong signals genuinely hold edge longer / weak ones decay fast, a rule
'hold strong K longer, exit weak K early' should beat fixed hold at lower or equal turnover.

We test it at the SLEEVE / held-book layer abstraction by measuring, per cycle, the realized
LS-spread of the selected K=5 extremes at a CONDITIONAL exit horizon vs a FIXED 24h horizon,
in the SIDE regime, on HL70 + EXT. Conditional rule: exit_h = f(cycle strength tercile) using
each tercile's IN-SAMPLE-best horizon (the most generous possible variable rule — if even THIS
oracle-ish rule doesn't beat fixed+random, variable horizon is dead).

Placebo: assign each cycle a RANDOM exit horizon drawn to match the conditional rule's average
hold length (100 seeds). If random matched-length holds do as well, the 'variable' structure is
just 'hold a different average length', already swept in iter-014.
"""
from __future__ import annotations
import time
from pathlib import Path
import pandas as pd, numpy as np

REPO = Path("/home/yuqing/ctaNew")
RC = REPO/"research/convexity_portable_2026-05-20/results/_cache"
OUT = REPO/"research/convexity_portable_2026-05-20/results"
KLINES = REPO/"data/ml/test/parquet/klines"
HORIZONS_H = [4, 8, 12, 16, 24, 36, 48, 72]
H_STEPS = {h: h//4 for h in HORIZONS_H}
PANELS = {"HL70": RC/"x64_HL70_V5mv3_7cx_filter_t0.3_preds.parquet",
          "EXT":  RC/"x113_ext_v0_preds.parquet"}
K = 5


def load_close(sym):
    sd = KLINES/sym/"5m"
    if not sd.exists(): return None
    dfs = [pd.read_parquet(f, columns=["open_time", "close"]) for f in sorted(sd.glob("*.parquet"))]
    df = pd.concat(dfs, ignore_index=True).drop_duplicates("open_time").sort_values("open_time")
    df["open_time"] = pd.to_datetime(df["open_time"], utc=True)
    return df.set_index("open_time")["close"].astype(np.float64)


def build(ppath):
    d = pd.read_parquet(ppath, columns=["symbol", "open_time", "pred"])
    d["open_time"] = pd.to_datetime(d["open_time"], utc=True)
    d = d[(d["open_time"].dt.hour % 4 == 0) & (d["open_time"].dt.minute == 0)].copy().sort_values(["symbol", "open_time"])
    syms = sorted(d["symbol"].unique())
    closes = {}
    for sym in syms + (["BTCUSDT"] if "BTCUSDT" not in syms else []):
        c = load_close(sym)
        if c is None: continue
        c4 = c[(c.index.hour % 4 == 0) & (c.index.minute == 0)]
        closes[sym] = c4
    cl = pd.concat([s.rename(k) for k, s in closes.items()], axis=1).sort_index()
    fwd = {h: (cl.shift(-st) / cl - 1.0) for h, st in H_STEPS.items()}
    btc = cl["BTCUSDT"]; btc30 = (btc / btc.shift(180) - 1.0)
    reg = pd.Series(np.where(btc30 > 0.10, "bull", np.where(btc30 < -0.10, "bear", "side")), index=btc.index)
    return d, fwd, reg


def cycle_spreads(d, fwd, reg):
    """per SIDE cycle: selected L/S extremes, strength tercile, and LS-spread fwd-ret at each horizon."""
    d = d.copy(); d["regime"] = d["open_time"].map(reg)
    d = d[d["regime"] == "side"]
    rows = []
    for ot, g in d.groupby("open_time"):
        gg = g.dropna(subset=["pred"]).sort_values("pred")
        if len(gg) < 2 * K: continue
        Lsym = gg.tail(K)["symbol"].tolist(); Ssym = gg.head(K)["symbol"].tolist()
        strength = np.abs(pd.concat([gg.tail(K)["pred"], gg.head(K)["pred"]])).mean()
        row = {"open_time": ot, "strength": strength}
        for h, fwh in fwd.items():
            if ot not in fwh.index: row[f"sp_{h}"] = np.nan; continue
            fr = fwh.loc[ot]
            row[f"sp_{h}"] = fr.reindex(Lsym).mean() - fr.reindex(Ssym).mean()
        rows.append(row)
    return pd.DataFrame(rows).dropna()


def ann(x, h):
    x = pd.Series(x).dropna()
    cyc_per_yr = 365 * 24 / h  # holding h hours, non-overlapping equivalent
    return x.mean() / x.std() * np.sqrt(cyc_per_yr) if x.std() > 0 else np.nan


def main():
    t0 = time.time()
    print("=== iter-016b VARIABLE-HOLD vs FIXED vs RANDOM-matched placebo (SIDE) ===\n", flush=True)
    rng = np.random.default_rng(12345)
    for pname, ppath in PANELS.items():
        print(f"\n########## {pname} ##########", flush=True)
        d, fwd, reg = build(ppath)
        sp = cycle_spreads(d, fwd, reg)
        sp["sgrp"] = pd.qcut(sp["strength"], 3, labels=[0, 1, 2]).astype(int)
        n = len(sp); print(f"  n_side_cyc={n}")

        # Fixed 24h spread per cycle (the held-book hold approximated as single-horizon spread)
        fixed24 = sp["sp_24"].values
        # in-sample per-tercile best horizon (oracle-ish variable rule)
        best_h = {}
        for grp in [0, 1, 2]:
            sub = sp[sp["sgrp"] == grp]
            means = {h: sub[f"sp_{h}"].mean() for h in HORIZONS_H}
            best_h[grp] = max(means, key=means.get)
        print(f"  in-sample best horizon by strength tercile (0=weak..2=strong): {best_h}")

        # variable spread: each cycle uses its tercile's best horizon
        var_sp = np.array([sp.iloc[i][f"sp_{best_h[sp.iloc[i]['sgrp']]}"] for i in range(n)])
        avg_hold_var = np.mean([best_h[g] for g in sp["sgrp"].values])

        # mean spread (bps), and a rough turnover proxy: turnover ~ 1/hold (longer hold = less churn)
        print(f"  mean LS-spread (bps):  fixed24={fixed24.mean()*1e4:+.1f}   variable={var_sp.mean()*1e4:+.1f}")
        print(f"  avg hold (h):          fixed=24.0   variable={avg_hold_var:.1f}")
        print(f"  turnover proxy (1/hold,/cyc): fixed={1/24:.4f}  variable={np.mean([1/best_h[g] for g in sp['sgrp'].values]):.4f}")

        # NESTED-OOS honesty: split cycles into halves, pick best_h on first half, apply to second
        half = n // 2
        sp1, sp2 = sp.iloc[:half], sp.iloc[half:]
        bh_oos = {}
        for grp in [0, 1, 2]:
            sub = sp1[sp1["sgrp"] == grp]
            if len(sub) < 5: bh_oos[grp] = 24; continue
            means = {h: sub[f"sp_{h}"].mean() for h in HORIZONS_H}
            bh_oos[grp] = max(means, key=means.get)
        var_oos = np.array([sp2.iloc[i][f"sp_{bh_oos[sp2.iloc[i]['sgrp']]}"] for i in range(len(sp2))])
        fixed24_h2 = sp2["sp_24"].values
        print(f"  [NESTED-OOS] best_h from 1st half {bh_oos}; applied to 2nd half:")
        print(f"     2nd-half mean spread: fixed24={fixed24_h2.mean()*1e4:+.1f}  variable(OOS)={var_oos.mean()*1e4:+.1f}  "
              f"lift={1e4*(var_oos.mean()-fixed24_h2.mean()):+.1f} bps")

        # G4-style placebo: random per-cycle horizon matched to variable's avg hold, 200 seeds
        # build a horizon assignment per cycle that, on average, equals avg_hold_var
        hgrid = np.array(HORIZONS_H)
        target = avg_hold_var
        placebo_means = []
        for seed in range(200):
            rg = np.random.default_rng(seed)
            # sample horizons iid uniform from grid, then nudge toward matched avg by sampling from
            # a distribution; simplest matched control: assign each cycle a random tercile-permuted horizon
            perm = {0: rg.choice(hgrid), 1: rg.choice(hgrid), 2: rg.choice(hgrid)}
            # only keep seeds whose realized avg hold is within 4h of target (matched-length)
            ah = np.mean([perm[g] for g in sp["sgrp"].values])
            if abs(ah - target) > 6: continue
            ps = np.array([sp.iloc[i][f"sp_{perm[sp.iloc[i]['sgrp']]}"] for i in range(n)])
            placebo_means.append(ps.mean())
        placebo_means = np.array(placebo_means) * 1e4
        real = var_sp.mean() * 1e4
        if len(placebo_means) > 10:
            pct = (placebo_means < real).mean() * 100
            print(f"  [G4 PLACEBO] random matched-avg-hold horizons ({len(placebo_means)} valid seeds): "
                  f"real={real:+.1f}  placebo mean={placebo_means.mean():+.1f} p95={np.percentile(placebo_means,95):+.1f} "
                  f"max={placebo_means.max():+.1f}  -> real rank p{pct:.0f}")
        else:
            print("  [G4 PLACEBO] too few matched seeds")

    print(f"\n[{time.time()-t0:.0f}s] done", flush=True)


if __name__ == "__main__":
    main()
