"""iter5 — time-series (per-name) framing. Does flow lead a name's OWN forward return,
beyond that name's own trailing return + vol, in the time series (not cross-section)?

Per symbol, per era: residualize signed_pressure(t) on [1, tr_5m, tr_1h, rv_1h](t) over the
name's time series (OLS), then Spearman(residual, fwd_ret(t+h)). Aggregate across symbols;
both-era; symbol-clustered bootstrap CI on the mean TS-IC. A usable TS signal needs the mean
TS-IC same-sign CI-off-zero in BOTH eras and non-trivial magnitude.
"""
from __future__ import annotations
import glob
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from live.flow_harness import SLIM, CUT, HORIZONS

CTRL = ["tr_5m", "tr_1h"]
FEAT = "signed_pressure_5min"


def ts_ic_for_symbol(sym):
    d = pd.read_parquet(f"{SLIM}/{sym}.parquet",
                        columns=["bar_time", FEAT, "price", *CTRL, "fwd_5m", "fwd_1h"])
    d["bar_time"] = pd.to_datetime(d["bar_time"], utc=True)
    d = d.drop_duplicates("bar_time").sort_values("bar_time").set_index("bar_time")
    full = pd.date_range(d.index.min(), d.index.max(), freq="5min")
    ret = d["price"].reindex(full).pct_change(fill_method=None)
    d = d.join(pd.DataFrame({"rv_1h": ret.rolling(12, min_periods=8).std()}, index=full), how="left")
    out = {}
    for era, m in [("OOS", d.index < CUT), ("REC", d.index >= CUT)]:
        sub = d[m]
        for k in ["5m", "1h"]:
            g = sub[[FEAT, *CTRL, "rv_1h", f"fwd_{k}"]].dropna()
            if len(g) < 500:
                out[(era, k)] = np.nan
                continue
            X = np.column_stack([np.ones(len(g))] + [g[c].to_numpy() for c in (*CTRL, "rv_1h")])
            b = np.linalg.lstsq(X, g[FEAT].to_numpy(), rcond=None)[0]
            resid = g[FEAT].to_numpy() - X @ b
            out[(era, k)] = spearmanr(resid, g[f"fwd_{k}"].to_numpy()).correlation
    return out


def boot_ci(vals, n=2000, seed=7):
    v = np.array([x for x in vals if np.isfinite(x)])
    if len(v) < 10:
        return (np.nan, np.nan, np.nan)
    rng = np.random.default_rng(seed)
    b = [rng.choice(v, len(v), replace=True).mean() for _ in range(n)]
    return (v.mean(), *np.nanpercentile(b, [2.5, 97.5]))


def main():
    syms = sorted(p.split("/")[-1][:-8] for p in glob.glob(f"{SLIM}/*.parquet"))
    res = {}
    for i, s in enumerate(syms):
        try:
            res[s] = ts_ic_for_symbol(s)
        except Exception as e:
            res[s] = {}
        if (i + 1) % 40 == 0:
            print(f"  {i+1}/{len(syms)}", flush=True)
    print(f"\nper-name TS partial-IC(signed_pressure -> fwd | own tr_5m,tr_1h,rv_1h), {len(syms)} symbols")
    print("mean across symbols [symbol-bootstrap CI] | frac same-sign as mean | both-era?\n")
    for k in ["5m", "1h"]:
        row = {}
        for era in ("OOS", "REC"):
            vals = [res[s].get((era, k), np.nan) for s in syms if s in res]
            row[era] = boot_ci(vals)
        (oa, ol, ou), (ra, rl, ru) = row["OOS"], row["REC"]
        both = (np.sign(oa) == np.sign(ra) and (ol > 0 or ou < 0) and (rl > 0 or ru < 0))
        # fraction of symbols with same sign as the era mean
        fo = np.mean([np.sign(res[s].get(("OOS", k), np.nan)) == np.sign(oa)
                      for s in syms if np.isfinite(res[s].get(("OOS", k), np.nan))])
        fr = np.mean([np.sign(res[s].get(("REC", k), np.nan)) == np.sign(ra)
                      for s in syms if np.isfinite(res[s].get(("REC", k), np.nan))])
        print(f"  {k:>3}: OOS {oa:+.4f}[{ol:+.4f},{ou:+.4f}] ({fo:.0%} agree) | "
              f"REC {ra:+.4f}[{rl:+.4f},{ru:+.4f}] ({fr:.0%} agree) | {'BOTH' if both else 'no'}")
    print("\nITER5DONE", flush=True)


if __name__ == "__main__":
    main()
