"""Compare low-vol symbol-selection RULES for book-B-only (user: try both). PIT monthly membership, matched
set-size (~94) so we compare SELECTION QUALITY not breadth. Rules:
  A) rank      : bottom-94 by trailing-30d rvol_7d        (CURRENT relative-rank baseline)
  B) abs_rvol  : {trailing rvol_7d < X_fixed}, X calibrated ONCE at OOS start to ~94 names, then FROZEN (absolute)
  C) multifac  : top-94 by composite = z(-idio_vol_to_btc_1h) + z(corr_to_btc_1d)  (mean-reversion-favorable)
Each → filter v0full preds to membership → bot --replay-all → Sharpe / maxDD / avg-size / churn. Read-only of panel; PIT.
"""
import sys, subprocess, json; from pathlib import Path
import numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
REPO = Path("/home/yuqing/ctaNew"); sys.path.insert(0, str(REPO))
import live.train_twobook_models as tt
OOS = pd.Timestamp("2025-10-04", tz="UTC"); ann = np.sqrt(6*365); SETN = 94
PAN = pd.read_parquet(tt.PANEL, columns=["symbol","open_time","rvol_7d","idio_vol_to_btc_1h","corr_to_btc_1d"])
PAN["open_time"] = pd.to_datetime(PAN["open_time"], utc=True)
PAN = PAN[(PAN.open_time.dt.hour%4==0)&(PAN.open_time.dt.minute==0)]
PREDS = pd.read_parquet("live/state/convexity/hl/v0full_hl60.parquet")
PREDS["open_time"] = pd.to_datetime(PREDS["open_time"], utc=True)
syms_all = sorted(PREDS.symbol.unique())
times = sorted(PREDS[PREDS.open_time>=OOS]["open_time"].unique())
anchors = sorted({OOS} | {pd.Timestamp(t) for t in times if pd.Timestamp(t).day==1})

def trailing(asof, win=30):
    lo = asof - pd.Timedelta(days=win); q = PAN[(PAN.open_time>=lo)&(PAN.open_time<asof)]
    g = q.groupby("symbol")
    return pd.DataFrame({"rvol": g["rvol_7d"].mean(), "idio": g["idio_vol_to_btc_1h"].mean(), "corr": g["corr_to_btc_1d"].mean()})

# calibrate absolute rvol threshold ONCE at OOS start to ~SETN names, then freeze
t0 = trailing(OOS).dropna(subset=["rvol"]); X_ABS = float(t0["rvol"].nsmallest(SETN).max())
def zc(s): return (s - s.mean())/s.std()

def membership(rule):
    cache = {}
    for a in anchors:
        tr = trailing(a).dropna(subset=["rvol"])
        if rule == "rank":      sel = tr["rvol"].nsmallest(SETN).index
        elif rule == "abs_rvol":sel = tr.index[tr["rvol"] < X_ABS]
        elif rule == "multifac":
            tr2 = tr.dropna(subset=["idio","corr"]); score = -zc(tr2["idio"]) + zc(tr2["corr"])
            sel = score.nlargest(SETN).index
        cache[a] = frozenset(sel)
    def at(t):
        a = max([a for a in anchors if a <= t], default=OOS); return cache[a]
    return at, cache

def run(rule):
    at, cache = membership(rule)
    sizes = [len(s) for s in cache.values()]
    # churn: avg symbols changed between consecutive distinct anchor-sets
    sets = list(cache.values()); ch = np.mean([len(sets[i]^sets[i-1]) for i in range(1,len(sets))]) if len(sets)>1 else 0
    sub = PREDS[PREDS.open_time>=OOS].copy()
    keep = sub.apply(lambda r: r["symbol"] in at(r["open_time"]), axis=1)
    bp = sub[keep]
    out = REPO/f"live/state/opt_loop/lowvolrule_{rule}.parquet"; bp.to_parquet(out)
    sd = REPO/f"live/state/opt_loop/lowvolrule_{rule}_state"; sd.mkdir(parents=True, exist_ok=True)
    env = {**__import__("os").environ, "PYTHONPATH": str(REPO), "CONVEXITY_PREDS_PATH": str(out),
           "CONVEXITY_STATE": str(sd), "STRAT_K": "3", "SIDE_MODE": "default"}
    r = subprocess.run([sys.executable,"-m","live.convexity_paper_bot","--replay-all"], env=env, cwd=str(REPO), capture_output=True, text=True)
    if r.returncode != 0: print(f"  {rule} FAILED: {r.stderr[-800:]}"); return
    c = pd.read_csv(sd/"cycles.csv"); p = c["pnl_bps"]/1e4 if "pnl_bps" in c else c.filter(like="pnl").iloc[:,0]
    eq = p.cumsum(); dd = (eq-eq.cummax()).min()
    print(f"  {rule:9s}: Sharpe {p.mean()/p.std()*ann:+.3f}  maxDD {dd*1e4:+.0f}bps  set~{int(np.mean(sizes))}  churn {ch:.1f}/mo")

print(f"X_ABS rvol threshold (frozen at OOS start) = {X_ABS:.4f}")
for rule in ["rank","abs_rvol","multifac"]: run(rule)
