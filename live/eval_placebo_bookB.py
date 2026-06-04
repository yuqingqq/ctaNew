"""#173 Random-universe placebo for book-B-only. Is the rvol-selected low-vol set special, or would random
~94-symbol subsets do as well? Run book-B-only (baseline) on the REAL rvol-bottom-94 set + K random ~94 subsets
(seeded, PIT monthly membership), compare Sharpe percentile. Real rule should beat placebo p90+. Monthly cadence."""
import sys, subprocess, os; from pathlib import Path
import numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
REPO = Path("/home/yuqing/ctaNew"); sys.path.insert(0, str(REPO))
import live.train_twobook_models as tt
OOS = pd.Timestamp("2025-10-04", tz="UTC"); ann = np.sqrt(6*365); SETN = 94; NSEED = 12
PAN = pd.read_parquet(tt.PANEL, columns=["symbol","open_time","rvol_7d"]); PAN["open_time"]=pd.to_datetime(PAN["open_time"],utc=True)
PAN = PAN[(PAN.open_time.dt.hour%4==0)&(PAN.open_time.dt.minute==0)]
PREDS = pd.read_parquet("live/state/convexity/hl/v0full_hl60.parquet"); PREDS["open_time"]=pd.to_datetime(PREDS["open_time"],utc=True)
PREDS = PREDS[PREDS.open_time>=OOS]
anchors = sorted({OOS} | {pd.Timestamp(t) for t in PREDS["open_time"].unique() if pd.Timestamp(t).day==1})
elig_by_anchor = {}
for a in anchors:
    lo=a-pd.Timedelta(days=30); q=PAN[(PAN.open_time>=lo)&(PAN.open_time<a)]
    elig_by_anchor[a] = q.groupby("symbol")["rvol_7d"].mean().dropna().sort_values()
def anchor_of(t): return max([a for a in anchors if a<=t], default=OOS)
PREDS["_anchor"] = PREDS["open_time"].map(anchor_of)

def run_membership(memb_by_anchor, tag):
    # vectorized filter: keep rows whose symbol is in that anchor's set
    keep_mask = PREDS.apply(lambda r: r["symbol"] in memb_by_anchor[r["_anchor"]], axis=1)
    bp = PREDS[keep_mask]
    out = REPO/f"live/state/opt_loop/placebo_{tag}.parquet"; bp.to_parquet(out)
    sd = REPO/f"live/state/opt_loop/placebo_{tag}_st"; sd.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "PYTHONPATH": str(REPO), "CONVEXITY_PREDS_PATH": str(out),
           "CONVEXITY_STATE": str(sd), "STRAT_K": "3", "SIDE_MODE": "default"}
    r = subprocess.run([sys.executable,"-m","live.convexity_paper_bot","--replay-all"], env=env, cwd=str(REPO), capture_output=True, text=True)
    if r.returncode != 0: return np.nan
    c = pd.read_csv(sd/"cycles.csv"); p = c["pnl_bps"]/1e4
    return p.mean()/p.std()*ann if p.std()>0 else 0.0

# REAL: rvol bottom-94
real = run_membership({a: frozenset(s.nsmallest(SETN).index) for a,s in elig_by_anchor.items()}, "real")
print(f"REAL rvol-bottom-{SETN} book-B Sharpe = {real:+.3f}")
# PLACEBO: random 94-subsets of the eligible universe (seeded)
rng = np.random.default_rng(173); ph=[]
for k in range(NSEED):
    memb = {a: frozenset(rng.choice(s.index.values, size=min(SETN,len(s)), replace=False)) for a,s in elig_by_anchor.items()}
    sh = run_membership(memb, f"rand{k}"); ph.append(sh); print(f"  placebo {k}: {sh:+.3f}")
ph=np.array([x for x in ph if np.isfinite(x)])
print(f"\nplacebo mean {ph.mean():+.3f}  p90 {np.percentile(ph,90):+.3f}  max {ph.max():+.3f}")
print(f"REAL rank = {(ph<real).mean()*100:.0f}% of placebo  ({'PASS p90' if (ph<real).mean()>=0.9 else 'FAIL'})")
