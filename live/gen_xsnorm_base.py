"""Base-book generator testing CROSS-SECTIONAL normalization for level-signal features.
Modes:
  base  : all 14 V0_LEAN features per-symbol (winsor/rank-z) — the current scheme (my-pipeline baseline)
  armA  : corr_to_btc_1d -> cross-sectional (within-cycle pct-rank, PIT); btc_rvol_7d -> global-z; rest per-symbol
  armB  : armA + {idio_vol_to_btc_1d, idio_vol_to_btc_1h, rvol_7d, atr_pct} also cross-sectional
Cross-sectional = within-cycle percentile rank across contemporaneous names (PIT: no future, no per-symbol history).
Same RidgeCV + 60d time-decay fit + monthly walk-forward as gen_lean_wf_preds.
Usage: python3 live/gen_xsnorm_base.py <START> <END> <base|armA|armB> <outdir>
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.linear_model import RidgeCV
import warnings; warnings.filterwarnings("ignore")
REPO = Path("/home/yuqing/ctaNew"); sys.path.insert(0, str(REPO))
import live.train_twobook_models as tt
x6 = tt.x6; V0 = list(tt.V0)
V0_LEAN = [f for f in V0 if not f.startswith("funding")]
EMB = pd.Timedelta(days=1); HL = 60.0

START = pd.Timestamp(sys.argv[1], tz="UTC"); END = pd.Timestamp(sys.argv[2], tz="UTC")
MODE = sys.argv[3]; OUT = REPO / sys.argv[4]; OUT.mkdir(parents=True, exist_ok=True)
CUTS = list(pd.date_range(START, END, freq="MS", tz="UTC"))
VOL5 = ["idio_vol_to_btc_1d", "idio_vol_to_btc_1h", "rvol_7d", "atr_pct"]
if MODE == "base":
    XS_FEATS, GLOBAL_FEATS = [], []
elif MODE == "armA":
    XS_FEATS, GLOBAL_FEATS = ["corr_to_btc_1d"], ["btc_rvol_7d"]
elif MODE == "armB":
    XS_FEATS, GLOBAL_FEATS = ["corr_to_btc_1d"] + VOL5, ["btc_rvol_7d"]
elif MODE == "opt":
    # full principled scheme: (1) level feats -> cross-sectional; (2) btc_rvol -> global z (fix artifact);
    # (4) already-normalized feats -> global z (no per-symbol double-normalization); (3) move/scale -> per-symbol.
    XS_FEATS = ["corr_to_btc_1d"] + VOL5
    GLOBAL_FEATS = ["btc_rvol_7d", "obv_z_1d", "autocorr_pctile_7d", "bars_since_high_xs_rank"]
else:
    raise SystemExit(f"bad mode {MODE}")
PERSYM = [f for f in V0_LEAN if f not in XS_FEATS and f not in GLOBAL_FEATS]
FEAT_ORDER = PERSYM + XS_FEATS + GLOBAL_FEATS
print(f"MODE={MODE} CUTS {CUTS[0].date()}..{CUTS[-1].date()} ({len(CUTS)-1}) | persym {len(PERSYM)} xs {XS_FEATS} global {GLOBAL_FEATS}", flush=True)

PAN = pd.read_parquet(tt.PANEL, columns=["symbol", "open_time", "exit_time", "return_pct", "alpha_vs_btc_realized"] + V0)
PAN["open_time"] = pd.to_datetime(PAN["open_time"], utc=True); PAN["exit_time"] = pd.to_datetime(PAN["exit_time"], utc=True)
PAN = PAN[(PAN.open_time.dt.hour % 4 == 0) & (PAN.open_time.dt.minute == 0)].sort_values(["symbol", "open_time"])
# cross-sectional pct-rank (PIT: within-cycle across names), standardized to ~unit scale for Ridge
for f in XS_FEATS:
    r = PAN.groupby("open_time")[f].rank(pct=True)
    PAN[f + "__xs"] = ((r - r.mean()) / (r.std() or 1.0)).astype(float)
_g = PAN.groupby("open_time"); _sd = _g["return_pct"].transform("std").replace(0, np.nan)
PAN["xs_z"] = ((PAN["return_pct"] - _g["return_pct"].transform("mean")) / _sd).clip(-10, 10)
PAN = PAN.sort_values(["symbol", "open_time"]).reset_index(drop=True)

def build_X(df, sstats, hstats, gstats):
    cols = []
    if PERSYM:
        cols.append(x6.apply_preproc(df, PERSYM, sstats, hstats))
    for f in XS_FEATS:
        cols.append(np.nan_to_num(df[f + "__xs"].to_numpy().reshape(-1, 1), nan=0.0))
    for f in GLOBAL_FEATS:
        z = (df[f].to_numpy() - gstats[f][0]) / (gstats[f][1] or 1.0)
        cols.append(np.nan_to_num(z.reshape(-1, 1), nan=0.0))
    return np.hstack(cols) if cols else np.zeros((len(df), 0))

rec = []
for i in range(len(CUTS) - 1):
    c0, c1 = CUTS[i], CUTS[i + 1]; fit_cut = c0 - EMB
    tr = PAN[(PAN.exit_time < fit_cut) & PAN["xs_z"].notna()]; te = PAN[(PAN.open_time >= c0) & (PAN.open_time < c1)]
    if not len(tr) or not len(te): continue
    t_end = tr["open_time"].max()
    gstats = {f: (float(tr[f].mean()), float(tr[f].std())) for f in GLOBAL_FEATS}   # PIT: train-only global z
    for sym, g in tr.groupby("symbol"):
        if len(g) < 300: continue
        try:
            s, h = x6.fit_preproc(g, PERSYM) if PERSYM else ({}, {})
            X = build_X(g, s, h, gstats)
            w = np.exp(-((t_end - g["open_time"]).dt.total_seconds().to_numpy() / 86400.0) / HL)
            m = RidgeCV(alphas=x6.RIDGE_ALPHAS).fit(X, g["xs_z"].to_numpy(), sample_weight=w)
            gte = te[te.symbol == sym]
            if len(gte):
                rec.append(pd.DataFrame({"symbol": sym, "open_time": gte["open_time"].values,
                    "alpha_A": gte["alpha_vs_btc_realized"].values, "return_pct": gte["return_pct"].values,
                    "exit_time": gte["exit_time"].values, "pred": m.predict(build_X(gte, s, h, gstats)), "fold": i}))
        except Exception:
            pass
out = pd.concat(rec, ignore_index=True)
for c in ("open_time", "exit_time"): out[c] = pd.to_datetime(out[c], utc=True)
out.to_parquet(OUT / "v0full_hl60.parquet")
print(f"DONE {out['symbol'].nunique()} syms {len(out)} rows -> {OUT}", flush=True)
