"""FROZEN 2022-holdout preds generator (Q2, RESEARCH_LOOP_20260707). Committed diff of
gen_residual_target.py per design-review F11: 2022 CUTS only; silent `except: pass` replaced with
logged drops; per-fold trained-symbol counts logged; plus the pre-registered mechanical stale-print
rule (F9): a test row is dropped if the symbol's trailing-30d zero-return fraction > 0.15 (PIT).
Emits the residual-target two-book preds for 2022:
  live/state/convexity/hl_2022_res_base/v0full_hl60.parquet
  live/state/convexity/hl_2022_res_long/v0full_hl60.parquet
ONE-SHOT: refuses to overwrite existing outputs.
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.linear_model import RidgeCV
import warnings; warnings.filterwarnings("ignore")
REPO = Path("/home/yuqing/ctaNew"); sys.path.insert(0, str(REPO))
import live.train_twobook_models as tt
x6 = tt.x6; V0_LEAN = list(tt.V0_LEAN); EMB = pd.Timedelta(days=1); HL = 60.0
RR = ["resid_rev_2", "resid_rev_3"]
STALE_ZFRAC, STALE_WIN = 0.15, 180   # trailing-30d (180×4h) zero-return fraction, PIT

CUTS = [pd.Timestamp(f"2022-{m:02d}-01", tz="UTC") for m in range(1, 13)] + [pd.Timestamp("2023-01-01", tz="UTC")]

D = REPO / "live/state/convexity"
for name in ("hl_2022_res_base", "hl_2022_res_long"):
    if (D / name / "v0full_hl60.parquet").exists():
        raise SystemExit(f"ONE-SHOT: {name} already exists — refusing to regenerate")

PAN = pd.read_parquet(tt.PANEL, columns=["symbol", "open_time", "exit_time", "return_pct",
                                          "alpha_vs_btc_realized"] + V0_LEAN)
PAN["open_time"] = pd.to_datetime(PAN["open_time"], utc=True)
PAN["exit_time"] = pd.to_datetime(PAN["exit_time"], utc=True)
PAN = PAN[(PAN.open_time.dt.hour % 4 == 0) & (PAN.open_time.dt.minute == 0)].sort_values(["symbol", "open_time"])
a = PAN.groupby("symbol")["alpha_vs_btc_realized"]
PAN["resid_rev_2"] = -a.transform(lambda s: s.shift(1).rolling(2).sum())
PAN["resid_rev_3"] = -a.transform(lambda s: s.shift(1).rolling(3).sum())
for c in RR: PAN[c] = PAN[c].fillna(0.0)
# PIT stale-print flag: trailing-30d fraction of exactly-zero 4h returns, shifted 1 bar
z = PAN.groupby("symbol")["return_pct"].transform(
    lambda s: s.eq(0).shift(1).rolling(STALE_WIN, min_periods=60).mean())
PAN["stale_frac"] = z
g = PAN.groupby("open_time")
sd = g["alpha_vs_btc_realized"].transform("std").replace(0, np.nan)
PAN["z_res"] = ((PAN["alpha_vs_btc_realized"] - g["alpha_vs_btc_realized"].transform("mean")) / sd).clip(-10, 10)
PAN = PAN.sort_values(["symbol", "open_time"]).reset_index(drop=True)
print(f"panel rows={len(PAN)} syms={PAN.symbol.nunique()}", flush=True)

def gen(feats, outdir):
    rec, dropped = [], []
    for i in range(len(CUTS) - 1):
        c0, c1 = CUTS[i], CUTS[i + 1]; fc = c0 - EMB
        tr = PAN[(PAN.exit_time < fc) & PAN["z_res"].notna()]
        te = PAN[(PAN.open_time >= c0) & (PAN.open_time < c1)]
        t_end = tr["open_time"].max()
        n_trained = 0
        for sym, gg in tr.groupby("symbol"):
            if len(gg) < 300: continue
            try:
                s, h = x6.fit_preproc(gg, feats)
                X = x6.apply_preproc(gg, feats, s, h)
                w = np.exp(-((t_end - gg["open_time"]).dt.total_seconds().to_numpy() / 86400.0) / HL)
                m = RidgeCV(alphas=x6.RIDGE_ALPHAS).fit(X, gg["z_res"].to_numpy(), sample_weight=w)
                gte = te[(te.symbol == sym) & (te["stale_frac"].fillna(0) <= STALE_ZFRAC)]
                if len(gte):
                    rec.append(pd.DataFrame({"symbol": sym, "open_time": gte["open_time"].values,
                        "alpha_A": gte["alpha_vs_btc_realized"].values, "return_pct": gte["return_pct"].values,
                        "exit_time": gte["exit_time"].values,
                        "pred": m.predict(x6.apply_preproc(gte, feats, s, h)), "fold": i}))
                n_trained += 1
            except Exception as e:
                dropped.append((str(c0.date()), sym, repr(e)[:80]))
        print(f"  fold {i} cut {c0.date()}: trained {n_trained} syms "
              f"(train hist {((fc - tr['open_time'].min()).days if len(tr) else 0)}d)", flush=True)
    if dropped:
        print(f"  DROPPED {len(dropped)} sym-folds (logged, not silent):")
        for d in dropped[:20]: print("   ", d)
    out = pd.concat(rec, ignore_index=True)
    for c in ("open_time", "exit_time"): out[c] = pd.to_datetime(out[c], utc=True)
    outdir.mkdir(parents=True, exist_ok=True)
    out.to_parquet(outdir / "v0full_hl60.parquet")
    return out["symbol"].nunique(), len(out)

print("res base", gen(V0_LEAN, D / "hl_2022_res_base"), flush=True)
print("res long", gen(V0_LEAN + RR, D / "hl_2022_res_long"), flush=True)
print("GEN2022DONE")
