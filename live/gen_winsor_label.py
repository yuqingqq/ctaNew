"""W1 training-label winsorization A/B (RESEARCH_LOOP_20260707 addendum 9, PRE-REGISTERED).

Tests: does truncating the TRAINING label's tails (per-cycle xs_z clipped +-2 instead of +-10)
improve the ordering the weights produce? "Fit the predictable middle, decide on the tail."

Isolation pins (addendum 9):
- ONLY the training target changes: xs_z built exactly as the incumbent (per-cycle z of
  alpha_vs_btc_realized), then clip(+-WINSOR_Z) instead of +-10. z first, then clip.
- Features (incl. resid_rev_2/3 from UNclipped alpha), two-book structure, WF cuts, RidgeCV,
  HL=60, embargo 1d, min 300 rows, EXCL — all frozen (gen_beta_label_ab pattern).
- Row availability unchanged by construction (clip preserves notna).
- Dose check printed: fraction of train-eligible rows with |xs_z| > WINSOR_Z.
- EVAL labels are never winsorized; scoring is the standard matched-population scorer vs the
  incumbent books.
Env: WINSOR_Z (default 2.0), WINSOR_TAG (default winz2).
Outputs: live/state/convexity/hl_<tag>_{base,long}[_oos]/v0full_hl60.parquet
"""
import os, sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.linear_model import RidgeCV
import warnings; warnings.filterwarnings("ignore")
REPO = Path("/home/yuqing/ctaNew"); sys.path.insert(0, str(REPO))
import live.train_twobook_models as tt
x6 = tt.x6; V0_LEAN = list(tt.V0_LEAN); EMB = pd.Timedelta(days=1); HL = 60.0
RR = ["resid_rev_2", "resid_rev_3"]
EXCL = {"LITUSDT", "VINEUSDT", "PUMPUSDT"}
WINSOR_Z = float(os.environ.get("WINSOR_Z", "2.0"))
TAG = os.environ.get("WINSOR_TAG", "winz2")

def main():
    PAN = pd.read_parquet(tt.PANEL, columns=["symbol", "open_time", "exit_time", "return_pct",
                                              "alpha_vs_btc_realized"] + V0_LEAN)
    PAN["open_time"] = pd.to_datetime(PAN["open_time"], utc=True)
    PAN["exit_time"] = pd.to_datetime(PAN["exit_time"], utc=True)
    PAN = PAN[(PAN.open_time.dt.hour % 4 == 0) & (PAN.open_time.dt.minute == 0)].sort_values(["symbol", "open_time"])
    a = PAN.groupby("symbol")["alpha_vs_btc_realized"]      # resid_rev from UNclipped alpha (pin)
    PAN["resid_rev_2"] = -a.transform(lambda s: s.shift(1).rolling(2).sum())
    PAN["resid_rev_3"] = -a.transform(lambda s: s.shift(1).rolling(3).sum())
    for c in RR: PAN[c] = PAN[c].fillna(0.0)
    g = PAN.groupby("open_time")
    sd = g["alpha_vs_btc_realized"].transform("std").replace(0, np.nan)
    z = (PAN["alpha_vs_btc_realized"] - g["alpha_vs_btc_realized"].transform("mean")) / sd
    PAN["xs_zw"] = z.clip(-WINSOR_Z, WINSOR_Z)              # z first, then clip (incumbent order)
    PAN["_clipped"] = (z.abs() > WINSOR_Z).astype(float)    # for per-fold weighted dose (9b-8)
    frac = (z.abs() > WINSOR_Z).mean()
    print(f"WINSOR_Z={WINSOR_Z} tag={TAG}; dose check: {frac:.4f} of rows clipped "
          f"(|z|>{WINSOR_Z}); NO-OP guard applies at scoring", flush=True)
    PAN = PAN.sort_values(["symbol", "open_time"]).reset_index(drop=True)

    def gen(cuts, feats, outpath, tagn):
        rec = []
        for i in range(len(cuts) - 1):
            c0, c1 = cuts[i], cuts[i + 1]; fc = c0 - EMB
            tr = PAN[(PAN.exit_time < fc) & PAN["xs_zw"].notna()]
            te = PAN[(PAN.open_time >= c0) & (PAN.open_time < c1) & (~PAN.symbol.isin(EXCL))]
            if not len(tr) or not len(te): continue
            t_end = tr["open_time"].max()
            wf = np.exp(-((t_end - tr["open_time"]).dt.total_seconds().to_numpy() / 86400.0) / HL)
            print(f"    {tagn} fold {i} weighted dose: "
                  f"{(wf * tr['_clipped'].to_numpy()).sum() / max(wf.sum(), 1e-9):.4f}", flush=True)
            n_tr, n_sym = 0, 0
            for sym, gg in tr.groupby("symbol"):
                if len(gg) < 300: continue
                try:
                    s, h = x6.fit_preproc(gg, feats); X = x6.apply_preproc(gg, feats, s, h)
                    w = np.exp(-((t_end - gg["open_time"]).dt.total_seconds().to_numpy() / 86400.0) / HL)
                    m = RidgeCV(alphas=x6.RIDGE_ALPHAS).fit(X, gg["xs_zw"].to_numpy(), sample_weight=w)
                    gte = te[te.symbol == sym]
                    if len(gte):
                        rec.append(pd.DataFrame({"symbol": sym, "open_time": gte["open_time"].values,
                            "alpha_A": gte["alpha_vs_btc_realized"].values, "return_pct": gte["return_pct"].values,
                            "exit_time": gte["exit_time"].values,
                            "pred": m.predict(x6.apply_preproc(gte, feats, s, h)), "fold": i}))
                    n_tr += len(gg); n_sym += 1
                except Exception:
                    pass
            print(f"    {tagn} fold {i} ({c0.date()}): {n_sym} syms, {n_tr} train rows", flush=True)
        out = pd.concat(rec, ignore_index=True)
        for c in ("open_time", "exit_time"): out[c] = pd.to_datetime(out[c], utc=True)
        Path(outpath).parent.mkdir(parents=True, exist_ok=True)
        out.to_parquet(outpath)
        print(f"  wrote {outpath}", flush=True)

    last = PAN["open_time"].max().normalize() + pd.Timedelta(days=1)
    REC_CUTS = [pd.Timestamp(t, tz="UTC") for t in ["2025-10-04", "2025-11-01", "2025-12-01", "2026-01-01",
                "2026-02-01", "2026-03-01", "2026-04-01", "2026-05-01", "2026-05-27"]] + [last]
    OOS_CUTS = list(pd.date_range("2023-01-01", "2025-10-01", freq="MS", tz="UTC"))
    D = REPO / "live/state/convexity"
    print("recent base:", flush=True); gen(REC_CUTS, V0_LEAN, D / f"hl_{TAG}_base/v0full_hl60.parquet", "rb")
    print("recent long:", flush=True); gen(REC_CUTS, V0_LEAN + RR, D / f"hl_{TAG}_long/v0full_hl60.parquet", "rl")
    print("oos base:", flush=True); gen(OOS_CUTS, V0_LEAN, D / f"hl_{TAG}_base_oos/v0full_hl60.parquet", "ob")
    print("oos long:", flush=True); gen(OOS_CUTS, V0_LEAN + RR, D / f"hl_{TAG}_long_oos/v0full_hl60.parquet", "ol")
    print("WINSORDONE")

if __name__ == "__main__":
    main()
