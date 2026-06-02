"""Autonomous feature-pruning study (funding-fixed panel). For each feature set: regenerate walk-forward
preds (flow book = v0set+flowset on flow-book syms; price book = v0set), run the monthly two-book backtest
(PIT dvol), log combined Sharpe/DD/PnL. Resumable: skips experiments already in results.csv.

Baseline = no-funding (V0 minus funding) + all 14 flow feats → known combined monthly +3.475.

Usage: CONVEXITY_PIT_DVOL=1 python3 live/prune_study.py [--only NAME]
"""
import sys, os, glob, json, argparse, time
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.linear_model import RidgeCV
import warnings; warnings.filterwarnings("ignore")
REPO = Path("/home/yuqing/ctaNew"); sys.path.insert(0, str(REPO))
import live.train_twobook_models as tt
import live.ab_split_rerank as ab
x6 = tt.x6; V0 = tt.V0
V0_NF = [f for f in V0 if not f.startswith("funding")]          # 14 (funding dropped = our new baseline)
EMB = pd.Timedelta(days=1); HL = 60.0
CUTS = [pd.Timestamp(t, tz="UTC") for t in ["2025-10-04","2025-11-01","2025-12-01","2026-01-01",
        "2026-02-01","2026-03-01","2026-04-01","2026-05-01","2026-05-27"]]
BASELINE = REPO/"research/convexity_portable_2026-05-20/results/_cache/x132_expanded_v0_preds.parquet"
WORK = REPO/"live/state/prune_study"; WORK.mkdir(parents=True, exist_ok=True)
RESULTS = WORK/"results.csv"
OOS0 = pd.Timestamp("2025-10-04", tz="UTC")

# ---- load panel + flow ONCE ----
print("loading panel + flow ...", flush=True)
F, FLOWCOLS = tt.build_flow(); FLOWSYMS = set(F.symbol.unique())
_last = pd.read_parquet(tt.PANEL, columns=["open_time"]); _last["open_time"] = pd.to_datetime(_last["open_time"], utc=True)
CUTS = CUTS + [_last["open_time"].max().normalize() + pd.Timedelta(days=1)]
PAN = pd.read_parquet(tt.PANEL, columns=["symbol","open_time","exit_time","return_pct"]+V0)  # rvol_7d is in V0
PAN["open_time"] = pd.to_datetime(PAN["open_time"], utc=True); PAN["exit_time"] = pd.to_datetime(PAN["exit_time"], utc=True)
PAN = PAN[(PAN.open_time.dt.hour%4==0)&(PAN.open_time.dt.minute==0)].merge(F, on=["symbol","open_time"], how="left")
_g = PAN.groupby("open_time"); _mu = _g["return_pct"].transform("mean"); _sd = _g["return_pct"].transform("std").replace(0,np.nan)
PAN["xs_z"] = ((PAN["return_pct"]-_mu)/_sd).clip(-10,10); PAN = PAN.sort_values(["symbol","open_time"]).reset_index(drop=True)
PRVOL = PAN[["symbol","open_time","rvol_7d"]].copy()
BL = pd.read_parquet(BASELINE); BL["open_time"] = pd.to_datetime(BL["open_time"], utc=True); BL["exit_time"] = pd.to_datetime(BL["exit_time"], utc=True)


def gen_preds(v0set, flowset, tag):
    """Walk-forward per-sym RidgeCV; flow book uses v0set+flowset (where flow populated), price book v0set."""
    def gen(use_flow, outpath):
        rec = []
        for i in range(len(CUTS)-1):
            c0, c1 = CUTS[i], CUTS[i+1]; fit_cut = c0-EMB
            tr = PAN[(PAN.exit_time < fit_cut) & PAN["xs_z"].notna()]; te = PAN[(PAN.open_time>=c0)&(PAN.open_time<c1)]
            t_end = tr["open_time"].max()
            for sym, g in tr.groupby("symbol"):
                if len(g) < 300: continue
                uf = use_flow and (sym in FLOWSYMS) and (len(flowset)>0) and g[flowset].notna().any().all()
                feats = v0set + flowset if uf else v0set
                if not feats: continue
                try:
                    s, h = x6.fit_preproc(g, feats); X = x6.apply_preproc(g, feats, s, h)
                    w = np.exp(-((t_end-g["open_time"]).dt.total_seconds().to_numpy()/86400.0)/HL)
                    m = RidgeCV(alphas=x6.RIDGE_ALPHAS).fit(X, g["xs_z"].to_numpy(), sample_weight=w)
                    gte = te[te.symbol==sym]
                    if len(gte): rec.append(pd.DataFrame({"symbol":sym,"open_time":gte["open_time"].values,
                                                          "pred_n":m.predict(x6.apply_preproc(gte,feats,s,h))}))
                except Exception: pass
        rec = pd.concat(rec, ignore_index=True); rec["open_time"] = pd.to_datetime(rec["open_time"], utc=True)
        oos = BL[BL.open_time>=CUTS[0]].copy().merge(rec, on=["symbol","open_time"], how="inner")
        oos["pred"] = oos["pred_n"]; oos.drop(columns=["pred_n"]).to_parquet(outpath); return len(oos)
    d = WORK/tag; d.mkdir(exist_ok=True)
    fp, pp = d/"flow.parquet", d/"price.parquet"
    gen(True, fp); gen(False, pp); return fp, pp


def backtest(fp, pp, tag):
    ff = pd.read_parquet(fp); v0 = pd.read_parquet(pp)
    for x in (ff, v0): x["open_time"] = pd.to_datetime(x["open_time"], utc=True)
    ff = ff[ff.open_time>=OOS0]; v0 = v0[v0.open_time>=OOS0]
    oos = sorted(set(ff.symbol.unique())); times = [pd.Timestamp(t) for t in sorted(ff.open_time.unique())]
    memb = ab.build_membership("monthly", times, PRVOL, oos, 80)
    d = WORK/tag
    ba, bb, nA, nB = ab.write_books("m", memb, ff, v0, d)
    ca = ab.run_replay(ba, d/"stateA"); cb = ab.run_replay(bb, d/"stateB")
    summ = ab.combine(ca, cb, d/"combine")
    return summ, nA, nB


# ---- experiment battery (priority-ordered) ----
FLG = {"tfi":["fl_tfi","fl_tfi_1d"], "sv_z":["fl_sv_z","fl_sv_z_1d"], "vpin":["fl_vpin","fl_vpin_1d"],
       "kyle":["fl_kyle","fl_kyle_1d"], "aggr":["fl_aggr","fl_aggr_1d"], "lg_share":["fl_lg_share","fl_lg_share_1d"],
       "bs_imb":["fl_bs_imb","fl_bs_imb_1d"]}
def v0_drop(*names): return [f for f in V0_NF if f not in names]
EXPERIMENTS = []
EXPERIMENTS.append(("baseline_nofund", V0_NF, FLOWCOLS))                       # known +3.475 (validation)
EXPERIMENTS.append(("no_flow", V0_NF, []))                                      # does flow add net value?
EXPERIMENTS.append(("readd_funding_rate", V0_NF+["funding_rate"], FLOWCOLS))    # keep only the +IC funding feat
for k,pair in FLG.items(): EXPERIMENTS.append((f"drop_flow_{k}", V0_NF, [c for c in FLOWCOLS if c not in pair]))
EXPERIMENTS.append(("flow_vpin_tfi_only", V0_NF, FLG["vpin"]+FLG["tfi"]))
EXPERIMENTS.append(("flow_vpin_only", V0_NF, FLG["vpin"]))
for f in V0_NF: EXPERIMENTS.append((f"drop_v0_{f}", v0_drop(f), FLOWCOLS))
EXPERIMENTS.append(("drop_idio_both", v0_drop("idio_vol_to_btc_1h","idio_vol_to_btc_1d"), FLOWCOLS))

EXPERIMENTS.append(("drop_bsh_pair", v0_drop("bars_since_high","bars_since_high_xs_rank"), FLOWCOLS))
# --- combined lean candidates (validation of stacking the individually-good prunes) ---
EXPERIMENTS.append(("lean_vpintfi", V0_NF, FLG["vpin"]+FLG["tfi"]))                                   # robust flow prune
EXPERIMENTS.append(("lean_norvol_vpintfi", v0_drop("rvol_7d"), FLG["vpin"]+FLG["tfi"]))               # + drop rvol_7d
EXPERIMENTS.append(("lean_norvol_nobsh_vpintfi", v0_drop("rvol_7d","bars_since_high","bars_since_high_xs_rank"), FLG["vpin"]+FLG["tfi"]))  # maximal lean


def load_done():
    if RESULTS.exists(): return set(pd.read_csv(RESULTS)["name"])
    return set()

def log_row(row):
    df = pd.DataFrame([row])
    df.to_csv(RESULTS, mode="a", header=not RESULTS.exists(), index=False)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--only", default=None); a = ap.parse_args()
    exps = [e for e in EXPERIMENTS if (a.only is None or e[0]==a.only)]
    done = load_done()
    for name, v0set, flowset in exps:
        if name in done: print(f"[skip] {name}", flush=True); continue
        t0 = time.time()
        try:
            fp, pp = gen_preds(v0set, flowset, name)
            summ, nA, nB = backtest(fp, pp, name)
            row = {"name":name, "n_v0":len(v0set), "n_flow":len(flowset),
                   "sharpe":round(summ["sharpe_both_active"],4), "totPnL":round(summ["totPnL_both_active"],1),
                   "maxDD":round(summ["maxDD_both_active"],1), "sharpe_A":round(summ["sharpe_bookA"],4),
                   "sharpe_B":round(summ["sharpe_bookB"],4), "secs":round(time.time()-t0)}
            log_row(row); print(f"[done] {name}: Sharpe {row['sharpe']:+.3f} DD {row['maxDD']:.0f} ({row['secs']}s)", flush=True)
        except Exception as e:
            print(f"[FAIL] {name}: {e}", flush=True)
    print("PRUNE STUDY COMPLETE", flush=True)

if __name__ == "__main__": main()
